# Building a fraud detection system, file by file

We're sitting in front of an empty folder. Over the next few hours we're going to
build something that decides whether a card transaction is fraudulent, in under
50 milliseconds, and — more importantly — that can *prove* it isn't lying to us
about how well it works.

That second part is the whole job. Fraud detection is unusually good at producing
numbers that look excellent and mean nothing. The class is rare, so accuracy is
meaningless. The data is temporal, so a random split leaks. The labels arrive
weeks late, so anything using "was the last transaction fraud?" is untrainable in
production. And the adversary adapts, so a stationary holdout measures the one
regime guaranteed not to persist.

So the order we build things in is not arbitrary. We're going to build the
*measurement apparatus* before we build anything worth measuring, because
otherwise we won't be able to tell whether what we built works.

A note on how to read this: I'll explain why a piece has to exist before showing
it. When I quote code I'll give you the real file and line. When a line encodes a
decision rather than mechanics, I'll slow down and show you the tempting wrong
version too — the near-misses are where the learning is, and in this domain the
near-misses usually make your numbers *better*, which is why they survive review.

---

## Part 0: what forces the design

Before we type anything, four facts about the problem dictate most of the
structure.

**Fact one: the positive class is ~0.6% of rows.** That kills accuracy as a
metric, makes log-loss a bad early-stopping signal, and means every metric we
report needs a variance estimate, because a few hundred test positives is not a
lot to average over.

**Fact two: a transaction is not independently fraudulent.** A £3 streaming
payment is unremarkable. Nine of them in four minutes from an unfamiliar device
is card testing. The signal lives in the relationship between transactions, which
means our unit of prediction is "transaction plus its history", not "transaction".

**Fact three: labels lag.** A chargeback is confirmed weeks after the
authorisation. At scoring time we genuinely do not know whether last Tuesday's
purchase turned out to be fraud. Any feature built from prior labels is a feature
we cannot compute in production.

**Fact four: an authorisation request is bounded.** Whatever features we invent,
the live system has to compute them from a payload that arrives over HTTP. If we
build features that need a customer's entire history, we've built something we
cannot deploy — or worse, something we deploy *differently* from how we trained
it, which fails silently.

Fact four is the one that bites hardest and it bit this project mid-build. We'll
get there.

---

## Part 1: the contract layer

### `src/data/schema.py` — build this first, it's 100 lines and everything imports it

**Why it exists.** Every module downstream needs to agree on what a transaction
*is*. Without a single definition, `simulate.py` writes `merchant`, `features.py`
reads `merchant_id`, and you find out at runtime three hours later. More
importantly, we want synthetic and real datasets to be interchangeable: a real
dataset gets adapted *into* this schema by its loader, and nothing downstream
knows or cares which it's looking at.

**Why it's separate.** It has zero dependencies. Everything else has to import it,
so if it lived inside (say) the simulator, the serving code would import the
simulator to score a transaction. That's how you end up shipping numpy's RNG to
production.

The core of it is a class of string constants, and one distinction that carries
real weight:

```python
# src/data/schema.py:52-74 (abridged)
class TransactionSchema:
    """Column names for the canonical transaction table.

    A row is one authorisation request. Everything in ``REQUEST_FIELDS`` is known
    at scoring time; everything in ``LABEL_FIELDS`` arrives later (a chargeback
    is typically confirmed weeks after the authorisation) and must never be used
    as a feature, including for prior transactions in a sequence.
    """
    TXN_ID = "txn_id"
    CUSTOMER_ID = "customer_id"
    TIMESTAMP = "timestamp"
    ...
    IS_FRAUD = "is_fraud"
    FRAUD_TYPE = "fraud_type"
```

That split between `REQUEST_FIELDS` and `LABEL_FIELDS` isn't documentation. It's
the contract that Fact three imposes, written down in the one place every module
reads. When we get to the leakage tests we'll assert it mechanically.

Then three integers that quietly constrain every embedding table we'll ever
build:

```python
# src/data/schema.py:100-102
PAD_INDEX = 0
UNK_INDEX = 1
N_RESERVED_CODES = 2
```

**Why two reserved indices and not zero.** Padding needs an index because our
sequences are variable-length and we're going to pad them into a rectangle.
Unknown needs a *separate* index because "a merchant I have never seen before" is
genuinely informative — first-time merchants are riskier — and collapsing it into
padding would teach the model that unfamiliar merchants and non-existent
transactions are the same thing.

The tempting wrong version is to use `0` for both, or to map unknowns to `-1`.
The `-1` version blows up the moment it hits `nn.Embedding`, which is at least
loud. The shared-`0` version trains fine and quietly wastes a genuinely useful
feature. Guess which one survives code review.

`FraudType` is a `StrEnum` — an enum whose members *are* strings, so
`FraudType.CARD_TESTING == "card_testing"` is `True`. That means we can write it
into a pandas column and compare against raw strings without conversion, while
still getting an authoritative list of valid values.

### `src/config.py` — every number that changes a reported result

**Why it exists.** Scatter `0.005` across six files and you have six things to
change and five you'll forget. More subtly: when the write-up says "we use a 0.5%
review capacity", the reader needs to be able to check that claim against one
place.

**Why it's separate rather than constants at the top of each module.** Because
the README quotes these values. A config module is the interface between the
prose and the code, and it's the only way the two can be kept honest.

It's a set of frozen dataclasses. Two constants sit at module level because
they're used everywhere:

```python
# src/config.py:23-27
SEEDS: tuple[int, ...] = (0, 1, 2)

#: Review capacities a real fraud team can actually work: fraction of the
#: scored population an analyst queue can absorb per day.
REVIEW_CAPACITIES: tuple[float, ...] = (0.001, 0.005, 0.01)
```

`SEEDS` being a tuple of three is a claim: every number we report is a mean over
three runs. It's the minimum defensible number and — spoiler — for one of our
headline metrics it turns out not to be enough. We'll report that rather than
hide it.

`REVIEW_CAPACITIES` encodes something about the world, not about the model. A
fraud team has N analysts, each of whom can work M alerts a day. That fixes the
alert budget as a *fraction of volume*, and the only precision that matters is
precision at that fraction. Reporting precision at a 0.5 probability threshold —
the default everyone reaches for — answers a question nobody in the building has.

Now the values that took actual thought.

```python
# src/config.py:60-61
    drift_start_frac: float = 0.85
    unreported_fraud_rate: float = 0.04
```

`drift_start_frac = 0.85` puts the adversarial change in the last 15% of the
timeline. Move it earlier and it lands inside the training window, so the model
learns the drifted pattern and there's nothing to detect. Move it later and the
test set has too few drifted transactions to measure. It needs to land after
training ends and with enough test period left to be visible — with a 60/15/25
split, 0.85 puts it about two-thirds of the way through test.

`unreported_fraud_rate = 0.04` means 4% of fraudulent transactions carry a
*legitimate* label, because the victim never disputed them. This isn't noise for
its own sake. It's the reason real fraud models plateau short of perfect
separation, and without it this synthetic problem is solvable to PR-AUC 1.0. Set
it to 0 and every model scores identically well and the comparison measures
nothing.

```python
# src/config.py:81-83
    train_frac: float = 0.60
    val_frac: float = 0.15
    embargo_days: float = 1.0
```

The embargo is the interesting one. A transaction one second after the training
cutoff builds its rolling features from transactions the model trained on. One
day of gap makes that structurally impossible for our longest window (7 days...
which it doesn't fully cover — more on that when we hit the honest-limitations
section).

```python
# src/config.py:129-130
    focal_alpha: float = 0.25
    pos_sample_weight: float = 40.0
```

`focal_alpha = 0.25` is the RetinaNet default and it does something
counter-intuitive that we'll measure later: it weights the *positive* class at
0.25 and the negative at 0.75. People assume alpha up-weights the rare class. It
doesn't — it's there to counteract the `(1-p)^gamma` term, which already
up-weights hard examples, most of which are positives. Get this backwards and
your probabilities are systematically wrong in a direction that ranking metrics
won't show you.

`pos_sample_weight = 40.0` takes a 0.6% positive rate to roughly 18% in the
resampled stream. Not 50% — that's a much more aggressive distortion, and the
model ends up estimating a base rate two orders of magnitude away from reality.

```python
# src/config.py:153, 175
    false_positive_cost: float = 8.0
    latency_budget_p99_ms: float = 50.0
```

£8 for declining a good customer is a guess. It's *documented* as a guess, and
we'll sweep it from £2 to £30 rather than defend the specific number. The 50 ms
budget is derived: a card authorisation has a few hundred milliseconds end to end
before the acquirer times out, shared between network, decisioning and the
issuer's core. The model is one hop inside that.

---

## Part 2: the world — `src/simulate.py`

**Why this comes before any model.** Two reasons, one obvious and one not.

The obvious one: we need data.

The non-obvious one: **no public fraud dataset tells you which attack produced a
given fraud.** IEEE-CIS has `isFraud`, a binary. That means you cannot ask the
single most useful question about a fraud model — "which attack types is it blind
to?" — because an adversary finds the blind spot first and aggregate metrics hide
it completely. If we generate the data, we know the mechanism, and
`src/evaluation.py` can break recall down by attack type.

**Why it's one file and not a package.** It's ~900 lines but it's one coherent
thing with a single entry point, and splitting it would mean the merchant
universe, the customer universe and the fraud injectors live apart while being
tightly coupled through the RNG. One file, clearly sectioned.

### The universe: power laws, because that's the point

```python
# src/simulate.py:160-174
def _power_law_weights(n: int, exponent: float, rng: np.random.Generator) -> np.ndarray:
    ranks = np.arange(1, n + 1, dtype=np.float64)
    weights = ranks ** (-exponent)
    rng.shuffle(weights)
    return weights / weights.sum()
```

Three lines, and the middle one is the entire justification for using embeddings
at all. `ranks ** (-exponent)` is Zipf: merchant #1 gets weight 1, #2 gets
2^-0.95, #100 gets 100^-0.95. A handful of merchants carry most volume and a long
tail is seen a few times each.

If we'd sampled merchants uniformly, one-hot encoding would work fine, target
encoding would be stable, and the "learned embeddings help with high-cardinality
categoricals" claim would be untestable. The `rng.shuffle` matters too: without
it, merchant ID would correlate with popularity, and the model could learn
"low ID = popular" from the integer code alone.

### Traffic: a thinned Poisson process

```python
# src/simulate.py:330
    keep = rng.random(size=n_candidates) < accept
```

We draw twice as many candidate transaction times as we need, uniformly, then
accept each with a probability that depends on hour-of-day and day-of-week. That
gives realistic clustering — people shop in the evening, more at weekends —
without simulating an actual point process.

The tempting wrong version is to draw times uniformly and be done. You get a
model that can't use hour-of-day, because there's no diurnal signal to use, and
then "3am transaction" isn't suspicious and one of our fraud mechanisms becomes
undetectable for reasons that have nothing to do with the model.

### The part that got rebuilt: making fraud hard enough to be worth detecting

Here's the thing I'd tell you before you write the fraud injectors, because I
didn't know it when I wrote them the first time.

**The first version of this simulator produced a problem where the LightGBM
baseline scored PR-AUC 1.0000 on the pre-drift test period.** Perfect. Which
means every model scores perfect, and the entire comparison — the whole point of
the project — measures nothing.

The bug wasn't in the fraud. It was in the *legitimate* traffic being too clean.
Every fraudulent transaction used a device the customer had never used. Every
legitimate transaction used the customer's one device. So "unseen device" was a
perfect classifier, and no amount of clever modelling was going to be measured
against that.

The fix is a list of confounders, each of which exists to blunt exactly one
give-away:

- Customers hold 2–4 devices with lopsided usage, and ~5% of online transactions
  come from a device never seen before — a new browser, a borrowed laptop. Now
  "unseen device" is evidence, not proof.
- ~10% of customers make legitimate bursts of small purchases inside an hour.
  That's what card testing has to hide behind.
- ~2% of legitimate purchases are large one-offs; travel produces foreign
  countries and unfamiliar merchants.
- Attackers reuse the victim's own device 15–25% of the time (malware, session
  hijack).

And the device fingerprint:

```python
# src/simulate.py:467
    if drifted or rng.random() > 0.35:
        return str(rng.choice(np.array(DEVICE_TYPES), p=_DEVICE_TYPE_P))
    return str(rng.choice(np.array(FRAUD_DEVICE_TYPES), p=_FRAUD_DEVICE_P))
```

Read that condition carefully. Pre-drift, only 35% of fraud sessions draw from
the "automation fingerprint" pool (headless clients, emulators); the other 65%
look like ordinary consumer devices. Post-drift, *none* do.

The tempting version is `if drifted: consumer else: fraud_pool` — a clean
separation. That gives you a feature that's 100% predictive pre-drift and 0%
post-drift, which is a step function, not a signal. 35% gives you something a
model has to *combine* with other evidence, which is the situation we're
actually trying to study.

### Adversarial drift: each mechanism attacks its own defence

The drift isn't "the distribution shifts". It's targeted:

- Card testing raises amounts into the normal range and stretches the burst over
  hours instead of minutes — defeating velocity-plus-small-amount rules.
- Account takeover mimics the victim's own amount distribution and reuses a known
  device — defeating "new device plus big amount".
- Merchant compromise fans out across long-tail merchants with a longer delay —
  eroding any merchant-risk-rate feature.

That's what makes the monitoring section testable rather than decorative. Random
noise injected into the test period would drift the data; it wouldn't attack the
model.

### Label noise, applied last

```python
# src/simulate.py:820 (context)
        hidden = np.zeros(len(df), dtype=bool)
        hidden[rng.choice(fraud_rows, size=n_unreported, replace=False)] = True
        df.loc[hidden, S.IS_FRAUD] = np.int8(0)
        df.loc[hidden, S.FRAUD_TYPE] = FraudType.NONE.value
```

Note this runs *after* concatenation and sorting, on the final frame. It has to:
we're flipping a random 4% of all fraud, and doing it per-mechanism during
generation would change the relative mechanism mix.

Note also the `df.loc[boolean_array, col]` rather than `df.iloc[integer_array,
col_position]`. We reset the index just above this. The `.iloc` version with a
positional column index is what I wrote first, and mypy rejected it — correctly,
because `iloc` with a `(row_array, column_int)` tuple is not a supported
signature and pandas' behaviour there is version-dependent.

### `src/data/loaders.py` — the seam that keeps real data optional

**Why it's separate from the simulator.** Because "where does a DataFrame come
from" and "how do we fabricate one" are different jobs, and everything downstream
should depend only on the first. `load_dataset("synthetic")` and
`load_dataset("ieee-cis")` return the same columns; nothing after this file knows
which it got.

The interesting decision is what happens when the file isn't there:

```python
# src/data/loaders.py (in _require)
    missing = [p for p in paths if not p.exists()]
    if not missing:
        return
    instructions = DOWNLOAD_INSTRUCTIONS[dataset].format(raw=paths[0].parent.resolve())
    listing = "\n".join(f"  - {p}" for p in missing)
    raise DatasetNotAvailableError(
        f"Missing files for dataset '{dataset}':\n{listing}\n{instructions}"
    )
```

No download, no credential prompt, no cache. The error message contains the URL,
the filename and the absolute path to drop it in. A library that tries to
authenticate to Kaggle on your behalf is a library that fails in CI, fails behind
a proxy, and stores a token you didn't ask it to store.

The two real-data adapters both lose something, and both say so on the loader
rather than in a footnote. IEEE-CIS has no cardholder key at all, so
`customer_id` becomes a `card1/card2/addr1` composite — which over-merges (two
cards sharing a BIN and postcode collide) and under-merges (a reissued card looks
new). ULB is worse for our purposes: 28 anonymised PCA components and no entity
identifiers whatsoever, so per-customer sequences cannot be constructed. It maps
every row to one pseudo-customer and logs a warning saying the sequence models
degenerate to a global stream.

That last one is worth including precisely *because* it's the dataset everyone
reaches for. Being explicit about why it cannot answer this project's question is
more useful than quietly running on it and reporting a number.

---

## Part 3: time — `src/data/splits.py`

**Why it's its own file** (it's 130 lines and does one thing): because the split
is a *claim* the whole project rests on, and it needs to be somewhere a reviewer
can find it in five seconds. Buried at the bottom of a training script, nobody
checks it.

```python
# src/data/splits.py:93-101
    train_end = t0 + span * cfg.train_frac
    val_start = train_end + embargo
    val_end = t0 + span * (cfg.train_frac + cfg.val_frac)
    test_start = val_end + embargo

    return TemporalSplit(
        train=(ts <= train_end).to_numpy(),
        val=((ts >= val_start) & (ts <= val_end)).to_numpy(),
        test=(ts >= test_start).to_numpy(),
```

The tempting wrong version, and it is *extremely* tempting because scikit-learn
hands it to you:

```python
train, test = train_test_split(df, test_size=0.25, random_state=42)  # WRONG
```

This is the single most common way fraud papers report numbers they can't
reproduce. It puts a customer's Thursday in train and their Wednesday in test.
Every rolling feature we're about to build then has partial knowledge of the test
set through the training set's view of the same customer. Your PR-AUC goes up.
Everything looks better. Nothing works.

Note that the three masks don't cover every row — the embargo periods belong to
no split. Those transactions still *exist* in the frame and still appear in later
rows' history windows, which is correct: history is history. They're just never
scored or trained on.

Also note `span * cfg.train_frac` splits by **time**, not by row count. With
seasonal volume the two differ. Time is the right one: it's what a deployment
boundary actually looks like.

---

## Part 4: the hard part — `src/data/features.py`

This is the file where the project is won or lost, it's the longest one, and it's
the one that got substantially rewritten mid-build. I'll take it slowly.

**Why it's separate from `sequences.py`.** Features answer "what do we know about
this transaction at this instant". Windows answer "which transactions does the
model see". Those change for different reasons — you'd add a feature without
touching windowing, and change window length without touching features. And
critically, the serving path needs the feature code but builds its own windows.

### The single most important structural decision

```python
# src/data/features.py:202-212 (signature)
def customer_causal_features(
    ts_sec: np.ndarray,
    amount: np.ndarray,
    merchant_key: np.ndarray,
    ...
    lookback: int = DEFAULT_LOOKBACK,
) -> dict[str, np.ndarray]:
```

**This function takes numpy arrays, not a DataFrame.** That signature constrains
everything downstream, and it's deliberate.

The batch pipeline has a DataFrame and will hand us columns. The serving path has
a JSON payload of ~60 records and would pay 1–2 ms of pure pandas overhead to
build a DataFrame it immediately tears apart again. If this function took a
DataFrame, serving would either eat that cost or — far more likely — someone
would write a "fast path" that reimplements the features in numpy.

And that reimplementation is *the* classic production fraud bug. Two feature
implementations that agree on the happy path and disagree on the boundary
between "seen this merchant" and "seen this merchant within the window". The
offline metrics stay excellent. The deployed model degrades. Nothing alerts.

One implementation, called by both paths, verified equal by a test. That's the
whole design and it starts with this signature.

### Timestamps: the one-line fix for a bug you'd never find

```python
# src/data/features.py:150-159
def to_epoch_seconds(nanoseconds: np.ndarray) -> np.ndarray:
    """Convert integer nanosecond timestamps to seconds at microsecond resolution."""
    return (np.asarray(nanoseconds, dtype=np.int64) // _NS_PER_US).astype(np.float64) / 1e6
```

pandas timestamps carry nanoseconds. JSON datetimes carry microseconds at best.
If the batch path does `df.timestamp.astype("int64") / 1e9` and the serving path
does `pd.Timestamp(record["timestamp"]).value / 1e9`, the two disagree in the
last few decimal places of every time delta.

Does it matter numerically? Not remotely — we're talking about nanoseconds inside
`log1p(seconds)`. Does it matter operationally? It's the difference between a
parity test that passes exactly and one that passes "within tolerance", and once
you're in tolerance-land you've lost the ability to detect a *real* divergence.
Truncate both to microseconds and the two paths agree bit-for-bit.

### The rewrite: bounded history

Here's the mistake, because it's more instructive than the fix.

Version one computed expanding statistics over each customer's entire history:
lifetime transaction count, tenure since first seen, mean spend since signup,
every merchant ever used. It used a Python loop with dictionaries tracking
last-seen times, and it worked, and the leakage tests passed.

Then I wrote the serving parity test and it failed, and it was right to.

**An authorisation request cannot carry an unbounded history.** So a real service
has two options: keep a stateful per-customer feature store — a second system
that will drift from the training code — or silently compute something different
from what the model was trained on. The second is the common failure and it's
invisible.

So every history-derived feature is now bounded:

```python
# src/data/features.py:147
DEFAULT_LOOKBACK = 31
```

**The cost is real and I'm not going to pretend otherwise.** We give up
long-horizon signal: true tenure, lifetime distinct merchants, "this customer has
never spent more than £50 in three years". On this data it costs little because
the injected mechanisms are short-horizon. On a real portfolio a customer's
multi-year baseline is genuinely informative, and the right answer there is a
feature store sharing this exact code path — not a shorter window.

The rewrite also made it ~5x faster by replacing the Python loop with sliding
windows, which was a happy accident rather than the goal.

### The window trick

```python
# src/data/features.py:167-181
def _window_view(values: np.ndarray, lookback: int, pad_value: Any) -> np.ndarray:
    """Matrix whose row ``i`` holds the ``lookback`` entries before position ``i``."""
    padded = np.concatenate([np.full(lookback, pad_value, dtype=values.dtype), values])
    view = np.lib.stride_tricks.sliding_window_view(padded, lookback)
    return view[: len(values)]
```

`sliding_window_view` gives you an `(n, k)` view over a 1-D array with no copy —
it's stride manipulation. Prepending `lookback` pad values means row `i` of the
view is `padded[i : i+k]`, which is original indices `i-k … i-1`. Exactly the
`lookback` entries *before* `i`, with padding where history runs out.

**The off-by-one to get wrong here:** if you pad with `lookback - 1` instead of
`lookback`, row `i` becomes `values[i-k+1 : i+1]` — which *includes the row
itself*. Every feature then sees the transaction it's describing. Your amount
z-score compares the amount to a mean that includes it, damping it toward zero.
Your "distinct merchants" count includes the current merchant, so `is_new` is
never true.

Would you catch it? The features would still be *shaped* right, still finite,
still correlated with fraud. PR-AUC would move by a bit. And this is the bad
part: including the current row makes the z-score *smaller* for anomalies, so it
would make the model slightly worse, not better — which means you'd notice it
eventually. The mirror-image error, using `lookback + 1`, silently drops the
oldest slot and nobody ever notices.

The trailing slice `view[: len(values)]` matters too: `sliding_window_view` over
an array of length `n + k` yields `n + 1` windows. Without the slice, you'd get
one extra row and every downstream assignment would misalign by shape, which at
least fails loudly.

### The line that enforces causality

```python
# src/data/features.py:262
    valid = np.isfinite(win_t) & (win_t < ts_sec[:, None])
```

This is the load-bearing line of the whole file. Two conditions:

`np.isfinite(win_t)` — window slots padded with `-inf` aren't real transactions.

`win_t < ts_sec[:, None]` — **strictly** less than. Not `<=`.

Why strict matters: two transactions can share a timestamp. With `<=`, a
transaction would count its own simultaneous twin as history — and in card
testing, bursts are exactly where near-simultaneous transactions cluster. The
model would get information from a transaction that, in a real system, is being
authorised in a different thread at the same moment.

The tempting wrong version is `win_t <= ts_sec[:, None]`, and it's tempting
because it "includes more history" and your velocity counts go up, which makes
the burst detection *better*. That's what makes it dangerous: the wrong version
scores higher.

Note also that this check is elementwise on the window matrix, not a slice
boundary. A slice-based version — "take everything before index i" — is correct
only if the array is sorted, and stays correct only as long as nobody introduces
a tie. This version is correct by construction.

### Amount z-score: two defensive constants

```python
# src/data/features.py:282-286
    prior_mean = np.where(has_hist, prior_mean, log_amount)
    # A floor on the denominator stops a customer with two near-identical
    # transactions from producing z-scores in the hundreds.
    amount_z = (log_amount - prior_mean) / (prior_std + 0.35)
    amount_vs_mean = log_amount - prior_mean
```

Line 282: with no history, set the prior mean to the *current* amount, making the
z-score exactly zero. The alternative — leaving it at 0.0 — makes a customer's
first transaction look like a massive positive anomaly, because `log1p(50) - 0`
is large. Every customer's first transaction would then be flagged. That's not a
subtle bug but it is an easy one, and it appears specifically in the rows you're
least likely to eyeball.

Line 285: the `+ 0.35` floor on the denominator. A customer with two prior
transactions of £20.00 and £20.01 has a prior standard deviation near zero, so a
£25 transaction produces a z-score in the hundreds. That single row then
dominates the batch's gradient. `0.35` is roughly the standard deviation of
`log1p(amount)` for a typical customer, so it says "never be more confident than
one typical customer's worth of spread". Raise it and the feature goes numb;
remove it and you get infinities.

The clip in `scale()` (±10σ) is the second line of defence for the same problem.
Belt and braces, because this is the failure mode that produces NaN gradients
forty minutes into a training run.

### Entity novelty, vectorised

```python
# src/data/features.py:313-320
    def entity(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (is_new, seconds since last seen, times seen, window codes)."""
        codes = pd.factorize(keys)[0].astype(np.int64)
        win_codes = _window_view(codes, k, -1)
        match = (win_codes == codes[:, None]) & valid
        seen = match.any(axis=1)
        last_seen = np.where(match, win_t, -np.inf).max(axis=1)
        since = np.where(seen, ts_sec - last_seen, NEVER_SECS)
        return (~seen).astype(np.float64), since, match.sum(axis=1).astype(np.float64), win_codes
```

`pd.factorize` turns arbitrary keys into integers so the comparison is a fast
integer op rather than object comparison. The codes are **local to this array** —
merchant "M001" might be code 3 for one customer and code 7 for another. That's
fine because every comparison is within a single call, and it's what lets the
serving path (which sees a different subset) produce identical results.

The `& valid` on line 317 is doing real work: without it, a match against a
padded slot would count. We pad codes with `-1` so a real code can't collide —
but `pd.factorize` returns `-1` for NaN, so if a key were ever NaN this would
silently break. Our keys are strings and never null, but that's an invariant held
by convention rather than assertion. I'd add a check if this saw real data.

`NEVER_SECS = 3.2e8` (about ten years) rather than NaN or infinity: it keeps the
tensors dense, lets a tree split on it cleanly, and survives `log1p` to a finite
number. NaN would propagate through the standardiser and poison a whole feature
column.

### Out-of-fold target encoding, and the leak the tests caught

Risk encodings are the only features that consume labels, so they get special
handling. The scheme is: block `k` is encoded using only blocks `< k`.

```python
# src/data/features.py:601-602
        ranks = train[S.TIMESTAMP].rank(method="first").to_numpy()
        block = np.floor((ranks - 1) / max(len(train) / self.n_blocks, 1)).astype(int)
```

`rank(method="first")` breaks ties by order of appearance, giving a total order.
The default `method="average"` gives *fractional* ranks for ties, and
`np.floor` on a fractional rank puts tied rows in whichever block the arithmetic
lands in — nondeterministically with respect to row order.

```python
# src/data/features.py:497-501
        prior_b = seen_pos / seen_n if seen_n > 0 else initial_prior
        k_arr = keys[mask]
        s = np.array([sum_y.get(k, 0.0) for k in k_arr], dtype=np.float64)
        c = np.array([cnt.get(k, 0.0) for k in k_arr], dtype=np.float64)
        out[mask] = (s + smoothing * prior_b) / (c + smoothing)
```

Here's the part worth stopping on. **The leakage test suite caught a real bug in
this function on its first run, and the bug was in the prior.**

Version one smoothed towards `self.prior` — the fraud rate over the *whole*
training window. Think about what that means for a row in block 0: its encoding
is smoothed towards a number computed partly from labels in blocks 1–4, which
arrive *after* it.

Is it a big leak? No. It's one scalar summarising ~1,100 labels; the influence of
any single future label is about 1/1100. You would never see it in a metric.

Is it a leak? Yes. And `test_training_risk_encoding_is_out_of_time_within_train`
flips the labels of the last training block and asserts the first block's
encodings don't move. Under version one, they moved.

The fix is `prior_b`: block `k` smooths towards the rate observed over blocks
`< k`, and block 0 smooths towards `initial_prior = 0.005` — a number fixed
before looking at the data at all.

The `smoothing = 50.0` pseudo-count is the usual bias/variance dial. A merchant
with 5 transactions and 1 fraud has an empirical rate of 20%; with smoothing 50
it reports `(1 + 50×0.005)/(5 + 50) ≈ 2.3%`. Without smoothing, every rare
merchant that happened to catch one fraud becomes a maximally-suspicious
merchant, and the model learns merchant IDs by heart.

### The serving path

```python
# src/data/features.py:773
            numeric[:, i] = causal[name] if name in causal else population[name]
```

`NUMERIC_FEATURES` is `CAUSAL_FEATURES + POPULATION_FEATURES`, and this line
fills the matrix in that exact order by looking each name up in whichever dict
has it. Column order is part of the contract with the model — the scaler's mean
vector and the model's input layer are both indexed by it — so building this by
`np.column_stack` of two dicts' `.values()` would work until someone reordered a
dict literal.

### What I'd change

The four risk encodings are the one place where the online and offline paths
*don't* agree exactly, and it's inherent: during training they're out-of-fold,
and an out-of-fold encoding is a property of the training procedure, not of the
transaction. For any window lying after the training period — every window a
deployed model actually sees — all 42 features match. Windows straddling the
boundary differ in those four columns. Both facts are tested. It's the right
trade but it's a wart, and if I were doing it again I'd consider dropping target
encoding entirely in favour of frequency encoding plus letting the embeddings
learn merchant risk, which would make parity unconditional.

---

## Part 5: windows — `src/data/sequences.py`

**Why separate.** See above — different reasons to change. Also this file imports
torch and `features.py` doesn't, which keeps the feature code usable in a context
with no deep learning framework installed.

### The lexsort trap

```python
# src/data/sequences.py:61-66
    codes, _ = pd.factorize(customer_ids)
    # Stable sort by (customer, time) gives contiguous, chronologically ordered
    # blocks per customer while preserving the caller's tie-breaking.
    order = np.lexsort((timestamps, codes))
    sorted_codes = codes[order]
    boundaries = np.flatnonzero(np.diff(sorted_codes)) + 1
```

`np.lexsort` sorts by the **last** key first. So `(timestamps, codes)` means
"primary key = codes, secondary = timestamps". That reads backwards to everyone
the first time.

The tempting wrong version is `np.lexsort((codes, timestamps))`, which sorts
primarily by *time* and secondarily by customer. Then `np.diff(sorted_codes)`
finds a boundary essentially everywhere, and `np.split` produces thousands of
one-element groups. Each customer's "history" becomes a single transaction.

Here's what makes it nasty: it doesn't crash. Every window is `[PAD, PAD, …, i]`,
the mask is valid, the model trains, and it learns a per-transaction classifier.
You get a number. It's just not the number you think you're measuring, and it's
*lower*, so you conclude sequence models don't help.

### Building the windows

```python
# src/data/sequences.py:68-73
    pad = np.full(max_len - 1, PAD_ROW, dtype=np.int64)
    for group in np.split(order, boundaries):
        padded = np.concatenate([pad, group])
        view = np.lib.stride_tricks.sliding_window_view(padded, max_len)
        windows[group] = view
    return windows
```

Same trick as `_window_view`, one difference: we pad by `max_len - 1`, not
`max_len`, because here the window is *inclusive* of the scored transaction. Row
`i`'s window is `[i-31 … i]`, 32 slots, with the scored transaction last.

That's the second off-by-one in this codebase and it's the opposite of the first.
In `_window_view` we want the rows strictly before `i`, so we pad by `k`. Here we
want `i` included, so we pad by `k-1`. Two functions, three lines apart in spirit,
opposite constants. If you're rebuilding this, write the test before the function.

**Why left-padded and not right-padded.** Because it means position `L-1` is
*always* the transaction being scored, for every row, regardless of history
length. The TCN reads `h[:, :, -1]`. The Transformer reads position `-1`. Neither
needs to know how much history this customer has. Right-padding would make the
prediction position vary per row and every architecture would need a gather.

### The gather

```python
# src/data/sequences.py:175-182
    win = data.windows[rows]
    valid = win >= 0
    safe = np.where(valid, win, 0)

    numeric = data.numeric[safe]
    numeric[~valid] = 0.0
    categorical = data.categorical[safe]
    categorical[~valid] = PAD_INDEX
```

`safe` replaces `-1` with `0` before indexing, because numpy would happily
interpret `-1` as "last row" and silently give you a real transaction from the
end of the dataset. Then we zero those slots out.

The critical detail: `data.numeric[safe]` is **fancy indexing**, which returns a
copy. So `numeric[~valid] = 0.0` mutates the copy. If you refactored this to
basic slicing — which returns a *view* — you'd be zeroing rows of the backing
array, corrupting the dataset progressively as training proceeded. Early batches
fine, later batches increasingly garbage, loss curve slowly rising. Good luck.

### Weighted sampling

```python
# src/data/sequences.py:245-246
            p = self.sample_weights / self.sample_weights.sum()
            order = self.rng.choice(self.rows, size=len(self.rows), replace=True, p=p)
```

`replace=True` is not optional. Sampling `len(rows)` items without replacement
from `len(rows)` items is just a permutation, and the weights do nothing. That's
a genuinely invisible bug: your "weighted sampling" arm trains fine, produces
sensible numbers, and is identical to the unweighted arm. You'd conclude that
resampling has no effect on calibration and write it up.

The test that catches it (`test_weighted_sampling_oversamples_positives`) draws
an epoch and asserts the positive rate is at least 5× the natural rate.

---

## Part 6: the leakage tests — write these now, not at the end

**Why here.** We have features and windows and no models. This is exactly the
right moment, because the tests are about the data pipeline and nothing else, and
because every model we build from here on will be evaluated with a metric we
can't yet trust.

**Why they're behavioural, not structural.** We could write a test that reads
`features.py` and asserts it never indexes past `i`. That test breaks on every
refactor and passes on any leak introduced through a library call. Instead:
**perturb the future, assert the past doesn't move.**

```python
# tests/test_leakage.py:74 (the core one)
@pytest.mark.parametrize("quantile", [0.3, 0.6, 0.85])
def test_causal_features_ignore_future_transactions(
    transactions: pd.DataFrame, quantile: float
) -> None:
    """Scrambling every transaction after t must not move any feature before t."""
```

It scrambles amount, merchant, MCC, device, device type, country, channel and
label for every transaction after a cut point, recomputes all features, and
asserts bit-identical equality (`rtol=0, atol=0`) for every row before the cut.

That catches leakage regardless of mechanism — including leakage introduced later
by someone who's never read this file.

The companion tests each catch something the others can't:

- **`test_causal_features_ignore_deleted_future_rows`** (line 93) *truncates* the
  future instead of perturbing it. Perturbation catches value leakage; truncation
  catches leakage through *counts*. A rolling window that accidentally looks
  forward changes size when later rows disappear but not when their values change.
- **`test_labels_never_enter_causal_features`** (line 112) flips every label and
  asserts every causal feature is unchanged. This is Fact three, mechanised.
- **`test_single_customer_features_are_prefix_stable`** (line 128) computes
  features for the first half of a customer's history and asserts they equal the
  first half of the full computation. This is the serving contract in miniature.
- **`test_windows_contain_only_own