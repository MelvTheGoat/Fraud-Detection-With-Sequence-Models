# Decisions

Every judgement call in this project, and the reasoning behind it. Where a
decision cost something, the cost is stated. Where the simpler option was taken
because the harder one was not worth it, that is stated too.

---

## 1. Framing

### 1.1 The thesis is a measurement, not a conclusion

The project asks whether sequence models beat gradient boosting on transaction
fraud, and reports the answer either way. Two things follow.

First, the LightGBM baseline is built to win: velocity windows, customer-relative
amount z-scores, entity novelty and recency, out-of-fold target encodings, and
native categorical splits on the high-cardinality columns. It sees the same
information the sequence models see, just flattened into one row. A weak baseline
would make the comparison meaningless and is obvious to any reader.

Second, the headline metric is expected cost per transaction, not PR-AUC. PR-AUC
ranks models; it does not tell you whether to deploy one, because it assumes a
false positive and a false negative are equally interesting and they are not.

### 1.2 Synthetic data first, real data optional

No public fraud dataset labels the *mechanism* behind each fraud, so no public
dataset can answer "which attack types does this model catch and which is it
blind to?" — the question that matters most, because an adversary finds the
blind spot first. The simulator knows the mechanism, so
`src/evaluation.py` can break recall down by attack type.

Real datasets are supported behind `--dataset` and are never downloaded
automatically. Both adaptations lose something material, documented on each
loader: IEEE-CIS has no true cardholder or merchant key (proxies are used), and
the ULB dataset has no entity identifiers at all, so per-customer sequences
cannot be constructed from it and it can only exercise the tabular baseline.

---

## 2. The simulator

### 2.1 Confounders are as important as the fraud

The first version of the simulator produced fraud that was trivially separable —
baselines hit PR-AUC 1.0000 on the pre-drift test period. That makes the whole
architecture comparison vacuous: every model scores 1.0 and there is nothing to
measure. The fix was not to weaken the models but to make the world realistic:

- Customers hold a pool of 2–4 devices with lopsided usage, plus a ~5% chance of
  transacting from a device never seen before (new browser, borrowed laptop). Phone
  upgrades retire a device mid-period. Without this, "unseen device" alone is
  a near-perfect rule.
- ~10% of customers make legitimate bursts of small purchases inside an hour —
  in-app top-ups, a transit run. This is what card testing hides behind.
- A heavy amount tail: ~2% of legitimate purchases are large one-offs, and
  travel periods produce foreign countries and unfamiliar merchants.
- Attackers reuse a victim's own device 15–25% of the time (malware, session
  hijack), so device novelty is evidence rather than proof.
- The automation fingerprint (headless clients, emulators) is present in only
  about a third of pre-drift fraud sessions.

After these changes the baseline scores ~0.95 PR-AUC pre-drift and ~0.70
post-drift, which leaves room for the comparison to say something.

### 2.2 Unreported fraud

4% of fraudulent transactions carry a legitimate label, because the victim never
disputed them. This is realistic and it is also what stops the problem being
cleanly separable. Every real fraud label set has these, and models that plateau
short of perfect separation are usually hitting exactly this ceiling.

**Cost:** measured recall is an underestimate of true detection, and no model can
reach recall 1.0 no matter how good it is.

### 2.3 Adversarial drift, not random drift

In the final 15% of the timeline every mechanism adapts *against its own
defence*: card testing raises amounts into the normal range and stretches over
hours; takeover mimics the victim's amount distribution and reuses a known
device; merchant compromise fans out across long-tail merchants with a longer
delay; the device fingerprint stops being distinctive.

This is the point of the monitoring section. Ordinary drift is incidental — the
world moves on without reference to your model. Fraud drift is targeted: the
attacker is optimising against the deployed model, so degradation concentrates in
whatever the model relied on most.

### 2.4 Merchant compromise is modelled as cross-customer

A merchant is breached, and the cloned cards are used *elsewhere* days later. The
fraud does not happen at the breached merchant. The only trace in the victim's own
history is that they shopped there recently — a signal that lives in the sequence
and that a per-transaction model can only reach through an aggregate.

This is the mechanism where sequence models have the clearest theoretical
advantage, which is exactly why it is worth including.

---

## 3. Temporal integrity

### 3.1 Behavioural leakage tests, not structural ones

`tests/test_leakage.py` does not inspect the feature code. It perturbs the
future, truncates the future, and flips every label, then asserts the past does
not move. That catches leakage regardless of how it was introduced, including by
a future refactor that nobody thought to re-check.

`test_leakage_detector_has_teeth` deliberately builds a leaky feature (the
classic `groupby.transform('mean')` instead of an expanding mean) and asserts the
check rejects it. Without that, a change making the assertions vacuous would
still pass.

### 3.2 The smoothing prior had to become expanding

The leakage suite caught a real dependency on first run. Target encodings were
smoothed towards the global training fraud rate, computed over the whole training
window — which includes labels that arrive *after* the row being encoded. It is
one scalar summarising every label, so the effect is tiny, but it is still a
dependency on the future.

The prior is now expanding: block *k* smooths towards the rate observed over
blocks *< k*, and block 0 smooths towards `initial_prior = 0.005`, an ex-ante
estimate fixed before looking at the data.

### 3.3 Out-of-fold means out-of-time

Shuffled K-fold target encoding is out-of-fold but not out-of-time: a row in
block 2 gets encoded partly with labels from block 5. Here block *k* uses only
blocks *< k*, which is both.

### 3.4 Prior fraud labels are never features

A chargeback is confirmed weeks after the authorisation, so at scoring time the
fraud status of a customer's previous transactions is genuinely unknown. Feeding
prior labels into the sequence would produce excellent offline numbers and a
model that cannot be deployed.

### 3.5 An embargo between splits

One day of gap between train/val and val/test, so a transaction just after a
boundary cannot build its rolling features out of data the model trained on.

---

## 4. Features

### 4.1 Bounded history — the biggest single decision

**Every history-derived feature is bounded at 31 prior transactions.**

The natural implementation uses expanding statistics over a customer's whole
history: lifetime count, tenure, mean spend since signup, every merchant ever
used. Offline that is easy and it is what the first version did.

The serving parity test then failed, and it was right to. An authorisation
request cannot carry an unbounded history. A real service either keeps a stateful
per-customer feature store — a second system that will drift from the training
code — or silently computes something different from what the model was trained
on. The second failure is the common one and it is invisible: every offline check
passes while the deployed model degrades.

Bounding the lookback buys exact online/offline parity from a stateless service.

**Cost, stated plainly:** long-horizon signal is given up. True tenure, lifetime
distinct merchants, and "this customer has never spent more than £50 in three
years" are all unavailable. On this data that costs little because the injected
mechanisms are short-horizon, but on a real portfolio a customer's multi-year
baseline is genuinely informative, and the right production answer is a feature
store with the same code path on both sides rather than a shorter window.

### 4.2 The service asks for 62 prior transactions, not 31

The model reads a 32-position window, and *every* position in that window carries
features built from its own 31 prior transactions. Reproducing the training-time
inputs exactly therefore needs 31 to fill the window plus 31 more behind the
oldest of them. A shorter history still scores — the oldest positions simply see
less than they did in training — so the service degrades rather than rejecting.

### 4.3 Timestamps are quantised to microseconds

pandas carries nanoseconds; JSON datetimes carry microseconds at best. Without a
common quantisation the two paths disagree in the last decimal place of every
time delta. Numerically trivial, operationally exactly the kind of skew that is
never noticed.

### 4.4 Parity has one documented exception

The four label-derived risk encodings differ between training and serving for
windows that overlap the training period, because in training they are computed
out-of-fold. An out-of-fold encoding is a property of the training procedure, not
of the transaction, and it has no online equivalent. For any window lying after
the training period — every window a deployed model actually sees — all 42
features match exactly. Both claims are tested.

### 4.5 Device identity is relational, device *type* is embedded

Raw device IDs are near-unique per customer, so an embedding over them would
learn nothing transferable. The useful device signal is relational (novelty,
recency, count), and those are numeric features. A separate `device_type` column
(browser/OS/app family, ~17 values shared across customers) carries the
embeddable signal.

---

## 5. Models

### 5.1 Embedding widths: `1.6 · cardinality^0.56`, capped at 16

The standard heuristic, with a cap that is doing real work. The merchant
vocabulary is ~2.9k and the uncapped rule asks for 139 dimensions — a
400k-parameter table fitted from roughly a thousand positive training examples.
Capping at 16 keeps it near 47k, which is still most of the model.

Reported per column in `results/embeddings.csv` and in the README, because a
reader comparing "78k parameters" against a GBM deserves to know that most of it
is a lookup table rather than computation.

### 5.2 Rare merchants share one embedding row

Merchants seen fewer than 3 times in training map to a shared "rare" code. A
per-merchant row for a merchant seen twice memorises noise, and "this is a
merchant almost nobody uses" is itself informative.

### 5.3 The scored transaction always occupies the last window position

Windows are left-padded so position `L-1` is always the transaction being
decided. All three architectures read their prediction there, which makes the
causal contract identical across them and means the ONNX exporter and serving
path are written once.

The GRU rolls its input so real transactions start at index 0 and padding trails
behind, then reads the state at `length - 1` — otherwise it would consume a
variable-length run of zeros before seeing any data.

### 5.4 Parameter counts are kept within a factor of ~1.5

Otherwise the comparison measures capacity rather than inductive bias. The TCN
gets four blocks because dilations 1/2/4/8 with kernel 3 give a receptive field
of 61, which covers the 32-position window; two blocks would reach only 13 and
the model would be structurally unable to see most of the history the others get.
A warning fires at construction if the receptive field is ever short.

### 5.5 The Transformer has two attention implementations

The fused kernel for training (about 20% faster end to end) and an explicit one
for extracting attention weights, which are part of the deliverable. A test
asserts the two compute the same function.

---

## 6. Training

### 6.1 Early stopping on validation PR-AUC, never loss

At a 0.65% positive rate the loss is dominated by easy negatives and keeps
drifting down long after ranking has stopped improving.

### 6.2 Three imbalance arms, not two

`focal` and `weighted` are the comparison the brief asks for. A plain `bce` arm
was added as the calibration reference: focal loss reshapes the loss surface and
resampling changes the base rate, so both distort probabilities, and without an
undistorted arm there is nothing to measure the distortion against.

### 6.3 Three seeds minimum, mean and standard deviation always

On a problem with a few hundred test positives, seed-to-seed spread is routinely
comparable to the gap between two architectures. The reporting path only ever
exposes mean ± std; there is no code path that reports a single run as a result.

### 6.4 Hyperparameters are hand-tuned, lightly, on both sides

No automated search for anyone. Tuning the deep models hard while leaving the
baseline at defaults is the usual way these comparisons get rigged; tuning only
the baseline would rig it the other way. Both get a sensible configuration and
the asymmetry is nil.

---

## 7. Evaluation, calibration, policy

### 7.1 Accuracy is reported only next to its own refutation

Every evaluation reports `accuracy` alongside `accuracy_of_always_legit`. They
are the same number to two decimal places. Stating that concretely is more
convincing than asserting that accuracy is inappropriate.

### 7.2 Precision at capacity uses exact alert volumes

Tree models produce many identical scores. A threshold-based implementation would
flag every tied row and hand the fraud team many times the queue they asked for,
so the top-k slice is taken by rank.

### 7.3 ECE uses quantile bins

Equal-width bins put ~99% of this data in the first bucket and report near-zero
calibration error regardless of how wrong the model is.

### 7.4 The policy threshold is chosen on validation and applied unchanged to test

A threshold tuned on the test set is a second fitting step reporting its own
training error. The gap to the unattainable oracle threshold is reported so the
size of that effect is visible.

### 7.5 Missed fraud is valued at the transaction amount

A flat per-fraud cost would hide the thing that matters most: the expensive
frauds are large, and a model can be excellent at catching cheap ones.

### 7.6 The false-positive cost is swept, not asserted

The cost of declining a good customer is the parameter nobody can measure
precisely. `results/cost_sensitivity.csv` shows how the operating point and the
model ranking move across a 15× range of it.

### 7.7 Every stage runs per seed, including the cost policy

The first implementation fitted the calibrator and chose the threshold on a
*seed-averaged* score, then reported the resulting cost as the headline. That is
an ensemble result. It is not wrong as a comparison — both sides were ensembled —
but it answers a question nobody asked, because the artefact this repository
deploys is a single checkpoint. Ensembling also helps the high-variance sequence
models far more than it helps LightGBM, whose seeds barely differ, so it
flattered exactly the side the project was trying to be sceptical about.

Costing per seed and aggregating like every other metric moved the headline from
"27% cheaper" to "7% cheaper, inside one standard deviation". The ranking
advantage survived; the cost advantage did not. The corrected number is the one
reported.

**Cost:** three seeds cannot resolve a 7% cost difference. The fix is more seeds,
not a different metric — but that is stated rather than quietly avoided.

### 7.8 The deployed model is selected on validation cost

Selecting on test cost and then reporting that cost would be circular. Selection
is on the *mean* validation cost across an arm's seeds, and the checkpoint that
ships is the seed closest to that mean — not the best individual run, which would
be selecting on noise. The shipped calibrator is fitted on that same seed's
validation scores.

---

## 8. Interpretability

### 8.1 Attention is reported as evidence pointers, not causal attribution

Attention weights show where the model read from, not why it decided. A
low-weight position can still be decisive (its information may already be in the
residual stream) and a high-weight one can be inert (attention multiplies values
that may be near zero). Different attention distributions can produce identical
outputs, so no single one is *the* explanation.

What attention is genuinely good for is narrower and still useful: pointing an
analyst at related prior activity. The caveat ships in the module, in the worked
example, and in the API response body — a reviewer who knows this literature will
check for it.

### 8.2 Every attention explanation is paired with a gradient attribution

Gradient × input answers a different, counterfactual-flavoured question — how
much does the score move if this input moves — and it can disagree with
attention. Where they disagree, that is worth knowing.

### 8.3 SHAP and attention are not in the same currency

TreeSHAP is exact for tree ensembles, additive, and in log-odds units; the
sequence attributions are first-order and spread over window positions. The
worked example shows both and says so.

---

## 9. Serving

### 9.1 Latency budget: p99 ≤ 50 ms, server-side

A card authorisation has a few hundred milliseconds end to end before the
acquirer times out, shared between network, decisioning, and the issuer's core.
The model is one hop inside that, so 50 ms is the claim. `/metrics` reports the
live distribution against the budget rather than asserting it holds.

### 9.2 ONNX Runtime with one intra-op thread

At batch size 1 there is nothing to parallelise across, and thread hand-off shows
up directly in the tail latency the budget is written against.

### 9.3 The feature path is benchmarked separately from the model

On models this small the feature build is often the larger half of the budget.
Timing only the forward pass is how a "2 ms model" turns into a 40 ms service.

### 9.4 Explanations are computed only for non-approved decisions

ONNX gives no access to attention or gradients, so contributors come from the
PyTorch model. Computing them for the ~99% of traffic that is approved would pay
that cost for nothing.

### 9.5 The audit log flushes per record

A crash must not lose the decisions taken in the seconds before it — those are
exactly the ones anyone will ask about. Each record carries a content hash, so
individual tampering is detectable. A full tamper-evident hash chain with
off-box storage is out of scope and is noted as such.

### 9.6 A service with no model reports `degraded`, not healthy

It comes up and returns 503 from `/score`, so an orchestrator sees an unhealthy
but diagnosable service rather than a crash loop.

---

## 10. Monitoring

### 10.1 Score PSI is a weak detector here — a negative result worth reporting

The standard playbook says to monitor PSI on the score distribution. On this
problem it barely works. More than 99% of transactions are legitimate and the
adversary does not change their behaviour at all, so the bulk of the distribution
sits still while the fraud tail is rewritten underneath it. The injected drift
moves score PSI by roughly 0.01, a tenth of the conventional warning threshold,
while precision at fixed capacity falls by more than half.

A monitor built on score PSI alone would have missed it. `test_score_psi_alone_is
_a_weak_detector_here` pins this down so that if it ever stops being true, the
write-up gets revisited rather than silently becoming wrong.

### 10.2 The alert rate at the fixed threshold is the signal that works

Equally label-free, and it looks only at the region the attacker is operating in.
When fraud learns to score low, the alert rate at an unchanged threshold falls
immediately.

### 10.3 Label-based precision is explicitly the lagging signal

It requires confirmed chargebacks, which arrive weeks later, so by the time it
moves the loss has already been taken. It confirms; it does not warn.

### 10.4 PSI bins are fixed from the reference period

Re-deriving quantile bins per window makes every window look stable by
construction. That is how a drift monitor ends up never firing.

---

## 11. Scope: things deliberately not done

- **No automated hyperparameter search.** Both sides get light hand-tuning; a
  sweep for one and not the other would rig the comparison.
- **No per-episode metrics.** Fraud arrives in episodes (a card-testing burst is
  8–14 transactions), and later transactions in an episode are easy because the
  earlier ones are in the window. Reported recall is therefore optimistic
  relative to "did we catch this attack early". Measuring first-in-episode recall
  would be a better headline; it was left out to keep scope contained, and it is
  the first thing worth adding.
- **No feature store.** See 4.1 — the bounded-lookback design avoids needing one,
  at the cost of long-horizon features.
- **No graph features.** Shared-device and shared-merchant graphs across customers
  are a strong signal for merchant compromise, and out of scope here.
- **No GPU path.** Everything is CPU-trainable by design; the models are small
  enough that a GPU would add complexity and PCIe latency for little gain.
- **No tamper-evident audit chain.** Per-record hashes only; see 9.5.
- **Pickle for the fitted feature builder.** Simple and adequate for a local
  artefact. A production system would use a versioned schema.
