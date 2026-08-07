# Fraud detection with sequence models — and an honest measurement of whether they help

An end-to-end system for real-time transaction fraud detection: synthetic data
with known fraud mechanisms, a deliberately strong gradient-boosting baseline,
three sequence architectures, cost-sensitive decisioning, calibration analysis, a
real-time scoring service, and adversarial drift monitoring.

**The thesis is not "deep learning is better."** It is: here is a problem where
sequence models plausibly beat gradient boosting, and here is the honest
measurement of whether they did.

> **Headline result.** Sequence models rank fraud materially better than the
> gradient-boosting baseline — PR-AUC **0.870 ± 0.014 vs 0.826 ± 0.001**, about
> three standard deviations — and the whole of that advantage appears *after* the
> adversary adapts. Pre-drift the two are indistinguishable and LightGBM is
> marginally ahead (0.956 vs 0.950); post-drift it is **0.765 vs 0.615**.
>
> At the chosen operating point, though, the expected-cost advantage is **not**
> statistically meaningful: **0.0799 ± 0.0122 vs 0.0860 ± 0.0035** per
> transaction, a gap smaller than one standard deviation of the seed spread. What
> the sequence models reliably buy here is robustness to the problem changing —
> not a cost saving you could defend to a finance team on three seeds.

---

## Why sequence models for this problem

Deep learning is not obviously the right tool for tabular fraud detection.
Gradient-boosted trees are the default for good reason: they handle mixed types,
missing values and non-linear interactions with almost no tuning, and on a wide
flat feature table they are very hard to beat. Reaching for a neural network
because it is a neural network is how projects waste six months.

There are two specific properties of this problem where a sequence model has a
genuine structural advantage, and they are why the comparison is worth running.

### 1. Fraud lives in temporal patterns across a customer's history

Almost no transaction is fraudulent on its own evidence. A £3 payment at a
streaming service is unremarkable; nine of them in four minutes from a device the
customer has never used is card testing. A £400 electronics purchase is
unremarkable; the same purchase from someone whose largest spend in six months
was £60, at 3am, three days after they shopped at a merchant that was breached,
is not. The signal is in the *relationship* between transactions — their order,
their spacing, and how the current one deviates from this customer's own recent
behaviour.

A tabular model can reach that signal, but only through features an engineer
thought to write down: count in the last hour, amount z-score against a rolling
mean, time since the same merchant. Each is a hand-specified projection of the
sequence, and each is a guess about which aspect of the ordering matters. A
sequence model gets the ordering itself and can learn projections nobody
enumerated — for instance that *this particular customer* shops in bursts, so a
burst is unremarkable for them and alarming for someone else.

### 2. High-cardinality entities want learned embeddings

Merchant, MCC and device type are categorical with power-law frequency: a few
merchants carry most of the volume and a long tail is seen a handful of times
each. One-hot encoding is hopeless at that width. Target encoding compresses each
merchant to a single number, which is noisy in the tail and discards every
relationship between merchants. A learned embedding can place merchants that
attract the same kind of fraud near each other and share statistical strength
across them.

Both are real advantages rather than borrowed enthusiasm. Whether they are *large
enough to pay for the complexity* is an empirical question, which is what this
repository measures.

The honest counter-argument, which the results have to answer: a well-built
aggregate feature set may already capture most of the temporal signal, and
LightGBM handles high-cardinality categoricals natively and well. If that turns
out to be true here, that is the finding — and a more useful one than another
report of a Transformer winning.

---

## What is being measured, and against what

The baseline is built to win. It gets velocity counts over 1h/24h/7d windows,
amount z-scores against the customer's own history, time since the previous
transaction and since the same merchant/MCC/device, entity novelty flags,
distinct-entity counts, out-of-fold merchant/MCC/country/device-type risk
encodings, and the raw high-cardinality categoricals as native LightGBM
categorical features.

The baseline therefore sees the same information the sequence models see, just
flattened into one row. That makes the comparison a clean question: **is there
signal in the ordering of a customer's transactions that a well-built aggregate
misses?**

A four-rule heuristic is included to establish the floor, so a reader can see how
much of the achievable signal is picked up by obvious rules — and therefore how
much the learned models are actually adding.

---

## The data

Synthetic by default, generated by `src/simulate.py`: ~300k transactions across
2,500 customers over 180 days at a fraud rate near 0.65%, with power-law
merchant/device/MCC cardinality and per-customer spending personalities.

Three fraud mechanisms are injected with **known labels**, which is the point of
generating data rather than downloading it — no public dataset tells you which
attack produced a given fraud, so no public dataset can answer "which mechanisms
is this model blind to?"

| mechanism | what happens | where the signal is |
| --- | --- | --- |
| **card testing** | a burst of small authorisations validating a stolen number | velocity and amount anomaly |
| **account takeover** | a sharp behavioural break from the customer's own baseline | deviation from *this customer*, not from the population |
| **merchant compromise** | cards breached at one merchant, cloned and used elsewhere days later | cross-customer, and visible in the victim's prior sequence |

### Making the problem hard enough to be worth measuring

The first version of the simulator produced fraud that was trivially separable —
the baseline reached **PR-AUC 1.0000** on the pre-drift test period. That makes
the whole architecture comparison vacuous: every model scores 1.0 and there is
nothing left to measure. The fix was not to weaken the models but to make the
world realistic:

- customers hold 2–4 devices with lopsided usage, and ~5% of online transactions
  come from a device never seen before (new browser, borrowed laptop);
- phone upgrades retire a device mid-period;
- ~10% of customers make legitimate bursts of small purchases inside an hour —
  which is exactly what card testing hides behind;
- ~2% of legitimate purchases are large one-offs, and travel produces foreign
  countries and unfamiliar merchants;
- attackers reuse a victim's own device 15–25% of the time;
- the automation fingerprint (headless clients, emulators) shows up in only about
  a third of pre-drift fraud sessions;
- 4% of frauds are never disputed, so they carry a legitimate label — which caps
  achievable recall, as it does in reality.

### Adversarial drift

In the final 15% of the timeline, every mechanism adapts **against its own
defence**: card testing raises amounts into the normal range and stretches over
hours; takeover mimics the victim's amount distribution and reuses a known
device; merchant compromise fans out across long-tail merchants with a longer
delay; the device fingerprint stops being distinctive.

This is what makes the monitoring section testable rather than decorative.

---

## Temporal integrity

The most dangerous bug in this problem class is a feature that reads the future.
It never announces itself: training converges, validation looks excellent, and
only the live system disagrees.

Three rules are enforced, all tested behaviourally rather than by inspection.

1. **A feature for a transaction at time *t* uses only transactions strictly
   before *t*, for that customer.** `tests/test_leakage.py` scrambles every
   transaction after a cut point and asserts nothing before the cut moved. It
   also deletes the future entirely, to catch leakage through counts rather than
   values.
2. **Labels are never features, not even prior labels.** A chargeback is
   confirmed weeks after the authorisation, so at scoring time the fraud status
   of a customer's previous transactions is genuinely unknown. Flipping every
   label in the dataset must leave every causal feature bit-identical.
3. **Anything that consumes labels is fitted on the training period only, and
   out-of-time within it.** Shuffled K-fold target encoding is out-of-fold but
   not out-of-time — a row in block 2 gets encoded with labels from block 5. Here
   block *k* uses only earlier blocks.

The suite includes `test_leakage_detector_has_teeth`, which builds a deliberately
leaky feature (the classic `groupby.transform("mean")` instead of an expanding
mean) and asserts the checks reject it. Without it, a refactor that made the
assertions vacuous would still pass.

**This suite found a real leak on its first run.** The target-encoding smoothing
prior was the global training fraud rate over the whole training window — which
includes labels arriving after the row being encoded. One scalar, a tiny effect,
and still a dependency on the future. The prior is now expanding.

---

## Online/offline parity

The second most dangerous bug is an online feature pipeline that quietly
disagrees with the offline one: a different window boundary, a different
unknown-category fallback, a different scaling constant, and the deployed model
is fed inputs it was never trained on.

The per-customer feature computation is written **once**, as a function over
ordered numpy arrays, and called by both the batch pipeline and the scoring
service. `tests/test_serving.py` asserts the two produce identical values.

That test forced the biggest design decision in the project. The first
implementation used expanding statistics over a customer's whole history —
lifetime count, tenure, every merchant ever seen. Offline that is easy; online it
is impossible, because an authorisation request cannot carry unbounded history.
The choice was between a stateful feature store (a second system that will drift
from the training code) and bounded features.

**Every history-derived feature is now bounded at 31 prior transactions.** The
cost is stated rather than hidden: long-horizon signal is given up. On this data
that costs little because the injected mechanisms are short-horizon; on a real
portfolio a customer's multi-year baseline is genuinely informative, and the
right answer there is a feature store sharing this same code path.

The service asks for 62 prior transactions: 31 to fill the model window, and 31
more so the oldest position in the window has the same history behind it that it
had during training.

---

## Models

All three architectures share one input encoder and read their prediction off the
final window position, which is always the transaction being scored. That keeps
the causal contract identical across them, and means the ONNX exporter and the
serving path are written once.

| | |
| --- | --- |
| **GRU** | single layer, hidden 64. Input is rolled so real transactions start at index 0 and padding trails behind, then the state is read at `length - 1` — so the recurrence never consumes a variable-length run of zeros before seeing data. |
| **TCN** | 4 dilated causal blocks, kernel 3, dilations 1/2/4/8, 48 channels. Causality is *structural*: convolutions are left-padded and the overhang sliced, so no masking error can let the future through. Receptive field 61, covering the 32-position window. |
| **Transformer** | 2 pre-norm blocks, `d_model` 64, 4 heads, learned positional encoding, causal + padding mask. Uses the fused attention kernel for training and an explicit one for extracting weights; a test asserts the two compute the same function. |

### Embedding dimensions, and how they were chosen

Widths follow the standard heuristic `1.6 · cardinality^0.56`, rounded, floored
at 2, and **capped at 16**. The cap does real work: the merchant vocabulary is
~2.9k and the uncapped rule asks for 139 dimensions — a 400k-parameter table
fitted from roughly a thousand positive training examples.

| column | cardinality | embedding dim | parameters |
| --- | --- | --- | --- |
| `merchant_code` | 2,910 | 16 | 46,560 |
| `mcc_code` | 22 | 9 | 198 |
| `device_type_code` | 19 | 8 | 152 |
| `country_code` | 10 | 6 | 60 |
| `channel_code` | 4 | 3 | 12 |
| `entry_mode_code` | 6 | 4 | 24 |
| **total** | | | **47,006** |

Merchants seen fewer than 3 times in training share one "rare" embedding row. A
per-merchant row for a merchant seen twice memorises noise, and "almost nobody
uses this merchant" is itself informative.

Note when reading parameter counts below: **most of each model is this lookup
table, not computation.**

---

## Results

### The dataset these numbers come from

296,859 transactions, 0.63% fraud, split by time with a one-day embargo:

| split | transactions | frauds | period |
| --- | --- | --- | --- |
| train | 177,737 | 1,137 | 2023-01-01 → 2023-04-18 |
| validation | 43,501 | 323 | 2023-04-20 → 2023-05-15 |
| test | 72,474 | 374 | 2023-05-16 → 2023-06-29 |

Adversarial drift begins **2023-06-03**, so the test window deliberately
straddles it: 18 days of the stationary problem followed by 27 days of an
adversary that has adapted. Every aggregate test number below is an average over
those two very different regimes, which is why they are also reported separately.

### Architecture comparison

Mean ± standard deviation over three seeds. Training times are wall-clock on 4
CPU cores, no GPU.

| model | PR-AUC | precision @0.5% | recall @0.5% | params | train (s) |
| --- | --- | --- | --- | --- | --- |
| `gru_bce` | 0.8703 ± 0.0137 | 0.8163 ± 0.0193 | 0.7923 ± 0.0188 | 78,031 | 118.9 |
| `gru_focal` | 0.8697 ± 0.0264 | 0.8237 ± 0.0266 | 0.7995 ± 0.0258 | 78,031 | 172.9 |
| `tcn_focal` | 0.8569 ± 0.0294 | 0.8090 ± 0.0241 | 0.7852 ± 0.0234 | 112,591 | 329.3 |
| `tcn_bce` | 0.8487 ± 0.0209 | 0.7961 ± 0.0168 | 0.7727 ± 0.0163 | 112,591 | 339.3 |
| `transformer_bce` | 0.8333 ± 0.0128 | 0.7980 ± 0.0152 | 0.7745 ± 0.0147 | 105,679 | 277.9 |
| `gru_weighted` | 0.8285 ± 0.0045 | 0.7888 ± 0.0104 | 0.7656 ± 0.0101 | 78,031 | 144.9 |
| `transformer_focal` | 0.8282 ± 0.0147 | 0.7815 ± 0.0207 | 0.7585 ± 0.0201 | 105,679 | 315.9 |
| `tcn_weighted` | 0.8266 ± 0.0269 | 0.7833 ± 0.0214 | 0.7602 ± 0.0208 | 112,591 | 255.9 |
| `lightgbm` | 0.8261 ± 0.0010 | 0.7649 ± 0.0069 | 0.7424 ± 0.0067 | 59,031 | 19.7 |
| `lightgbm_weighted` | 0.8061 ± 0.0020 | 0.7309 ± 0.0057 | 0.7094 ± 0.0056 | 156,492 | 42.3 |
| `transformer_weighted` | 0.7944 ± 0.0191 | 0.7668 ± 0.0207 | 0.7442 ± 0.0201 | 105,679 | 347.7 |
| `rules` | 0.0466 | 0.0496 | 0.0481 | 4 | 0.0 |

For the LightGBM rows, "params" is total leaf count across all trees — a
parameter count is a category error for a tree ensemble, and leaf count is the
honest analogue. `rules` is four hand-written conditions.

Read the standard deviations before the means. `gru_bce` leads `tcn_focal` by
0.013 PR-AUC with a seed spread of 0.014–0.029 — that ordering is noise. What is
**not** noise is the gap from the best sequence models (~0.870) to LightGBM
(0.826): about 3 standard deviations of the GRU's own seed variation, and
LightGBM's spread is 0.001, so the baseline is not the uncertain side.

Two things worth noticing:

- **The GRU wins, and it is the smallest and fastest model.** 78k parameters and
  ~2 minutes to train, against the Transformer's 106k and ~5 minutes. The extra
  machinery bought nothing here. On 1,137 positive training examples that is
  unsurprising: attention has more capacity to spend and not enough signal to
  spend it on.
- **Accuracy is not in this table on purpose.** Predicting "never fraud" scores
  99.48% accurate on this test set, and so does every model above. The evaluation
  code prints both numbers side by side so the point cannot be missed.

### Precision and recall at review capacity

The operating points a fraud team can actually staff.

| model | P@0.1% / R@0.1% | P@0.5% / R@0.5% | P@1% / R@1% |
| --- | --- | --- | --- |
| `gru_bce` | 0.991 / 0.193 | 0.816 / 0.792 | 0.466 / 0.904 |
| `gru_focal` | 0.995 / 0.194 | 0.824 / 0.799 | 0.464 / 0.899 |
| `tcn_focal` | 0.991 / 0.193 | 0.809 / 0.785 | 0.458 / 0.889 |
| `tcn_bce` | 0.995 / 0.194 | 0.796 / 0.773 | 0.457 / 0.885 |
| `transformer_bce` | 0.991 / 0.193 | 0.798 / 0.775 | 0.449 / 0.871 |
| `gru_weighted` | 0.991 / 0.193 | 0.789 / 0.766 | 0.452 / 0.877 |
| `transformer_focal` | 0.995 / 0.194 | 0.781 / 0.758 | 0.445 / 0.862 |
| `tcn_weighted` | 1.000 / 0.195 | 0.783 / 0.760 | 0.451 / 0.875 |
| `lightgbm` | 1.000 / 0.195 | 0.765 / 0.742 | 0.453 / 0.879 |
| `lightgbm_weighted` | 1.000 / 0.195 | 0.731 / 0.709 | 0.446 / 0.865 |
| `transformer_weighted` | 0.991 / 0.193 | 0.767 / 0.744 | 0.433 / 0.839 |
| `rules` | 0.014 / 0.003 | 0.050 / 0.048 | 0.050 / 0.096 |

### Which mechanisms each model catches

Recall at 0.5% capacity, per attack type. This is the breakdown aggregate metrics
hide, and it is why the synthetic data is worth generating.

| model | account takeover | card testing | merchant compromise |
| --- | --- | --- | --- |
| `gru_bce` | 0.639 ± 0.046 | 0.916 ± 0.013 | 0.767 ± 0.047 |
| `gru_focal` | 0.653 ± 0.049 | 0.922 ± 0.019 | 0.760 ± 0.013 |
| `tcn_focal` | 0.613 ± 0.045 | 0.929 ± 0.016 | 0.736 ± 0.013 |
| `tcn_bce` | 0.590 ± 0.024 | 0.922 ± 0.006 | 0.736 ± 0.036 |
| `transformer_bce` | 0.560 ± 0.032 | 0.941 ± 0.000 | 0.767 ± 0.023 |
| `gru_weighted` | 0.574 ± 0.026 | 0.922 ± 0.013 | 0.729 ± 0.036 |
| `transformer_focal` | 0.553 ± 0.039 | 0.923 ± 0.024 | 0.729 ± 0.027 |
| `tcn_weighted` | 0.583 ± 0.042 | 0.898 ± 0.005 | 0.752 ± 0.048 |
| `lightgbm` | 0.604 ± 0.007 | 0.840 ± 0.011 | 0.783 ± 0.027 |
| `lightgbm_weighted` | 0.588 ± 0.014 | 0.797 ± 0.005 | 0.736 ± 0.027 |
| `transformer_weighted` | 0.507 ± 0.055 | 0.927 ± 0.008 | 0.744 ± 0.023 |
| `rules` | 0.049 | 0.011 | 0.209 |

Card testing is the sequence models' clearest win — 0.92 vs LightGBM's 0.84 —
which is exactly the mechanism whose signature is a *pattern across consecutive
transactions* rather than a property of any one of them. On merchant compromise
LightGBM is marginally ahead (0.783 vs 0.767), and account takeover is close.
That is a coherent story rather than a uniform win: ordering helps most where the
evidence is literally the ordering.

### Before and after the adversary adapts

| model | PR-AUC pre-drift | PR-AUC post-drift | drop |
| --- | --- | --- | --- |
| `gru_bce` | 0.9499 | 0.7654 | 0.1845 |
| `gru_focal` | 0.9517 | 0.7595 | 0.1922 |
| `tcn_focal` | 0.9451 | 0.7377 | 0.2074 |
| `tcn_bce` | 0.9542 | 0.7142 | 0.2400 |
| `transformer_bce` | 0.9491 | 0.6874 | 0.2617 |
| `gru_weighted` | 0.9434 | 0.6682 | 0.2752 |
| `transformer_focal` | 0.9533 | 0.6672 | 0.2861 |
| `tcn_weighted` | 0.9466 | 0.6641 | 0.2825 |
| `transformer_weighted` | 0.9226 | 0.6227 | 0.2999 |
| `lightgbm` | 0.9559 | 0.6153 | 0.3406 |
| `lightgbm_weighted` | 0.9476 | 0.5826 | 0.3650 |
| `rules` | 0.0948 | 0.0232 | 0.0716 |

**This is the real finding.** Before the drift every model sits between 0.943 and
0.956 and LightGBM is at the top of that range — the sequence models buy nothing
at all on the stationary problem. After the drift the ordering inverts and the
spread opens to 0.14 PR-AUC, with LightGBM losing the most (−0.341) and the GRU
the least (−0.185).

The mechanism is visible in the feature set. The baseline leans hardest on
features that the adversary specifically attacked — the device fingerprint,
merchant risk rates, small-amount velocity. When card testing raises its amounts
and merchant compromise fans out across the long tail, those features stop
carrying signal and the aggregate has nothing left. The sequence models still see
that three transactions arrived from the same unfamiliar device inside four
minutes, which is a *relational* fact that survives the adversary changing the
values.

The honest caveat: this is one synthetic drift, designed by the same author as
the models. It shows sequence models degrading more gracefully against *this*
adaptation. It is evidence, not proof, that they would against a real one.

### Expected cost per transaction — the headline

Not PR-AUC. A blocked legitimate transaction and a missed fraud are not
symmetric, so the ranking that matters is the one under an explicit cost model:
**£8 for declining a good customer**, **£15 plus the full transaction amount for
missing a fraud**, £2 to review. Thresholds are chosen on validation and applied
unchanged to test, per seed, and aggregated the same way every other table is.

| model | cost / txn | vs block-nothing | alert rate | precision | recall |
| --- | --- | --- | --- | --- | --- |
| `gru_focal` | 0.0799 ± 0.0122 | -0.2566 | 0.0082 | 0.565 | 0.880 |
| `gru_bce` | 0.0815 ± 0.0049 | -0.2550 | 0.0100 | 0.471 | 0.901 |
| `transformer_focal` | 0.0839 ± 0.0088 | -0.2526 | 0.0073 | 0.592 | 0.829 |
| `transformer_bce` | 0.0841 ± 0.0084 | -0.2524 | 0.0089 | 0.502 | 0.859 |
| `tcn_focal` | 0.0850 ± 0.0108 | -0.2514 | 0.0088 | 0.527 | 0.874 |
| `tcn_bce` | 0.0858 ± 0.0081 | -0.2507 | 0.0082 | 0.543 | 0.860 |
| `lightgbm` | 0.0860 ± 0.0035 | -0.2505 | 0.0096 | 0.472 | 0.875 |
| `tcn_weighted` | 0.0879 ± 0.0097 | -0.2485 | 0.0095 | 0.477 | 0.869 |
| `lightgbm_weighted` | 0.0896 ± 0.0018 | -0.2469 | 0.0075 | 0.561 | 0.816 |
| `gru_weighted` | 0.0905 ± 0.0059 | -0.2459 | 0.0091 | 0.503 | 0.863 |
| `transformer_weighted` | 0.1009 ± 0.0007 | -0.2356 | 0.0086 | 0.503 | 0.823 |
| `rules` | 0.5059 | +0.1694 | 0.0454 | 0.064 | 0.561 |
| _block nothing_ | 0.3365 | - | 0.0000 | - | 0.000 |
| _block everything_ | 7.9690 | - | 1.0000 | - | 1.000 |

**And here the story stops being flattering.** The best sequence arm costs
0.0799 ± 0.0122 against LightGBM's 0.0860 ± 0.0035 — a 7% improvement, and a gap
*smaller than one standard deviation* of the sequence model's own seed spread.
Its three seeds cost 0.0711, 0.0749 and 0.0938; LightGBM's cost 0.0821, 0.0869
and 0.0890. Those distributions overlap. On this evidence you cannot claim a
cost saving.

That is a correction to an earlier version of this write-up, and the mistake is
worth preserving because it is easy to make. The first implementation costed the
*seed-averaged* score — it fitted one calibrator on the mean of three seeds'
predictions and thresholded that. Averaging three models' scores is an ensemble,
and ensembling helps the high-variance sequence models much more than it helps
LightGBM, whose seeds barely differ. The result was a reported 27% cost
reduction. Nothing was mislabelled and nothing leaked: the comparison was even
fair, since both sides were ensembled. It was just answering a question nobody
asked, because **the artefact this repository deploys is one checkpoint, not an
ensemble of three.** `src/experiment.py` now runs calibration, thresholding and
costing per seed, and the honest number is the one above.

The ranking advantage survives this correction; the cost advantage does not.
That asymmetry is itself informative: PR-AUC integrates over every threshold,
while expected cost is evaluated at a single operating point chosen on
validation, and that choice is noisy when it is made from ~320 validation
frauds.

`results/cost_sensitivity.csv` sweeps the false-positive cost from £2 to £30;
`results/policy_per_seed.csv` has the individual runs.

### Calibration

Fitted on validation, measured on test. The right-hand columns are the ones that
matter: aggregate calibration on this problem is dominated by the 99%+ of
transactions correctly scored near zero, so a model can look immaculate overall
and still be wrong in the only band where a decision is taken.

| model | ECE raw | ECE temp | ECE isotonic | alert-region pred | alert-region actual | ratio |
| --- | --- | --- | --- | --- | --- | --- |
| `gru_bce` | 0.00055 | 0.00042 | 0.00019 | 0.804 | 0.816 | 0.98x |
| `gru_focal` | 0.00517 | 0.00155 | 0.00013 | 0.570 | 0.824 | 0.69x |
| `gru_weighted` | 0.00092 | 0.00129 | 0.00023 | 0.856 | 0.789 | 1.08x |
| `lightgbm` | 0.00190 | 0.00168 | 0.00033 | 0.648 | 0.765 | 0.85x |
| `lightgbm_weighted` | 0.00131 | 0.00011 | 0.00037 | 0.763 | 0.731 | 1.04x |
| `rules` | 0.09544 | 0.00641 | 0.00270 | 0.515 | 0.050 | 10.38x |
| `tcn_bce` | 0.00086 | 0.00070 | 0.00027 | 0.802 | 0.796 | 1.01x |
| `tcn_focal` | 0.00595 | 0.00152 | 0.00017 | 0.574 | 0.809 | 0.71x |
| `tcn_weighted` | 0.00058 | 0.00142 | 0.00020 | 0.964 | 0.783 | 1.23x |
| `transformer_bce` | 0.00093 | 0.00096 | 0.00047 | 0.804 | 0.798 | 1.01x |
| `transformer_focal` | 0.00664 | 0.00149 | 0.00035 | 0.594 | 0.781 | 0.76x |
| `transformer_weighted` | 0.00115 | 0.00173 | 0.00038 | 0.871 | 0.767 | 1.14x |

Two results here contradict the received wisdom, which is why they were measured
rather than assumed:

1. **The neural nets are not overconfident — they are *under*confident**, and
   only under focal loss. In the alert region `gru_focal` predicts 0.570 where
   the true rate is 0.824, a ratio of 0.69. The cause is not mysterious: focal
   loss is configured with `alpha = 0.25`, which weights the *positive* class at
   0.25 and the negative at 0.75, systematically shrinking positive
   probabilities. Focal loss also flattens the distribution from both ends — it
   raises the median prediction from 0.00002 to 0.004 while lowering the top —
   which is why its aggregate ratio (2.0x) points the opposite way to its
   alert-region ratio (0.69x). One number would have told the wrong story.
2. **Weighted sampling did not destroy calibration.** The expectation was that
   oversampling positives 40x would inflate probabilities badly. In the alert
   region the weighted arms land at 1.04–1.23x — comparable to plain
   cross-entropy, not the blow-up that was predicted.
   What resampling did damage is *ranking*: `gru_weighted` loses 0.042 PR-AUC
   against `gru_bce`. The trade ran opposite to the textbook version.

Isotonic regression fixes everything to ECE ≈ 0.0001–0.0005 and is rank
preserving, so it costs no PR-AUC. It is what the deployed service applies —
fitted on the deployed seed's own validation scores, not on a seed average.


---

## Serving

The best model is exported to ONNX and served by FastAPI.

**Latency budget: p99 ≤ 50 ms, server-side.** A card authorisation has a few
hundred milliseconds end to end before the acquirer times out, shared between
network, decisioning and the issuer's core. The model is one hop inside that.

| path | p50 (ms) | p95 (ms) | p99 (ms) |
| --- | --- | --- | --- |
| `model:pytorch` | 0.790 | 1.053 | 1.310 |
| `model:onnxruntime` | 0.284 | 0.356 | 0.421 |
| `features` | 1.303 | 1.769 | 2.244 |
| `end_to_end:onnxruntime` | 3.885 | 4.850 | 7.154 |
| `end_to_end:pytorch` | 4.100 | 5.763 | 6.685 |

**The budget is met with a 7x margin: p99 is 7.2 ms against 50 ms.** Two
things in that table are worth more than the headline number:

- ONNX Runtime is **2.8x faster than eager PyTorch** on the forward pass at batch
  size 1 (0.284 ms vs 0.790 ms p50). Both runtimes are benchmarked rather than
  assumed, because on models this small the gap is not always worth the export.
- **The feature build costs more than the model** — 1.30 ms against 0.28 ms. That
  is the number that gets missed when only the forward pass is timed, and it is
  why `features` is measured as its own path. If this service needed to be
  faster, the model is not where the time is.

These are measured on a shared 4-core CPU with other work running, so treat them
as an upper bound rather than a tuned benchmark.

Endpoints:

| endpoint | purpose |
| --- | --- |
| `POST /score` | transaction + customer history in; fraud probability, decision, top contributing prior transactions, model version out |
| `GET /health` | liveness *and* whether a model is actually loaded — a service that is up with no model is not healthy |
| `GET /model-info` | version, family, parameter count, window length, embedding widths, thresholds, calibration method, latency budget |
| `GET /metrics` | live latency percentiles against the budget, decision mix, request counts |

Every decision is written to an append-only audit log — score, threshold, model
version, contributing transactions, content hash — **before** the response
returns, because a customer who complains months later deserves an answer that
does not depend on re-running a model that has since changed.

```bash
docker compose up --build      # mounts ./artifacts read-only, ./audit writable
```

---

## Interpretability

`src/explain.py` produces attention weights from the Transformer, gradient
attributions for every architecture, and TreeSHAP for the LightGBM baseline.
`results/worked_example.txt` contains one flagged transaction explained end to
end.

**On attention as explanation, plainly:** attention weights show where the model
*read from*, not why it decided. They are not a causal account. A low-weight
position can still be decisive — its information may already be carried in the
residual stream — and a high-weight one can be inert, since attention multiplies
values that may be near zero. Different attention distributions can produce
identical outputs, so no single one is *the* explanation.

What attention is genuinely good for here is narrower and still useful: pointing
an analyst at the related prior activity worth reviewing. Every attention
explanation in this repository ships alongside a gradient attribution, which
answers a different, counterfactual-flavoured question — how much does the score
move if this input moves — and which can disagree. The caveat is in the module,
in the worked example, and in the API response body.

SHAP and attention are also not in the same currency: TreeSHAP is exact and
additive in log-odds over engineered features; the sequence attributions are
first-order and spread over window positions.

---

## Monitoring

The synthetic data has adversarial drift injected in its final period, so the
monitor can be *tested* rather than asserted. Both signals below are label-free
and available immediately; precision requires confirmed chargebacks and therefore
arrives weeks late.

| start | n | score_psi | alert_rate | alert_rate_ratio | precision_at_capacity | recall_at_capacity | severity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-05-16 23:59 | 11623 | 0.0000 | 0.0108 | 0.9508 | 1.0000 | 0.6211 | ok |
| 2023-05-24 00:00 | 11615 | 0.0001 | 0.0091 | 0.8004 | 0.9831 | 0.7838 | ok |
| 2023-05-31 00:00 | 11489 | 0.0001 | 0.0065 | 0.5725 | 0.6379 | 0.7551 | warn |
| 2023-06-07 00:01 | 11583 | 0.0001 | 0.0063 | 0.5527 | 0.6724 | 0.8125 | warn |
| 2023-06-14 00:02 | 11530 | 0.0011 | 0.0074 | 0.6466 | 0.5690 | 0.6111 | warn |
| 2023-06-21 00:00 | 11653 | 0.0006 | 0.0070 | 0.6172 | 0.6102 | 0.9231 | warn |
| 2023-06-28 00:02 | 2981 | 0.0001 | 0.0070 | 0.6178 | 0.2000 | 0.2000 | warn |

Drift begins **2023-06-03**. The window starting 2023-05-31 is the first
containing post-drift traffic, and it is the window that fires.

**The conventional signal failed, and that is worth reporting.** Score PSI — the
standard drift monitor — peaked at **0.0011** across the entire test period,
about a hundredth of the conventional 0.10 warning threshold. It never fired. It
was never going to: more than 99% of transactions are legitimate and the
adversary does not change their behaviour at all, so the distribution PSI
measures is dominated by a population that did not move. Meanwhile precision at
fixed capacity fell from 1.00 to 0.64 in the same window.

**The alert rate at the fixed production threshold caught it immediately.** It
fell to 57% of its reference value in the first window containing drifted
traffic, crossing the warning threshold while the loss was still accumulating
rather than after it had been booked. It is equally label-free and it looks only
at the band the attacker is operating in.

`tests/test_monitoring.py::test_score_psi_alone_is_a_weak_detector_here` pins
this negative result down, so if it ever stops being true the write-up gets
revisited instead of quietly becoming wrong.

### Why fraud drift is not ordinary drift

Ordinary covariate drift is incidental. Customers gradually change what they buy,
a merchant grows, a country is onboarded; the world moves on without reference to
your model, and it moves slowly and in no particular direction.

Fraud drift is **targeted**. The adversary is optimising against the deployed
model, and three consequences follow directly from that:

1. **Degradation concentrates in whatever you were best at.** The attacker
   attacks the feature you rely on most. That is exactly what happened here:
   LightGBM leaned hardest on the device fingerprint and merchant risk rates, so
   when those were neutralised it lost the most (−0.341 PR-AUC against the GRU's
   −0.185). An aggregate metric can look stable while one mechanism collapses,
   which is why recall is tracked per mechanism per window.
2. **It is abrupt, not gradual.** A new technique appears when it appears.
   Smoothing over a long window to suppress false alarms delays detection of
   precisely the event worth detecting.
3. **The labels arrive too late to be the alarm.** Waiting for confirmed
   chargebacks means learning about a loss you have already taken. The alarm has
   to be built from what is available at authorisation time.

---

## So: did deep learning beat gradient boosting?

**On ranking, yes and clearly. On the metric that decides deployment, not
provably — and that distinction is the whole finding.**

| claim | evidence | holds up? |
| --- | --- | --- |
| Sequence models rank better | PR-AUC 0.870 ± 0.014 vs 0.826 ± 0.001, ~3σ | **yes** |
| They catch more at a workable alert budget | precision @0.5% 0.824 vs 0.765 | **yes** |
| They do **not** beat the baseline on the stationary problem | pre-drift 0.950 vs 0.956 — LightGBM marginally ahead | **yes** |
| The advantage is robustness to adversarial drift | post-drift 0.765 vs 0.615 | **yes** |
| Largest exactly where ordering *is* the evidence | card-testing recall 0.92 vs 0.84 | **yes** |
| They save money at the deployed operating point | 0.0799 ± 0.0122 vs 0.0860 ± 0.0035 | **no — inside the noise** |
| Architecture choice within the sequence family matters | top four arms separated by less than their seed spread | **no** |
| The winner is the smallest sequence model | GRU: 78k parameters, ~2 min to train, beats the 106k Transformer | **yes** |

Two findings I did not expect, and would lead with in a review:

**1. Had the test period ended before the drift, this project would have
concluded that sequence models are not worth it** — and would have been right on
that evidence. Pre-drift, a well-built LightGBM matches or beats every neural
network here, trains in 20 seconds instead of 5 minutes, and has a tenth of the
seed variance. Anyone evaluating on a stationary holdout would have measured
that, concluded correctly, and shipped the tree. Fraud is adversarial; a
stationary holdout measures the one regime guaranteed not to persist.

**2. A better ranker is not automatically a cheaper one.** PR-AUC integrates over
all thresholds; expected cost is evaluated at one operating point, picked on
validation from a few hundred frauds. The sequence models win the first
comfortably and cannot demonstrate a win on the second at three seeds. If the
deployment decision rests on cost, the honest answer is "run more seeds, or
accept that you are buying drift-robustness rather than a saving."

### When each is the right choice

**Use gradient boosting when** the entity vocabulary is small or stable, history
per entity is short, you need to ship this week, or you cannot support a
retraining and monitoring loop. It is genuinely competitive here, trains in
seconds, has near-zero seed variance, and it *won* the stationary comparison. The
right way to lose to a tree baseline is to not run it.

**Use a sequence model when** the attack you care about is defined by a pattern
across transactions rather than by any one of them, when entity cardinality is
high enough that embeddings can share strength, and — the load-bearing condition
— when you expect the adversary to adapt and can support retraining. Start with a
GRU. Reach for attention only when you have evidence the GRU is the constraint,
which on ~1,100 positive training examples it was not.

### What would change this conclusion

- **More seeds.** Three is the stated minimum and it is visibly not enough to
  resolve a 7% cost difference. Ten would settle it and costs nothing but time.
- **More positives.** With 1,137 fraudulent training transactions the Transformer
  is capacity-starved; the architecture ordering could invert on a real portfolio
  with 10–100× the fraud volume.
- **A different drift.** This one was designed by the same author as the models.
  A drift attacking *sequence* structure rather than feature values would
  presumably hurt the sequence models more.
- **Longer horizons.** Every feature is bounded at 31 prior transactions (see
  DECISIONS.md §4.1). A model with genuine long-range memory might beat both.
- **Cross-customer structure.** Merchant compromise is a graph phenomenon;
  neither family sees it as one. A graph model is the obvious next thing to try.

### The honest limitations

- One synthetic population from one generator, with one dataset seed. Model seeds
  vary; the world does not.
- Recall is measured per transaction, not per *episode*. A card-testing burst is
  5–14 transactions and the later ones are easy because the earlier ones are
  already in the window, so every number here is optimistic relative to "did we
  catch this attack early". Fixing that is the first thing I would add.
- The cost model is assumed. £8 per false positive is a guess; the sensitivity
  sweep shows the *ordering* is stable from £2 to £30, but the ordering is inside
  the noise band anyway.
- No fairness evaluation is possible on data with no demographic attributes. That
  is a gap, not a clean bill of health. See MODEL_CARD.md.

---

## Reproducing

Everything is CPU-only and free. No dataset is downloaded and no credentials are
needed.

```bash
python -m venv .venv && source .venv/bin/activate

# CPU-only torch is far smaller than the default wheel, which pulls the CUDA
# runtime onto a machine with no GPU.
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[serve,explain,dev]"

make simulate      # generate the synthetic dataset (~5 s)
make experiment    # train everything, evaluate, calibrate, cost, monitor, export
make test          # full test suite, including the leakage checks
make serve         # scoring API on :8000
python -m src.report   # regenerate every table in this README from results/
```

`make experiment` reuses cached scores in `results/scores/` when present; pass
`--retrain` to force a full retrain.

### Optional: real datasets

Two public datasets are supported behind `--dataset`. Neither is downloaded
automatically and no Kaggle credentials are requested — put the file in
`data/raw/` and the loader finds it. Running without the file prints the download
instructions and stops.

```bash
python -m src.experiment --dataset ieee-cis   # train_transaction.csv, train_identity.csv
python -m src.experiment --dataset ulb        # creditcard.csv
```

Both adaptations lose something, and the loaders say so rather than hiding it:

- **IEEE-CIS** has no true cardholder or merchant key. A `card1/card2/addr1`
  composite stands in for the customer and `ProductCD` + email domain for the
  merchant. Fraud mechanisms are unknown, so the per-mechanism breakdown is
  suppressed.
- **ULB** has no entity identifiers at all — 28 anonymised PCA components, an
  elapsed-time column and an amount. Per-customer sequences cannot be constructed
  from it, so it exercises the tabular baseline honestly and the sequence models
  only degenerately. It is included because it is the dataset everyone reaches
  for, and it is worth being explicit about why it cannot support this project's
  question.

---

## Repository layout

```
src/
  simulate.py          synthetic generator: mechanisms, confounders, adversarial drift
  config.py            every knob that changes a reported number
  data/
    schema.py          canonical transaction schema shared by all datasets
    loaders.py         synthetic + optional real datasets, with download instructions
    features.py        bounded causal features; one implementation, batch and serving
    sequences.py       left-padded windows, masking, weighted sampling
    splits.py          temporal splits with an embargo
  baselines.py         LightGBM, built to win, plus a four-rule floor
  models/              GRU / TCN / Transformer over one shared encoder
  train.py             PR-AUC early stopping, focal vs weighted vs plain, multi-seed
  evaluation.py        PR-AUC, precision at capacity, per-mechanism recall
  calibration.py       Brier, ECE, reliability, temperature and isotonic
  policy.py            cost model, threshold sweep, expected cost per transaction
  explain.py           attention, gradient attribution, SHAP, worked example
  monitoring.py        PSI, alert-rate drift, rolling precision, detection summary
  experiment.py        the whole pipeline, one command
  report.py            renders this README's tables from results/
  api/                 ONNX export, FastAPI service, audit log, latency benchmark
tests/                 including test_leakage.py, which is the one that matters
```

## Further reading in this repository

- **DECISIONS.md** — every judgement call and its reasoning, including what each
  one cost.
- **MODEL_CARD.md** — intended use, limitations, ethical considerations.
