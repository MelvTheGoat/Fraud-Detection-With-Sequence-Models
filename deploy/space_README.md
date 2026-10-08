---
title: Transaction Fraud Scoring
emoji: 🪪
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Sequence models for real-time card fraud, served on ONNX Runtime
---

# Transaction fraud scoring

A GRU over a customer's last 32 transactions, calibrated, thresholded against an
explicit cost model, exported to ONNX and served behind FastAPI. Pick a customer,
edit the transaction being authorised, and see what the model does with it.

**Trained on synthetic data.** The transactions come from a simulator in the
source repository, not from a real portfolio. The numbers below are honest about
that simulator and claim nothing about real card fraud.

## What it does

The page itself reports the numbers for the exact bundle it is serving, read out
of the run that produced it, including how many seeds that run used. It is
deliberately not hardcoded here: a README quoting a three-seed error bar over a
bundle built from a one-seed run states a measurement nobody made.

For reference, the repository's published three-seed run gives:

| | |
|---|---|
| Test PR-AUC | **0.8703 ± 0.0137** (GRU) vs **0.8261 ± 0.0010** (LightGBM on the same features) |
| Cost per transaction | **0.0799 ± 0.0122** vs **0.0860 ± 0.0035** — a 7% reduction, inside one standard deviation |
| p99 scoring latency | **7.15 ms** against a 50 ms budget |
| Model size | ~100k parameters over a 32-position window |

The cost number is the honest one. Averaging scores across seeds made it look
like 27%; measuring each seed separately and then aggregating gave 7%, and 7% is
what the repository reports.

## Try the episode buttons

Editing the single transaction being authorised barely moves the score, and that
is the point rather than a fault. Both fraud mechanisms in the simulator are
*bursts* of three to fifteen transactions, and the features the model reads —
velocity in the last hour, merchant novelty, device novelty, amount against the
customer's own median — are properties of a sequence. A model that fired on one
unusual purchase would decline every holiday.

So the two preset buttons graft a synthetic burst onto the customer's real
history, following the pre-drift variants in `src/simulate.py`. On a legitimate
customer that takes the raw score from about 1e-05 to 0.73 and the decision from
approve to block. The injected rows are shown in purple and labelled as fiction.

## Endpoints

- `/` — the demo page
- `/score` — the real scoring endpoint (POST a transaction plus the customer's history)
- `/model-info` — what is deployed, including thresholds and embedding widths
- `/metrics` — live latency percentiles against the budget, and the decision mix
- `/health` — reports `degraded` rather than crashing when no model bundle is present
- `/docs` — generated OpenAPI documentation

## Two things this demo does differently from production

**The model bundle is baked into the image.** In production it is mounted
read-only, so the image is independent of any one training run and a model can be
rolled forward without a rebuild. A Space has no volumes, so the demo bakes it in
and is pinned to a single run.

**The audit log is ephemeral.** Every decision is still written to an
append-only JSONL log with a per-record SHA-256, but on a Space that log dies
with the container. In production it is a durable mount, because an audit trail
you can lose is not an audit trail.

## Source

<https://github.com/MelvTheGoat/Fraud-Detection-With-Sequence-Models>

The repository has the full experiment: the simulator, the leakage test suite,
the temporal splits, three architectures, three imbalance strategies, the
calibration and cost-policy study, the drift monitor, and the write-up of what
worked and what did not.
