# Deploying the demo

The scoring API in `src/api/` was built to be deployed: FastAPI, a mounted model
bundle, a health endpoint that degrades instead of crashing, an append-only audit
log, and an ONNX graph that runs on one CPU thread. What it did not have was a
way for someone without a terminal to try it, and that is what this directory
adds.

The obstacle is not hosting, it is the request shape. `/score` wants the
customer's prior transactions along with the one being authorised — 62 of them
for exact parity with training, because every position in the 32-step window
carries features built from its own 31 prior transactions. Nobody is typing 62
transactions into a form. So the demo ships a pool of real customer histories
taken from the **test** period of the simulated dataset and lets a visitor edit
only the candidate transaction.

```
deploy/
  app.py              FastAPI wrapper: imports src.api.app and adds / and /demo/*
  static/index.html   the page, no build step, no framework, no CDN
  make_demo_data.py   samples the case pool out of the simulated dataset
  Dockerfile          demo image, bundle baked in (build context = repo root)
  space_README.md     README.md for a Hugging Face Space, with its front matter
  build_space.sh      stages build/space/ and optionally pushes it
```

`app.py` attaches routes to the existing application rather than mounting a
second one, so `/score`, `/health`, `/model-info`, `/metrics` and `/docs` keep
their production paths. The page you click around in is talking to the same
endpoint a payment switch would call, with the same schema. That is the point: a
demo served through a parallel code path proves nothing about the thing you
actually deploy.

## Build it

```bash
make simulate                                        # ~10s
python -m src.experiment --seeds 0 --models gru       # the short version
python -m deploy.make_demo_data                       # writes deploy/demo_data.json
```

`--seeds 0 --models gru` trains one architecture at one seed and still writes a
complete bundle. Use `make experiment` instead if you want the artifacts to come
from the run that produced the published numbers — three seeds, three
architectures, three imbalance strategies — which takes a couple of hours on four
cores.

The bundle lands in `artifacts/`:

| file | what it is |
|---|---|
| `deployment.json` | model name, thresholds, calibration method, why this arm |
| `feature_builder.pkl` | fitted scaler, category encoders, target-encoding maps |
| `<arm>_seed<n>.pt` | the checkpoint, for explanations and as a runtime fallback |
| `model.onnx` | the exported graph that actually serves |
| `calibrator.pkl` | the isotonic map fitted on *that seed's* validation scores |

## Run it

```bash
docker build -f deploy/Dockerfile -t fraud-demo .
docker run --rm -p 7860:7860 fraud-demo
# http://localhost:7860
```

Or without Docker:

```bash
FRAUD_ARTIFACT_DIR=artifacts uvicorn deploy.app:app --port 7860
```

If the bundle is missing the process still starts: `/health` reports `degraded`
with the reason, `/score` returns 503, and the page says so instead of showing a
blank panel. That behaviour is deliberate and is tested in
`tests/test_api.py`.

## Host it

**Hugging Face Spaces** is the closest fit: it builds a Dockerfile from a git
repository, gives a public URL, and needs no credit card. One caveat worth
checking before you count on it — Hugging Face's pricing page lists "Host
ZeroGPU, Gradio & Docker Spaces" as a PRO benefit, and in mid-2026 several
people reported free accounts being refused Docker Spaces with no announcement.
CPU Basic hardware (2 vCPU, 16 GB) is free. Whether the Docker *SDK* is free on
your account, [their pricing page](https://huggingface.co/pricing) is the
authority, not this paragraph.

```bash
# create the Space in the web UI (SDK: Docker, hardware: CPU basic), then:
export HF_TOKEN=hf_...            # a write token from huggingface.co/settings/tokens
./deploy/build_space.sh your-username/fraud-scoring
```

The script stages `build/space/` with the Dockerfile and README at the root where
a Space expects them, puts the `.onnx`/`.pt`/`.pkl` files in LFS, and pushes. On
free CPU hardware a Space sleeps after a long idle period and wakes when someone
opens it, so the first request after a quiet night is slow.

**Anything else that runs a container** works the same way. Measured here, in
the container, under `--memory=1g --cpus=1`:

| | |
|---|---|
| Image size | 2.01 GB — CPU PyTorch is most of it |
| Resident memory, serving | **290 MiB** |
| Scoring latency, approve path | p50 2.0 ms, p95 3.1 ms |
| Scoring latency, blocked path | p50 8.9 ms, p95 11.7 ms — it runs the explanation |
| Startup to `model_loaded: true` | 4 s once the image is in page cache; ~35 s on the first start after a build or a registry pull |

290 MiB is the number that decides whether a plan fits, and it fits a 512 MB
tier: the staged Space image was run under `--memory=512m` and served every
endpoint at 263 MiB, so this is verified rather than extrapolated. The 2 GB image matters for build time and for registry pulls, not for
whether the thing runs. If a host gives you 512 MB, you have headroom; if it
gives you 256 MB, you do not.

The blocked path costs about 4× the approve path because it runs a backward pass
through the eager model for the contributing-transaction list. At the deployed
alert rate of 0.9% that path is roughly where the p99 lives, which is worth
knowing before you read a p99 measured only on approvals.

One unexplained observation, recorded rather than tidied away: a single first
blocked request after a container start took 2841 ms. It has not recurred in
about twenty restarts, a cold first backward pass costs 10 ms on the host, and a
controlled two-image comparison showed a startup warmup made no difference
(14.8–18.6 ms without it, 12.3–45.2 ms with). So it is not torch's lazy
initialisation, and I do not know what it was.

**Or let people run it themselves.** For a reviewer reading the repository, two
commands and a localhost URL are often better than a hosted link, because they
can change the config and retrain. The `docker run` above is the whole
instruction.

## The static route: no host at all

`deploy/export_browser_demo.py` produces a payload for a page that needs no
server. ONNX Runtime Web executes the same exported graph in the visitor's
browser, and the fitted isotonic calibrator is replayed from its 42 breakpoints
— linear interpolation between them is exactly what `sklearn` does, so this is a
replay rather than a refit.

```bash
python -m deploy.export_browser_demo --out build/browser
```

Verified in Chromium against the Python service on identical inputs:

| scenario | browser | server | difference |
|---|---|---|---|
| a fraud case, unaltered | 0.9917612 | 0.9917612 | 2.6e-08 |
| legitimate + card-testing burst | 0.7304477 | 0.7304479 | 3.0e-07 |

Float32 rounding on the same graph. Every scenario carries the server's score so
the page can display both and let a visitor see the agreement instead of
believing a claim about it.

**The features are not ported, and will not be.** Building them needs a fitted
scaler, six category encoders and out-of-time target-encoding maps. Reimplementing
that in JavaScript would mean rewriting the one component whose entire purpose is
that the online and offline paths agree bit for bit, and a parity bug there is
invisible *and* flattering — the two properties that make a bug survive. So the
real `FeatureBuilder` computes the input tensors here, for a cross product of 15
cases and 10 variants, and the static page offers those 150 scenarios and nothing
else.

The trade is explicit: the container takes any transaction you POST and needs
somewhere to run; the static page runs anywhere and takes a menu. Two costs worth
knowing before you choose it — the ONNX Runtime WebAssembly binary is about
11 MB on first load, and single-threaded wasm inference measured ~86 ms against
~2 ms for ONNX Runtime on the server.

## Two honest differences from production

**The bundle is baked into the image.** The root `Dockerfile` and
`docker-compose.yml` mount `artifacts/` read-only, so the production image is
independent of any single training run and a model rolls forward without a
rebuild. A Space has no volumes, so the demo image is pinned to one run. Two
Dockerfiles is cheaper than one with a branch in it.

**The audit log is ephemeral.** Decisions are still appended to JSONL with a
per-record SHA-256, but on a Space that file dies with the container. In
production it is a durable mount. An audit trail you can lose is not an audit
trail, and a demo is not the place to pretend otherwise.

## What the demo deliberately does not do

It does not let you upload your own transactions. The feature builder's
target-encoding maps and scaler were fitted on this simulator's merchant and
device vocabularies; feed it a different distribution and the scores are
meaningless in a way that looks plausible. A file upload would be easy and
dishonest.

It also oversamples fraud in the case list — fraud is 0.6% of the dataset, so an
unweighted sample of twenty cases would contain none. The page says so.
