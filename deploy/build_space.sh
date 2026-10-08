#!/usr/bin/env bash
# Stage the demo as a Hugging Face Space repository, and optionally push it.
#
# A Space is just a git repository that Hugging Face builds. It needs the
# Dockerfile at its root and a README.md carrying the YAML front matter, which is
# not how this repository is laid out, so the Space is assembled into
# build/space/ rather than pushed from here directly. Nothing in this script
# touches the GitHub remote.
#
#   ./deploy/build_space.sh                      # stage only
#   ./deploy/build_space.sh you/fraud-scoring    # stage, then push to that Space
#
# Authentication for the push: either run `hf auth login` first, or export
# HF_TOKEN with a write token from https://huggingface.co/settings/tokens.
set -euo pipefail

SPACE="${1:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="$ROOT/build/space"

cd "$ROOT"

# --- preconditions ---------------------------------------------------------- #
# The bundle is what makes this a demo rather than a 503. Checking for each file
# by name beats letting the container come up degraded and reading the logs.
missing=()
for f in deployment.json feature_builder.pkl model.onnx; do
  [[ -f "artifacts/$f" ]] || missing+=("artifacts/$f")
done
compgen -G "artifacts/*_seed*.pt" > /dev/null || missing+=("artifacts/<model>_seed<n>.pt")
[[ -f deploy/demo_data.json ]] || missing+=("deploy/demo_data.json")

if (( ${#missing[@]} )); then
  echo "Missing, so there is nothing to deploy yet:" >&2
  printf '  %s\n' "${missing[@]}" >&2
  echo >&2
  echo "Build them with:" >&2
  echo "  make simulate" >&2
  echo "  python -m src.experiment --seeds 0 --models gru   # or: make experiment" >&2
  echo "  python -m deploy.make_demo_data" >&2
  exit 1
fi

# --- stage ------------------------------------------------------------------ #
rm -rf "$STAGE"
mkdir -p "$STAGE"

cp -r src "$STAGE/src"
cp -r deploy "$STAGE/deploy"
cp -r artifacts "$STAGE/artifacts"
cp pyproject.toml requirements-serve.txt "$STAGE/"

# The Space wants these two at its root under these exact names.
cp deploy/Dockerfile "$STAGE/Dockerfile"
cp deploy/space_README.md "$STAGE/README.md"

# Staging a copy of the staging script into the Space is just noise.
rm -f "$STAGE/deploy/build_space.sh"
find "$STAGE" -name '__pycache__' -type d -prune -exec rm -rf {} +
find "$STAGE" -name '*.pyc' -delete

# Hugging Face rejects plain files over 10 MB and wants them in LFS. The whole
# bundle is currently about 1.4 MB -- the largest single file is a 327 KB ONNX
# graph -- so LFS is not configured unless something actually crosses the line.
# An earlier version of this script declared *.onnx, *.pt and *.pkl as LFS
# unconditionally, which is worse than useless: it sets the filter attribute
# whether or not git-lfs is installed, and on a machine without it the files go
# up as plain blobs that the server has been told to expect as pointers.
LARGE=$(find "$STAGE" -type f -size +10M -printf '%P\n' 2>/dev/null || true)

echo "Staged $(du -sh "$STAGE" | cut -f1) at $STAGE"
echo
echo "Test it locally first:"
echo "  docker build -f deploy/Dockerfile -t fraud-demo . && docker run --rm -p 7860:7860 fraud-demo"
echo "  open http://localhost:7860"

[[ -n "$SPACE" ]] || { echo; echo "Pass a Space id (you/name) to push."; exit 0; }

# --- push ------------------------------------------------------------------- #
echo
echo "Pushing to https://huggingface.co/spaces/$SPACE"

REMOTE="https://huggingface.co/spaces/$SPACE"
if [[ -n "${HF_TOKEN:-}" ]]; then
  # The token goes in the URL for this one push and is not written to a config.
  REMOTE="https://user:${HF_TOKEN}@huggingface.co/spaces/$SPACE"
fi

cd "$STAGE"
git init -q -b main

if [[ -n "$LARGE" ]]; then
  if ! command -v git-lfs > /dev/null; then
    echo "These files need Git LFS and git-lfs is not installed:" >&2
    echo "$LARGE" >&2
    exit 1
  fi
  git lfs install --local
  while IFS= read -r path; do git lfs track "$path"; done <<< "$LARGE"
fi

git add -A
git -c user.name="$(git -C "$ROOT" config user.name || echo deploy)" \
    -c user.email="$(git -C "$ROOT" config user.email || echo deploy@localhost)" \
    commit -q -m "Deploy fraud scoring demo"

# A Space repository already carries a commit from its own creation, so the
# first push from a freshly staged tree replaces it rather than fast-forwarding.
# Pushing to the URL directly rather than `git remote add` keeps the token out
# of .git/config, even though this staging directory is throwaway.
echo "This REPLACES the current contents of the Space."
git push --force "$REMOTE" main

echo
echo "Live shortly at https://huggingface.co/spaces/$SPACE"
