#!/usr/bin/env bash
# Push a deployable tree to a branch Render can build.
#
# Render builds from a git branch, and `artifacts/` is gitignored -- correctly,
# since it is generated. So a Render service pointed at `main` fails on
# `COPY artifacts` with no obvious reason why. This script pushes a branch that
# does carry the bundle, with the Dockerfile and render.yaml at its root where
# Render looks for them.
#
# The branch is force-pushed and holds a single commit with no history. It is a
# build input, not a place to develop: never merge it into main.
#
#   ./deploy/push_render.sh                      # stage only, no push
#   ./deploy/push_render.sh deploy-render        # push to that branch of origin
set -euo pipefail

BRANCH="${1:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="$ROOT/build/space"

cd "$ROOT"

# Reuse the Space staging logic rather than keeping two copies in step: the
# staged tree is the same self-contained thing either way, and build_space.sh
# with no argument stages without pushing anywhere.
./deploy/build_space.sh > /dev/null

# render.yaml is read from the repository root, not from deploy/.
cp deploy/render.yaml "$STAGE/render.yaml"

# The Space README carries Hugging Face YAML front matter, which is noise on
# Render and renders as a stray table on GitHub.
cp "$ROOT/deploy/README.md" "$STAGE/README.md"

echo "Staged $(du -sh "$STAGE" | cut -f1) at $STAGE"
echo "  Dockerfile    -> $( [ -f "$STAGE/Dockerfile" ] && echo present || echo MISSING )"
echo "  render.yaml   -> $( [ -f "$STAGE/render.yaml" ] && echo present || echo MISSING )"
echo "  artifacts/    -> $(find "$STAGE/artifacts" -type f | wc -l) files"

[[ -n "$BRANCH" ]] || { echo; echo "Pass a branch name to push, e.g. deploy-render."; exit 0; }

REMOTE_URL=$(git -C "$ROOT" remote get-url origin)
echo
echo "Force-pushing a single commit to $BRANCH on $REMOTE_URL"

cd "$STAGE"
rm -rf .git
git init -q -b "$BRANCH"
git add -A
git -c user.name="$(git -C "$ROOT" config user.name || echo deploy)" \
    -c user.email="$(git -C "$ROOT" config user.email || echo deploy@localhost)" \
    commit -q -m "Deployable tree: scoring demo with the model bundle"
git push --force "$REMOTE_URL" "HEAD:refs/heads/$BRANCH"

echo
echo "Done. In Render: New > Blueprint, pick this repo, set the branch to $BRANCH."
echo "Re-run this script after retraining to ship a new bundle."
