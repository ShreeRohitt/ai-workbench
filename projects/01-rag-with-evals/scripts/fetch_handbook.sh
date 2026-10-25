#!/usr/bin/env bash
# Fetch the GitLab handbook's People / Total Rewards pages (raw Markdown) into data/raw/.
# Pinned to one commit so chunk stats and eval results are reproducible.
# Source: https://gitlab.com/gitlab-com/content-sites/handbook (MIT, (c) GitLab B.V.)
set -euo pipefail

COMMIT="37ce9f1fc6cb7cd8a0a8cf470c0de27d2cb53329"
SECTIONS=(content/handbook/people-group content/handbook/total-rewards)

here="$(cd "$(dirname "$0")/.." && pwd)"
dest="$here/data/raw/handbook"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

git clone --filter=blob:none --no-checkout -q https://gitlab.com/gitlab-com/content-sites/handbook.git "$tmp"
git -C "$tmp" sparse-checkout set "${SECTIONS[@]}"
git -C "$tmp" checkout -q "$COMMIT"

rm -rf "$dest"
mkdir -p "$dest"
cp -R "$tmp/content/handbook/." "$dest/"
echo "$COMMIT" > "$dest/COMMIT"
echo "Fetched $(find "$dest" -name '*.md' | wc -l | tr -d ' ') pages at $COMMIT into $dest"
