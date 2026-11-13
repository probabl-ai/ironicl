#!/usr/bin/env bash
# Build the ironicl wheel from a source ref, with this branch's public README
# as the PyPI long description. Builds a wheel only: an sdist would publish the
# Rust sources.
#
# Usage: ./build_wheel.sh [ref]   (default ref: main)
# Then:  uv publish dist/*.whl
set -euo pipefail

ref="${1:-main}"
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(git -C "$here" rev-parse --path-format=absolute --git-common-dir)/.."
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

git -C "$repo" archive "$ref" | tar -x -C "$work"
cp "$here/README.md" "$work/README.md"

# Wheels for Apple Silicon should run on every macOS release Metal supports here.
export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-11.0}"
uv build --wheel --out-dir "$here/dist" "$work"

ls -l "$here/dist"
