#!/usr/bin/env bash
set -euo pipefail

STUDY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KIT_ROOT="$(dirname "$STUDY_DIR")"
cd "$KIT_ROOT"
exec python -m judge_query.run --study "${STUDY_DIR}/study.toml" "$@"
