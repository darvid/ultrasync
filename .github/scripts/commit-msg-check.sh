#!/usr/bin/env bash
set -euo pipefail

msg_file="${1:-}"
if [[ -z "$msg_file" || ! -f "$msg_file" ]]; then
  echo "commit-msg-check: missing commit message file" >&2
  exit 1
fi

if rg -n "^fix\\(ci\\):" "$msg_file" >/dev/null 2>&1; then
  echo "commit-msg-check: use 'ci:' or 'chore(ci):' instead of 'fix(ci):'" >&2
  exit 1
fi
