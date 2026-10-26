#!/usr/bin/env bash
# Download the IBM HR Analytics Employee Attrition dataset (fictional data created by IBM).
# Checksum-pinned so every run uses exactly the same file.
set -euo pipefail

URL="https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv"
SHA256="a5c31e38bd7fafc9bc333884eb181b06b41b8e5e488e8f7ccb27199fb3be7659"

dest="$(cd "$(dirname "$0")/.." && pwd)/data/raw/attrition.csv"
mkdir -p "$(dirname "$dest")"
curl -sSfL "$URL" -o "$dest"
echo "$SHA256  $dest" | shasum -a 256 -c -
