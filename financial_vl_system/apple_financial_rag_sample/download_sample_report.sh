#!/usr/bin/env bash
set -euo pipefail
mkdir -p reports questions
wget -O reports/apple_2023_10k.pdf "https://s2.q4cdn.com/470004039/files/doc_earnings/2023/q4/filing/_10-K-Q4-2023-As-Filed.pdf"
echo "Downloaded Apple 2023 Form 10-K to reports/apple_2023_10k.pdf"
