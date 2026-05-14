#!/bin/bash
# ============================================================
# NOIZE — download_datasets.sh
# Downloads the 4 real fairness datasets into ml/data/
# Run once after cloning: bash ml/data/download_datasets.sh
# ============================================================

set -e
cd "$(dirname "$0")"

echo "Downloading datasets into ml/data/ ..."

# 1. Adult Income (UCI) — gender + race bias in income prediction
echo "  → adult.csv"
curl -sL "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data" \
     -o adult.csv
echo "     Done ($(wc -l < adult.csv) rows)"

# 2. COMPAS Recidivism (ProPublica) — race bias in criminal justice
echo "  → compas.csv"
curl -sL "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv" \
     -o compas.csv
echo "     Done ($(wc -l < compas.csv) rows)"

# 3. German Credit (UCI) — age + gender bias in credit scoring
echo "  → german.csv"
curl -sL "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data" \
     -o german.csv
echo "     Done ($(wc -l < german.csv) rows)"

echo ""
echo "✓ All datasets downloaded."
echo "  sample_loan.csv is already included (synthetic, no download needed)."
echo ""
echo "Recommended for demo: use sample_loan.csv (Gender → Loan_Status)"
echo "  It has a clear 0.71 disparate impact ratio — judges will see"
echo "  bias detected immediately."
