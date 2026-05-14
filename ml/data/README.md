# NOIZE — ml/data/

This folder holds the CSV datasets used by the ML engine.

## Included

| File | Rows | Protected attr | Target | Bias type |
|---|---|---|---|---|
| `sample_loan.csv` | 200 | Gender | Loan_Status | Gender → approval rate (DI = 0.71) |

## Download real datasets

```bash
bash download_datasets.sh
```

This downloads:
- `adult.csv` — UCI Adult Income (gender/race bias in income prediction)
- `compas.csv` — ProPublica COMPAS (race bias in recidivism scoring)
- `german.csv` — UCI German Credit (age/gender bias in credit scoring)

## For your demo

Use `sample_loan.csv` — it's already here, no download needed, and it
shows a clear disparate impact ratio of **0.71** (below the 0.8 threshold)
that judges will see detected immediately.

Select in the Flutter app:
- Target variable: `Loan_Status`
- Protected attribute: `Gender`
