# NOIZE — AI Fairness Audit Platform

**GDSC Solution Challenge 2026** | Team Noize

Detects, explains, and fixes bias in AI systems before it harms real people.

[![SDG 10](https://img.shields.io/badge/SDG%2010-Reduced%20Inequalities-red)](https://sdgs.un.org/goals/goal10)
[![SDG 16](https://img.shields.io/badge/SDG%2016-Peace%20%26%20Justice-blue)](https://sdgs.un.org/goals/goal16)
[![Gemini](https://img.shields.io/badge/Google-Gemini%20API-orange)](https://aistudio.google.com)

---

## What it does

Organizations use AI to make decisions about hiring, loans, and healthcare.
NOIZE audits those AI systems in three layers:

| Layer | What it checks | Tools |
|---|---|---|
| Pre-model audit | Is the raw data biased before training? | AIF360, Fairlearn |
| Post-model audit | Are the model's decisions fair across groups? | Fairlearn, scikit-learn |
| AI Governance | Plain-English explanations of every finding | Gemini API |

---

## Project structure

```
noize/
├── backend/    FastAPI — auth, upload, storage
├── ml/         Bias detection, fairness metrics, mitigation
└── flutter/    Mobile app
```

---

## Quick start

### 1. Clone
```bash
git clone https://github.com/YOUR_USERNAME/noize.git
cd noize
```

### 2. Set up environment
```bash
cd backend
cp .env.example .env
# Fill in SECRET_KEY, REFRESH_SECRET_KEY, GEMINI_API_KEY
```

### 3. Start ML engine (Terminal 1)
```bash
cd ml
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Start backend (Terminal 2)
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

### 5. Run Flutter app (Terminal 3)
```bash
cd flutter
flutter pub get
flutter run
```

---

## Demo dataset

`ml/data/sample_loan.csv` is included — 200 rows, Gender → Loan_Status,
Disparate Impact = 0.71 (clearly biased). Use this for your demo.

---

## Google technologies
- Gemini 1.5 Flash — bias explanations and governance reports
- Google OAuth 2.0 — user authentication
- Firebase Firestore — audit trail storage (production)

## UN SDG alignment
- **SDG 10** — detects discriminatory AI decisions in hiring, lending, healthcare
- **SDG 16** — transparent, auditable AI governance for organizations
