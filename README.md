# NOIZE — AI Fairness Auditing Platform

> **"Not just detecting bias — explaining and fixing it."**

NOIZE is a full-stack AI fairness auditing platform that detects, explains, and mitigates bias in machine learning systems. Built for organizations, researchers, and developers who need transparent, explainable, and compliant AI decision-making.

🌐 **Live Demo:** [noize-9216c.web.app](https://noize-9216c.web.app)
⚡ **Backend API:** [determined-enchantment-production-de8a.up.railway.app](https://determined-enchantment-production-de8a.up.railway.app/docs)
🤖 **ML Engine:** [noizesubmission-production.up.railway.app](https://noizesubmission-production.up.railway.app/health)

---

## The Problem

AI systems in hiring, lending, and healthcare make life-altering decisions — often with hidden, systematic bias baked in from flawed historical data. There are no easy-to-use, end-to-end tools that detect, explain, and fix this bias in one place.

---

## Our Solution

NOIZE provides a complete fairness auditing pipeline:

```
Upload Dataset → Pre-Audit → Data Mitigation → Train Model → Post-Audit → Model Mitigation → Generate Report
```

| Stage | What it does |
|---|---|
| **Pre-Audit** | Detects bias in raw datasets before training |
| **Post-Audit** | Analyzes model predictions for discriminatory patterns |
| **Mitigation** | Applies Reweighing, DIR, and threshold optimization automatically |
| **Explainability** | Google Gemini converts metrics into plain English |
| **Governance** | Full audit trail with compliance-ready PDF reports |

---

## Features

- 📂 **Dataset Upload & Analysis** — CSV/JSON with automatic schema detection and sensitive attribute flagging
- ⚖️ **Fairness Metrics Engine** — Disparate Impact, Statistical Parity, Equal Opportunity, Equalized Odds
- 🤖 **Model Training Interface** — Train classification models directly in the platform
- 🛡️ **Bias Mitigation Suite** — Reweighing, Disparate Impact Remover, Calibrated Equalized Odds
- 🧠 **Gemini Explainability Engine** — AI-generated plain English insights from Google Gemini API
- 📋 **Governance & Audit Logs** — Full decision trail with timestamps, model versions, and applied mitigations
- 📊 **Visual Dashboards** — Interactive charts and fairness score visualization
- 📄 **PDF Report Generation** — Compliance-ready downloadable reports
- 🔍 **Proxy Bias Detection** — Finds hidden discrimination in correlated features
- ⚡ **Retroactive Fixing** — Remediate deployed models post-production

---

## Tech Stack

### Frontend
| Technology | Purpose |
|---|---|
| Flutter (Dart) | Cross-platform web UI |
| Google Fonts | Typography |
| Provider | State management |
| HTTP | API communication |
| File Picker | Dataset upload |
| Firebase Hosting | Deployment |

### Backend API
| Technology | Purpose |
|---|---|
| FastAPI | REST API framework |
| Python 3.11 | Core language |
| Uvicorn + Gunicorn | Production WSGI server |
| Python-JOSE | JWT authentication |
| Passlib + Bcrypt | Password hashing |
| SlowAPI | Rate limiting |
| Pandas | Data processing |
| Google Auth | OAuth integration |
| Railway | Deployment |

### ML Engine
| Technology | Purpose |
|---|---|
| Scikit-learn | Model training and evaluation |
| Pandas / NumPy | Data manipulation |
| AIF360 | Advanced fairness metrics |
| Fairlearn | Bias mitigation algorithms |
| Plotly | Interactive visualizations |
| Matplotlib / Seaborn | Statistical charts |
| ReportLab | PDF report generation |
| Google Gemini API | AI-powered explanations |
| Railway | Deployment |

### Infrastructure
| Technology | Purpose |
|---|---|
| Docker | Containerization |
| GitHub Actions | CI/CD pipeline |
| Google Cloud Platform | Cloud infrastructure |
| Firebase Hosting | Frontend hosting |
| Railway | Backend + ML deployment |

---

## Architecture

```
┌─────────────────────────────────────┐
│         Flutter Web Frontend         │
│     (Firebase Hosting)               │
└──────────────┬──────────────────────┘
               │ HTTPS
┌──────────────▼──────────────────────┐
│         Backend API (FastAPI)        │
│     JWT Auth · Rate Limiting         │
│     Audit Logs · PDF Reports         │
│     (Railway)                        │
└──────────────┬──────────────────────┘
               │ Internal HTTP
┌──────────────▼──────────────────────┐
│         ML Engine (FastAPI)          │
├────────────────┬────────────────────┤
│   Pre-Audit    │    Post-Audit       │
│  Bias Detector │  Decision Auditor   │
│  Proxy Detect  │  Model Trainer      │
│  Mitigation    │  Mitigation         │
└────────────────┴──────┬─────────────┘
                        │
┌───────────────────────▼─────────────┐
│         Google Gemini API            │
│     AI-Powered Explanations          │
└─────────────────────────────────────┘
```

---

## Datasets Supported

NOIZE audits these real-world bias benchmark datasets out of the box:

| Dataset | Biased Feature | Affected Group | Type of Bias |
|---|---|---|---|
| Adult Income | Gender, Race | Women, minorities | Income prediction |
| COMPAS | Race | Black defendants | Risk scoring |
| German Credit | Age, Gender | Elderly, women | Credit denial |
| Healthcare | Race | Minorities | Treatment gaps |

---

## Getting Started

### Prerequisites
- Python 3.11+
- Flutter 3.x
- Docker (optional)

### Local Development

**1. Clone the repo**
```bash
git clone https://github.com/your-username/Noize_Submission.git
cd Noize_Submission
```

**2. Start ML Engine**
```bash
cd ml
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

**3. Start Backend**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8001
```

**4. Start Frontend**
```bash
cd flutter
flutter pub get
flutter run -d chrome
```

### Environment Variables

**Backend:**
```env
SECRET_KEY=your-secret-key
REFRESH_SECRET_KEY=your-refresh-secret-key
ML_ENGINE_URL=http://localhost:8000
GEMINI_API_KEY=your-gemini-api-key
ALLOWED_ORIGINS=http://localhost:8080
UPLOAD_DIR=uploads
MAX_UPLOAD_MB=50
TOKEN_EXPIRE_MINUTES=60
REFRESH_TOKEN_EXPIRE_DAYS=7
```

**ML Engine:**
```env
GEMINI_API_KEY=your-gemini-api-key
ALLOWED_ORIGINS=http://localhost:8001
```

### Docker (both services)
```bash
docker-compose up --build
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/auth/register` | Register a new user |
| POST | `/auth/login` | Login and get JWT token |
| POST | `/upload/` | Upload a CSV dataset |
| POST | `/analyze/` | Run bias analysis |
| GET | `/metrics/{result_id}` | Get fairness metrics |
| GET | `/results/{result_id}` | Get full result |
| GET | `/history/` | Get analysis history |
| POST | `/compare/` | Compare two results |
| GET | `/explain/{result_id}` | Get AI explanation |
| DELETE | `/results/{result_id}` | Delete a result |

Full interactive API docs: [/docs](https://determined-enchantment-production-de8a.up.railway.app/docs)

---

## Testing

```bash
# Backend tests (54 tests)
cd backend
python -m pytest tests/ -v

# ML engine tests (60 tests)
cd ml
python -m pytest tests/ -v
```

**Test coverage:**
- ✅ Authentication & authorization
- ✅ File upload & path traversal security
- ✅ Bias detection algorithms
- ✅ Fairness metrics computation
- ✅ Mitigation strategies
- ✅ Edge cases (empty datasets, single groups, huge files)
- ✅ API endpoints (metrics, results, history, compare, explain)

---

## Competitive Advantage

| Feature | AIF360 | What-If Tool | FairNow | **NOIZE** |
|---|---|---|---|---|
| Pre-model audit | ✅ | ✅ | ✅ | ✅ |
| Post-model audit | ✅ | ✅ | ✅ | ✅ |
| Governance layer | ❌ | ❌ | ✅ | ✅ |
| Gemini explanations | ❌ | ❌ | ❌ | ✅ 🔥 |
| Audit trail logs | ❌ | ❌ | ✅ | ✅ |
| Google tools | ❌ | ✅ | ❌ | ✅ |
| Free & open | ✅ | ✅ | ❌ paid | ✅ |
| Retroactive fix | ❌ | ❌ | ❌ | ✅ 🔥 |
| Student friendly | ❌ | ✅ | ❌ | ✅ |

---

## Team

**Team NOIZE** — Built for Google Developer Student Clubs Solution Challenge Hackathon 2025

---

## License

MIT License — free to use, modify, and distribute.
