# 🤖 Agentic AI 120

**120 days of learning and building AI agents, in public.**

Hi, I'm **Mayank** 👋, a web developer learning Agentic AI from the ground up. This repo is my daily log: every concept, experiment and project from Day 1 to Day 120.

> 🏗️ Part of **Build With Mayank**, where I learn in public and share everything I build.

---

## 📍 Progress

| Day | Date | Topic | Notes | Status |
|-----|------|-------|-------|--------|
| 1 | 04 Nov 2026 | Dev setup: uv, Git/GitHub flow, secrets hygiene, first Gemini API call | [Day 1](notes/day01.md) | ✅ Done |

---

## 🗺️ Roadmap

The full 120-day roadmap is coming soon. 🚧

---

## 📁 Project Structure

```
agentic-ai-120/
├── src/agentic_ai_120/      # Reusable core code (shared across projects)
├── notebooks/               # Daily experiments and practice scripts
│   └── day01/               # Exploring the Gemini client, response object and tokens
├── notes/                   # Daily learning notes
├── projects/                # Portfolio projects
├── evals/                   # Testing how good the AI outputs are
├── hello_llm.py             # Day 1: first Gemini API call with model fallback
├── .env.example             # Template for API keys (copy to .env)
├── .pre-commit-config.yaml  # Checks for leaked secrets and code style
└── pyproject.toml           # Project dependencies (managed by uv)
```

---

## 🚀 Getting Started

**1. Clone the repo**
```bash
git clone https://github.com/letsbuildwithmayank/agentic-ai-120.git
cd agentic-ai-120
```

**2. Install dependencies** (needs [uv](https://docs.astral.sh/uv/))
```bash
uv sync
```

**3. Turn on the pre-commit checks**
```bash
uv run pre-commit install
```

**4. Add your API keys**

Copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```
Get a free Gemini key from [Google AI Studio](https://aistudio.google.com).

**5. Run your first LLM call**
```bash
uv run hello_llm.py
```

---

## 🛠️ Tech Stack

- **Python** with **uv** for environments and dependencies
- **Gemini API** (google-genai SDK)
- **Git + GitHub** with branches, PRs and conventional commits
- **pre-commit** with **gitleaks** (secret scanning) and **ruff** (linting and formatting)

---

## 🔐 Security

API keys live only in `.env`, which is git-ignored. Every commit is scanned by gitleaks so real keys never reach GitHub.

---

⭐ **Follow along!** Star this repo to watch the journey.