# 🤖 Agentic AI 120

**120 days of learning and building AI agents, in public.**

Hi, I'm **Mayank** 👋, a web developer learning Agentic AI from the ground up. This repo is my daily log: every concept, experiment and project from Day 1 to Day 120.

> 🏗️ Part of **Build With Mayank**, where I learn in public and share everything I build.

---

## 📍 Progress

| Day | Date | Topic | Status |
|-----|------|-------|--------|
| 1 | 04 Nov 2026 | Dev setup: uv, Git/GitHub flow, secrets hygiene, first Gemini API call | ✅ Done |

---

## 🗺️ Roadmap

The full 120-day roadmap is coming soon. 🚧

---

## 📁 Project Structure

```
agentic-ai-120/
├── src/agentic_ai_120/   # Reusable core code (shared across projects)
├── notebooks/            # Daily experiments and practice scripts
│   └── day01/            # Exploring the Gemini client, response object and tokens
├── notes/                # Daily learning notes
├── projects/             # Portfolio projects
├── evals/                # Testing how good the AI outputs are
├── hello_llm.py          # Day 1: first Gemini API call with model fallback
├── .env.example          # Template for API keys (copy to .env)
└── pyproject.toml        # Project dependencies (managed by uv)
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

**3. Add your API keys**

Copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```
Get a free Gemini key from [Google AI Studio](https://aistudio.google.com).

**4. Run your first LLM call**
```bash
uv run hello_llm.py
```

---

## 🛠️ Tech Stack

- **Python** with **uv** for environments and dependencies
- **Gemini API** (google-genai SDK)
- **Git + GitHub** with branches, PRs and conventional commits

---

## 🔐 Security

API keys live only in `.env`, which is git-ignored. Never commit real keys.

---

⭐ **Follow along!** Star this repo to watch the journey.