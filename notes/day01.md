# Day 1: Dev Setup + First LLM Call 🚀

**Date:** 04 Nov 2026

---

## 🎯 What I did today
- Set up Git, VS Code, uv and Python 3.15
- Created the agentic-ai-120 repo with a proper folder structure
- Learned the full Git flow: branch → commit → push → PR → merge
- Kept API keys safe with .env, .gitignore and pre-commit (gitleaks + ruff)
- Made my first Gemini API call and printed token usage

---

## 🧠 What I learned

### 1. uv and virtual environments
👉 (Apne shabdon mein: venv kya hai, uv kya karta hai, pyproject.toml vs uv.lock)

### 2. Git: the 3 areas
👉 (Working folder → staging → commit. git add aur git commit ka farak)

### 3. Branches and PRs
👉 (Branch kya hai, main pe seedha kaam kyun nahi karte, PR kya hai)

### 4. Secrets hygiene
👉 (.env vs .env.example, key leak ho jaaye to kya karoge)

### 5. Gemini response object
👉 (response → candidates → content → parts. List vs object pehchaan)

### 6. Tokens
👉 (Input / output / thinking tokens. English vs Hinglish experiment ka result)

---

## 🐛 Problems I faced (and how I fixed them)
| Problem | Fix |
|---|---|
| `uv` not recognized after install | Restarted the terminal so PATH reloads |
| `touch` not working | PowerShell uses `New-Item` / `Set-Content` instead |
| Git push: password not supported | Used browser login with Git Credential Manager |
| Gemini 503 (server busy) | Added a fallback list of models with try/except |
| Gemini 404 (old model) | Used `client.models.list()` to find current models |
| `response.text` was `None` | It was a TTS model, the answer was audio in `inline_data` |

---

## 💡 Biggest takeaway
👉 (Aaj ki sabse badi seekh, ek line mein)

---

## ⏭️ Next
- Day 2: 👉 (apne plan se dekh ke likho)