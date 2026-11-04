# Day 1: Dev Setup + First LLM Call 🚀

**Date:** 04 Nov 2026

---

## 🎯 What I did today

- Set up Git, VS Code, uv and Python 3.15
- Created the `agentic-ai-120` repo with a clean folder structure
- Learned the full Git flow: branch → commit → push → PR → merge → pull
- Kept API keys safe with `.env`, `.gitignore` and pre-commit (gitleaks + ruff)
- Made my first Gemini API call and printed token usage
- Explored the response object, saved an audio reply, and compared token counts across languages

---

## 🧠 What I learned

### 1. uv and virtual environments

- **Python** runs the code, a **virtual environment (venv)** keeps each project's packages isolated, and **uv** manages both.
- Every project gets its own `.venv/`, so two projects that need different versions of a library never clash.
- `pyproject.toml` is the **shopping list** (packages I asked for). `uv.lock` is the **receipt** (exact versions installed), so anyone running `uv sync` gets the same setup.
- uv creates `.venv` lazily, only when it is first needed (`uv sync`, `uv add`, `uv run`).
- `uv run file.py` always uses the project's venv, so there is no need to activate it manually.
- `uv add --dev` is for tools only developers need (like `ruff`), not the app itself.

### 2. Git: the 3 areas

| Area | Command to move forward | Meaning |
|------|------------------------|---------|
| Working folder | `git add` | My normal files, Git is not saving them yet |
| Staging area | `git commit` | Changes picked for the next snapshot |
| Commit (history) | – | A permanent snapshot with author, time and message |

- `git add` stores the file **as it is at that moment**. If I edit it again, I need to `git add` again.
- Each commit links to the previous one, which forms the history chain.
- Inside `.git/`: `objects/` stores content (blobs, trees, commits), `index` is the staging area, `refs/heads/main` points to the latest commit, and `HEAD` tells which branch I am on.
- Git tracks **files, not folders**. An empty folder needs a placeholder file like `.gitkeep`.
- `git mv` moves a file and records it as a rename, instead of a delete plus a new file.

### 3. Branches and PRs

- A branch is a copy of the code **at the moment it was created**. Work on it freely while `main` stays safe.
- A new branch only sees what was committed on the branch it was created from. Files from another unmerged branch are not there.
- A **Pull Request** asks to merge a branch into `main`, with a chance to review the changes first.
- After merging on GitHub, run `git checkout main` and `git pull` to sync the laptop, then delete the old branch with `git branch -d`.
- `origin` is just a nickname for the GitHub URL. `-u` links a local branch to its GitHub branch so later I can just run `git push`.
- Uncommitted changes travel with me when I switch branches. Always check `git status` before deleting a branch.
- For small docs changes in a solo repo, committing directly to `main` is fine. Code changes go through a branch.

### 4. Secrets hygiene

- Real keys go in `.env`, which is listed in `.gitignore` and never committed.
- `.env.example` has only the key names with empty values and is safe to commit.
- Add `.env` to `.gitignore` **before** creating the `.env` file.
- If a key leaks: **revoke it immediately**, create a new one, then clean the Git history (deleting the file is not enough).
- **pre-commit** runs checks before every commit: **gitleaks** blocks hardcoded secrets, **ruff** fixes and formats code.

### 5. Gemini response object

```
response (GenerateContentResponse)          → object
 └── candidates                              → list
      └── [0] (Candidate)                    → object
           ├── finish_reason (STOP, MAX_TOKENS, SAFETY)
           └── content (Content)             → object
                ├── role ("model" / "user")
                └── parts                    → list
                     └── [0] (Part)          → object
                          ├── text
                          ├── inline_data (mime_type, data)
                          └── function_call
```

- **Class** = blueprint, **object** = the real thing built from it, **attribute** = data (no brackets), **method** = action (with brackets `()`).
- After a list, use `[0]`. After an object, use `.name`.
- A `Part` holds only **one** kind of content. Check which attribute is not `None` to know what it is.
- `response.text` is a shortcut that joins all text parts. It returns `None` when the answer is audio or an image.
- `type(obj).model_fields` lists every attribute, and `obj.model_dump(exclude_none=True)` shows what is actually filled.

### 6. Tokens

- Models read text as small chunks called **tokens** (roughly 1 token ≈ ¾ of an English word).
- **Input tokens** = my prompt, **output tokens** = the answer, **thinking tokens** = the model's internal reasoning, **total** = all of them added up.
- Tokens matter for **cost**, the **context window** limit, and **free tier rate limits** (error 429).
- `or 0` gives a default value when an attribute is `None`, so the math does not crash.

**Experiment: same sentence in three languages**

| Language | Words | Tokens | Tokens per word |
|----------|-------|--------|-----------------|
| English | 9 | 12 | 1.33 |
| Hindi | 10 | 14 | 1.40 |
| Hinglish | 10 | 15 | 1.50 |

Hinglish used about 25% more tokens than English, because romanized Hindi has no fixed spelling and gets split into more pieces. Small differences add up fast at scale.

---

## 🐛 Problems I faced (and how I fixed them)

| Problem | Fix |
|---------|-----|
| `uv` not recognized after install | Restarted the terminal so PATH reloads |
| `touch` not working | PowerShell uses `New-Item` / `Set-Content` instead |
| `git push` failed: password not supported | Signed in through the browser with Git Credential Manager |
| `git commit` opened an editor | Forgot `-m`. Wrote the message on line 1, saved and closed the tab |
| Gemini 503 (server busy) | Added a fallback list of models with `try/except` |
| Gemini 404 (old model) | Used `client.models.list()` to find current model names |
| `response.text` was `None` | It was a TTS model. The answer was audio in `inline_data` |
| GitHub 500 on push | Server-side error. The commit was safe locally, so I retried |
| `ruff` shown as legacy alias | Renamed the hook id to `ruff-check` |

---

## 💡 Biggest takeaway

Read errors from the bottom line, and never trust memory for versions or model names. Ask the tool (`autoupdate`, `models.list()`) instead.

---

## ⏭️ Next

- Put `gemini-3.8-flash` first in the model fallback list
- Update the LinkedIn "Build With Mayank" profile
- Start Day 2 of the roadmap