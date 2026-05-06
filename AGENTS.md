# AGENTS.md (a.k CLAUDE.md, ...)

This file provides guidance to Agentic Coding like Claude Code (claude.ai/code) or Github Copilot when working with code in this repository.

---

## Project Goal

- Build a **anomaly detection model** based-on Flow-Matching paradigm.

- Build a **multi-agent diagnostic system** on top of a completed anomaly detection model for 47 paint-shop robots.

---

## Commands

```bash
# Install / sync dependencies (includes dspy-ai, python-dotenv from Sprint 0)
uv sync

# Add a new dependency
uv add <lib>

# Run any script
uv run python app/<module>.py

# Start API server
uv run uvicorn app.main:app --reload

# Run all tests
uv run pytest tests/ -v

# Run a single test
uv run pytest tests/test_orchestrator.py::test_name -v
```

---

## Operational Rules

- **Language:** Python 3.13
- **Imports:** Relative imports only at the same level (`from .x import`). Never `..` or `...` to parent packages.
- **Dependencies:** Always `uv add <lib>`. Never invent or manually install libraries.
- **Execution:** Always `uv run ...`
- **Commits:** Propose a commit only when logic + test exist for that milestone.

**Note:** For creating plans (sprints and global plans), use the `manage-plans` skill. For implementing tasks and creating artifacts, use the `implement-atomic-task` skill. For defining standard PyTorch models, use the `define-deeplearning-model` skill.

### Commit Message Standard

**Subject line** (English, imperative):
```
feat|fix|chore|docs|refactor: short message
```

**Description** (English bullet points):
- What changed
- Why it changed
- Verification / test results (must reference specific files)