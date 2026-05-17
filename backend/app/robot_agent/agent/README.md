# Robot Diagnostic Agent — Architecture Guide

## Overview

The agent package implements a **multi-agent diagnostic system** for 47
paint-shop robots.  At its core is a **Planning-First** workflow driven
by DSPy (Declarative Self-Improving Python) and a **skill system** that
provides the LLM with domain-specific diagnostic sub-routines.

---

## Pipeline: Triage → Diagnostic → Systemic

```
┌────────────┐   ┌──────────────────┐   ┌──────────────────┐
│   Triage   │──→│ DiagnosticAgent  │──→│ SystemicAgent    │
│ (poller)   │   │ (per-robot)      │   │ (multi-robot)    │
└────────────┘   └──────────────────┘   └──────────────────┘
```

1. **Triage (upstream)** — A poller detects raw sensor anomalies and creates
   a `DiagnosticIncident`.
2. **DiagnosticAgent** — Runs the Planning-First pipeline for a single robot
   incident.  Uses DSPy + skills to build and execute a diagnosis plan.
3. **SystemicAgent** — When multiple robots fail simultaneously, escalates to
   a factory-level analysis.  The `SystemicAgentSignature` DSPy signature
   determines whether the root cause is systemic (e.g. central pump failure).

---

## Planning-First Workflow

Every incident session follows two phases enforced by
`DiagnosticAgent.run_incident()`:

### Phase 1 — Plan (`_run_plan_phase`)

| Step | Action |
|------|--------|
| 1    | LLM must call `load_skill("create_plan")` as its **very first action**. |
| 2    | The `create_plan` skill returns a task roadmap (JSON) stored in `PlanManager`. |
| 3    | After the plan is created, phase 1 ends. |

### Phase 2 — Execute (`_run_execute_phase`)

| Step | Action |
|------|--------|
| 1    | Follow plan tasks sequentially (Routing → Diagnosis → Action). |
| 2    | Use tools: `load_skill`, `query_pm_cards`, `hitl_check`, `update_task_status`. |
| 3    | Mark each task `completed` via `update_task_status`. |
| 4    | Call `complete` with a final summary **only after all tasks are done**. |

The agent state machine:

```
idle  ──start_session──→  running  ──complete──→  completed
                               │──blocked──→  blocked
                               │──failed──→   failed
```

---

## DSPy Integration

DSPy provides the LM interaction layer:

- **`AgentStepSignature`** — A `dspy.Signature` defining inputs (system prompt,
  incident context, loaded skills, plan, past steps) and outputs (reasoning,
  action, action_input).  Used with `dspy.ChainOfThought`.
- **`SystemicAgentSignature`** — Separate signature for multi-robot analysis
  with outputs: `reasoning`, `root_cause_candidate`, `recommended_action`,
  `is_systemic`.
- **`dspy.LM`** — Instantiated by `get_lm()` in `llm_config.py` using
  environment variables (`LM_PROVIDER`, `LM_MODEL`, `LM_API_KEY`).

The LLM call pattern:

```python
predictor = dspy.ChainOfThought(AgentStepSignature)
with dspy.context(lm=lm, cache=False):
    result = predictor(
        system_prompt=...,
        incident_context=...,
        available_skills=...,
        loaded_skills=...,
        current_plan=...,
        past_steps=...,
    )
```

---

## Skill System

Skills are Markdown files that guide the LLM through domain-specific
diagnostic sub-routines.

### Structure

```
agent_knowledge/skills/
├── _SKILL_TEMPLATE.md          # Template for new skills
├── create_plan/SKILL.md        # Plan creation — loaded first
├── routing/SKILL.md            # Route incident → correct diagnose skill
├── diagnose_surge/SKILL.md     # Pressure surge diagnosis
├── diagnose_dropout/SKILL.md   # Signal dropout diagnosis
├── diagnose_deviation/SKILL.md # Deviation diagnosis
├── diagnose_instability/SKILL.md
├── diagnose_transition/SKILL.md
├── correlate_history/SKILL.md  # Cross-reference past incidents
└── verify_defect/SKILL.md      # Confirmation procedure
```

### How Skills Are Loaded

1. **Discovery** — `skills.py` scans `agent_knowledge/skills/` for directories
   containing a `SKILL.md` file.  The results are stored in `ALLOWED_SKILLS`
   (a frozenset) and the in-memory `skill_catalog`.
2. **Loading** — The LLM calls `load_skill(<name>)`; `LoadSkillTool.__call__`
   validates the name against `ALLOWED_SKILLS` and reads the `SKILL.md` content.
3. **Context injection** — Loaded skill content is appended to
   `loaded_skills_content` and fed back into the DSPy predictor's
   `loaded_skills` input field in subsequent turns.

---

## RAG with ChromaDB

The `rag_tools.py` module provides two DSPy-compatible tools:

| Tool | Function |
|------|----------|
| `QueryPMCardsTool` | Searches the `pm_cards` ChromaDB collection for similar past incidents. |
| `IndexPMCardsTool` | Upserts a new PM Card into the vector store. |

Both use `get_collection("pm_cards")` from `chroma/client.py`.  The
collection stores Preventive Maintenance card text and metadata, and is
queried via semantic similarity.

Integration flows:

```
LLM decides → load_skill("routing")
            → query_pm_cards("paint pressure spike R-07")
            → ChromaDB returns top-5 similar historical cards
            → LLM reasons with retrieved context
```

---

## How to Add a New Diagnostic Skill

1. **Create the skill directory and file:**

   ```
   backend/agent_knowledge/skills/<your_skill_name>/
   └── SKILL.md
   ```

   Start from the template at `backend/agent_knowledge/skills/_SKILL_TEMPLATE.md`.

2. **Write the `SKILL.md` with YAML frontmatter:**

   ```markdown
   ---
   name: <your_skill_name>
   description: <one-line description>
   required_tools: []
   ---

   # Skill Title

   ## Purpose
   ...

   ## Diagnostic Steps
   ### Step 1: ...
   ### Step 2: ...

   ## Completion
   Call `update_task_status` when done.
   ```

3. **The skill is auto-discovered.**  `skills._discover_allowed_skills()` runs
   at import time and picks up any new directory with a `SKILL.md` — no code
   changes needed.

4. **If the skill needs a new tool**, implement the tool class in
   `diagnostic_agent.py` (or `rag_tools.py` for RAG tools), add it to
   `AVAILABLE_TOOLS`, and handle the action in `_run_execute_phase`.

---

## File Reference

| File | Purpose |
|------|---------|
| `__init__.py` | Package exports (`DiagnosticAgent`, `create_agent`) |
| `diagnostic_agent.py` | Core agent: state machine, plan & execute phases |
| `systemic_agent.py` | Multi-robot systemic analysis signature |
| `skills.py` | Skill discovery, loading, `LoadSkillTool` |
| `rag_tools.py` | ChromaDB RAG query/index tools |
| `llm_config.py` | LLM configuration and `dspy.LM` factory |
