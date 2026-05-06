# Agent Knowledge Base

## Structure

```
agent_knowledge/
├── AGENTS.md           # System prompt base
├── skills/             # Diagnostic skills (SKILL.md per skill)
│   ├── create_plan/    # Planning skill
│   ├── routing/       # Signal family routing skill
│   ├── diagnose_surge/ # SURGE family diagnosis
│   ├── correlate_history/ # History correlation
│   └── check_fmea/    # FMEA lookup
└── rules/              # Policy rules
    ├── hitl-rules.md   # HITL gate rules
    └── severity-policy.md # Severity scoring
```

## Skills
Each skill is a SKILL.md file loaded progressively by DiagnosticAgent.

## Rules
Markdown files defining agent behavior policies.
