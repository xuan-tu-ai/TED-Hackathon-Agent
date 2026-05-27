"""Agent knowledge base for the WeldBot AI Diagnostic Agent.

This package provides the knowledge base consumed by the multi-agent system
for welding robot anomaly diagnosis. It is organised into three parts:

    AGENTS.md       System prompt that defines agent role, rules, and flow.
    skills/         Diagnostic skill definitions (SKILL.md) loaded
                    progressively by the DiagnosticAgent at runtime.
    rules/          Policy rule files (severity, HITL) enforced at the
                    tool-call / framework level.

Skills are structured as directories under ``skills/``, each containing a
single ``SKILL.md`` file with YAML front-matter (``name``, ``description``,
``required_tools``).  New skills should follow the template at
``skills/_SKILL_TEMPLATE.md``.
"""