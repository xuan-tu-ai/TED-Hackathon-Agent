"""Primary diagnostic agent for single-robot anomaly incidents.

Implements the **Planning-First** workflow:

1. **Plan phase** — The agent loads the ``create_plan`` skill and builds a
   task roadmap tailored to the incident.
2. **Execute phase** — The agent follows the plan sequentially, using DSPy
   tools (``load_skill``, ``query_pm_cards``, ``hitl_check``,
   ``update_task_status``) to gather evidence and reach a diagnosis.
3. **Completion** — Once all tasks are marked ``completed`` the agent outputs
   a final summary.

The :class:`DiagnosticAgent` is the central coordinator.  Use
:func:`create_agent` to instantiate it and :meth:`DiagnosticAgent.run_incident`
to process a single :class:`DiagnosticIncident`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, ClassVar

import ast
import dspy
import json
import re
from .event_emitter import AgentEventEmitter
from .hitl_gate import HitlGate
from .plan_manager import PlanManager
from .config import config
from .rag_tools import QueryPMCardsTool
from .session_trace import SessionTraceWriter
from .signal_tools import QuerySensorStatsTool
from .skills import LoadSkillTool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("diagnostic_agent")

AGENTS_MD_PATH = config.agents_md_path
HITL_RULES_MD_PATH = config.rules_dir / "hitl-rules.md"

@dataclass
class DiagnosticIncident:
    """A single anomaly incident detected for one robot.

    Parameters
    ----------
    incident_id : str
        Unique identifier (e.g. ``"INC-20250501-001"``).
    robot_code : str
        Robot identifier (e.g. ``"R-07"``).
    abnormal_family : str
        High-level anomaly category (e.g. ``"paint_pressure"``).
    abnormal_subtype : str
        Specific anomaly subtype (e.g. ``"pressure_spike"``).
    sensor : str
        Sensor that triggered the alert.
    anomaly_score : float
        Severity / confidence score of the anomaly.
    severity_hint : str
        Human-readable severity label (e.g. ``"critical"``).
    context : dict
        Additional incident context from upstream perception.
    """

    incident_id: str
    robot_code: str
    abnormal_family: str
    abnormal_subtype: str
    sensor: str
    anomaly_score: float
    severity_hint: str
    per_sensor_scores: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentStep:
    """A single step recorded during an agent session.

    Parameters
    ----------
    step_id : str
        Unique step identifier (e.g. ``"sess-INC-001-step-1"``).
    action : str
        Tool call string (e.g. ``"load_skill(create_plan)"``).
    skill_loaded : str | None
        Skill name if the action was ``load_skill``.
    reasoning : str
        LLM-generated reasoning for this step.
    timestamp : datetime
        When the step was recorded.
    """

    step_id: str
    action: str
    skill_loaded: str | None
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)

class AgentStepSignature(dspy.Signature):
    """DSPy signature for the Paint Robot Diagnostic Agent.

    Guides the LLM through the **Planning-First** workflow. The
    agent is instructed to create a plan first, then execute tasks
    sequentially using the available tools.

    Inputs
    ------
    system_prompt : str
        The core ``AGENTS.md`` rules.
    incident_context : str
        The current incident description.
    loaded_skills : str
        Contents of the skills loaded so far.
    available_skills : str
        Catalog of all available skills with metadata.
    current_plan : str
        The diagnostic plan as JSON with task statuses.
    past_steps : str
        Previous actions and observations.

    Outputs
    -------
    reasoning : str
        Thought process for the next action.
    action : str
        Tool to call (one of ``load_skill``, ``query_pm_cards``,
        ``hitl_check``, ``update_task_status``, ``create_plan``,
        ``complete``).
    action_input : str
        Argument for the chosen tool.
    """
    system_prompt = dspy.InputField(desc="The core AGENTS.md rules.")
    incident_context = dspy.InputField(desc="The current incident context.")
    loaded_skills = dspy.InputField(desc="The contents of the skills you have loaded so far.")
    available_skills = dspy.InputField(desc="List of all available skills and their metadata. Choose from these ONLY.")
    current_plan = dspy.InputField(desc="The current diagnostic plan and task statuses. Follow this roadmap.")
    past_steps = dspy.InputField(desc="Previous actions and their observations. Pay close attention to what you already did.")

    reasoning = dspy.OutputField(desc="Your thought process for what to do next based on the incident context and past steps.")
    action = dspy.OutputField(desc="The tool to call. Must be exactly one of: load_skill, query_pm_cards, query_sensor_stats, hitl_check, update_task_status, create_plan, complete.")
    action_input = dspy.OutputField(desc="The argument for the tool. \n- load_skill: 'skill_name'\n- update_task_status: '{\"task_id\": 1, \"status\": \"completed\"}'\n- create_plan: Use only to initialize or radically change the roadmap.\n- complete: 'summary_text'. Use ONLY when ALL plan tasks are 'completed'.")

class DiagnosticAgent:
    """Orchestrates the Planning-First diagnostic workflow for a robot incident.

    The agent proceeds in two phases:

    - **Plan phase** — Loads ``create_plan`` skill and builds a task roadmap.
    - **Execute phase** — Executes plan tasks using DSPy tools.

    State transitions: ``idle`` → ``running`` → ``completed`` | ``blocked`` | ``failed``.

    Attributes
    ----------
    SYSTEM_PROMPT : str
        Content of ``AGENTS.md`` used as the system prompt.
    TIER2_RULES : str
        Human-in-the-loop escalation rules from ``hitl-rules.md``.
    AVAILABLE_TOOLS : list[str]
        Tool names the LLM may invoke.
    """

    SYSTEM_PROMPT: ClassVar[str] = AGENTS_MD_PATH.read_text(encoding="utf-8")
    TIER2_RULES: ClassVar[str] = HITL_RULES_MD_PATH.read_text(encoding="utf-8")
    AVAILABLE_TOOLS: ClassVar[list[str]] = [
        "load_skill",
        "query_pm_cards",
        "query_sensor_stats",
        "hitl_check",
        "update_task_status",
        "create_plan",
        "complete",
    ]

    def __init__(self):
        self.load_skill_tool = LoadSkillTool()
        self.session_id: str | None = None
        self.current_incident: DiagnosticIncident | None = None
        self.steps: list[AgentStep] = []
        self.context: dict[str, Any] = {}
        self.state: str = "idle"
        self._trace_writer: SessionTraceWriter | None = None
        self.emitter = AgentEventEmitter()
        self.plan = PlanManager(self.emitter)
        self.hitl_gate = HitlGate(self.emitter)
        self.current_skill: str | None = None
        self.loaded_skills_set: set[str] = set()
        self.skill_catalog: str = self._load_skill_catalog()

    def _load_skill_catalog(self) -> str:
        """Build a human-readable catalog of available skills with metadata.

        Reads each ``SKILL.md`` YAML frontmatter (``---...---``) and
        returns a formatted string for inclusion in the DSPy prompt.
        """
        catalog = []
        skills_dir = config.skills_base
        if not skills_dir.exists():
            return "No skills found."

        for skill_dir in skills_dir.iterdir():
            if skill_dir.is_dir():
                skill_md = skill_dir / "SKILL.md"
                if skill_md.exists():
                    content = skill_md.read_text(encoding="utf-8")
                    match = re.search(r"---(.*?)---", content, re.DOTALL)
                    if match:
                        metadata = match.group(1).strip()
                        catalog.append(f"- Skill: {skill_dir.name}\n  Metadata: {metadata}")
                    else:
                        catalog.append(f"- Skill: {skill_dir.name} (No metadata found)")
        return "\n\n".join(catalog)

    def start_session(self, incident: DiagnosticIncident) -> str:
        """Initialise a new diagnostic session for the given incident.

        Resets all internal state, the plan, the HITL gate, and
        prepares a :class:`SessionTraceWriter`. Returns the session ID.
        """
        self.session_id = f"sess-{incident.incident_id}"
        self.current_incident = incident
        self.steps = []
        self.context = {}
        self.state = "idle"
        self.hitl_gate.hitl_pending = {}
        self.hitl_gate.incident_id = incident.incident_id
        self.hitl_gate.robot_code = incident.robot_code
        self.current_skill = None
        self.plan.reset()
        self.plan.set_incident_id(incident.incident_id)
        self.loaded_skills_set = set()
        self._trace_writer = SessionTraceWriter(incident.incident_id)
        logger.info(f"Session {self.session_id} started for incident {incident.incident_id}")
        return self.session_id

    def step(self, action: str, reasoning: str, skill_loaded: str | None = None) -> AgentStep:
        """Record an agent step, trace it, and broadcast via the event emitter."""
        step = AgentStep(
            step_id=f"{self.session_id}-step-{len(self.steps) + 1}",
            action=action,
            skill_loaded=skill_loaded,
            reasoning=reasoning,
        )
        self.steps.append(step)
        logger.info(f"Step {step.step_id}: {action} (skill={skill_loaded})")
        if self._trace_writer is not None:
            self._trace_writer.write_step(
                step_id=step.step_id,
                action=action,
                reasoning=reasoning,
                skill_loaded=skill_loaded,
            )
        
        parts = action.split("(", 1)
        tool_name = parts[0]
        tool_input = parts[1][:-1] if len(parts) > 1 and action.endswith(")") else ""

        self.emitter.emit_step({
            "type": "agent_step",
            "session_id": self.session_id,
            "incident_id": self.current_incident.incident_id if self.current_incident else None,
            "step_id": step.step_id,
            "action": action,
            "tool_name": tool_name,
            "tool_input": tool_input,
            "reasoning": reasoning,
            "skill_loaded": skill_loaded,
            "observation": None,
            "timestamp": step.timestamp.isoformat(),
        })

        if skill_loaded is not None:
            self.current_skill = skill_loaded

        return step

    def _broadcast_observation(self, step_id: str, observation: str) -> None:
        """Emit an observation update for a given step via the event bus."""
        self.emitter.emit_observation({
            "type": "agent_step_update",
            "session_id": self.session_id,
            "incident_id": self.current_incident.incident_id if self.current_incident else None,
            "step_id": step_id,
            "observation": observation,
        })

    def load_skill(self, skill_name: str) -> str:
        """Load a diagnostic skill's content by name.

        Delegates to :class:`LoadSkillTool`.
        """
        content = self.load_skill_tool(skill_name)
        return content

    def hitl_check(self, action: str, parameters: dict[str, Any], reasoning: str) -> dict[str, Any]:
        """Run a human-in-the-loop check via the :class:`HitlGate`.

        Returns the gate result and updates ``self.state`` to
        ``blocked`` / ``failed`` / ``running`` accordingly.
        """
        incident_summary = ""
        if self.current_incident:
            incident_summary = (
                f"Robot {self.current_incident.robot_code}, "
                f"{self.current_incident.abnormal_family}/{self.current_incident.abnormal_subtype}, "
                f"score={self.current_incident.anomaly_score}"
            )

        result = self.hitl_gate.check(
            action=action,
            parameters=parameters,
            reasoning=reasoning,
            step_count=len(self.steps),
            current_skill=self.current_skill,
            incident_summary=incident_summary,
        )

        if result.get("permanently_blocked") or result.get("blocked"):
            if result.get("timeout") or result.get("resolved") is False:
                if result.get("timeout") or (
                    not result.get("resolved") and not result.get("permanently_blocked")
                ):
                    self.state = "failed"
                else:
                    self.state = "blocked"
            else:
                self.state = "running" if result.get("resolved") else "blocked"
        return result

    def _parse_llm_output(self, result) -> tuple[str, str, str]:
        """Extract reasoning, action, and action_input from a DSPy prediction.

        Strips ``<think>...</think>`` tags from reasoning and attempts
        to parse JSON-wrapped ``action_input`` values.
        """
        reasoning = getattr(result, 'reasoning', "No reasoning provided.")
        reasoning = re.sub(r"<think>.*?</think>", "", reasoning, flags=re.DOTALL).strip()

        raw_action = getattr(result, 'action', "").strip()
        raw_input = getattr(result, 'action_input', "").strip()

        if "{" in raw_input and "}" in raw_input:
            try:
                json_match = re.search(r"(\{.*\})", raw_input, re.DOTALL)
                json_str = json_match.group(1) if json_match else None
                if json_str:
                    try:
                        data = json.loads(json_str)
                    except json.JSONDecodeError:
                        try:
                            data = ast.literal_eval(json_str)
                        except (ValueError, SyntaxError):
                            data = None
                    if isinstance(data, dict):
                        raw_input = json.dumps(data)
            except Exception as exc:
                logger.debug("action_input JSON parse skipped: %s — treating as raw string", exc)

        if not raw_action and "(" in raw_input:
            match = re.match(r"(\w+)\((.*)\)", raw_input)
            if match:
                raw_action = match.group(1)
                raw_input = match.group(2)

        action = raw_action
        action_input = raw_input.strip("'\" ")
        return reasoning, action, action_input

    def _run_plan_phase(self, incident_context: str, loaded_skills_content: str, past_steps_list: list[str]) -> tuple[str, str, list[str]]:
        """Execute the planning sub-loop.

        Runs up to 5 LLM turns forcing the agent to load the
        ``create-plan`` skill and build a task roadmap. Returns
        updated loaded-skills content and step history.
        """
        lm = config.get_lm()
        predictor = dspy.ChainOfThought(AgentStepSignature)

        plan_turn_count = 0
        max_plan_turns = 5
        has_plan = False
        past_steps_str = "None yet."
        empty_action_count = 0
        max_empty_actions = 3

        while plan_turn_count < max_plan_turns and self.state not in ["blocked", "completed", "failed"]:
            plan_turn_count += 1
            past_steps_str = "\n".join(past_steps_list) if past_steps_list else "None yet."

            phase_instruction = (
                "CURRENT PHASE: PLANNING. "
                "Your goal is to create a roadmap for this incident. Your very first action MUST be 'load_skill(\"create-plan\")'. "
                "Do NOT perform routing or diagnosis until the plan is created."
            )
            full_prompt = f"{self.SYSTEM_PROMPT}\n\n{phase_instruction}"

            logger.info(f"--- LLM Plan Turn {plan_turn_count} (Incident: {self.current_incident.incident_id if self.current_incident else 'unknown'}) ---")
            try:
                t0 = time.time()
                with dspy.context(lm=lm, cache=False):
                    result = predictor(
                        system_prompt=full_prompt,
                        incident_context=incident_context,
                        available_skills=self.skill_catalog,
                        loaded_skills=loaded_skills_content,
                        current_plan=json.dumps(self.plan.current_plan, indent=2) if self.plan.current_plan else "No plan created yet.",
                        past_steps=past_steps_str
                    )
                elapsed = time.time() - t0
                if elapsed < 0.05:
                    logger.error("LLM plan response in %.1fms — TOO FAST, not a real API call! cache=False may be ignored by DSPy.", elapsed * 1000)
                else:
                    logger.info("LLM plan response in %.1fms", elapsed * 1000)

                reasoning, action, action_input = self._parse_llm_output(result)

                logger.info(f"Plan turn {plan_turn_count}: {action}({action_input})")

                if not action:
                    empty_action_count += 1
                    logger.warning("Empty action from LLM in plan phase (attempt %d/%d). reasoning=%s action=%r",
                                   empty_action_count, max_empty_actions, reasoning[:200] if reasoning else "NONE", action)
                    if empty_action_count >= max_empty_actions:
                        logger.error("LLM returned empty action %d times in plan phase — stopping", max_empty_actions)
                        self.state = "failed"
                        break
                    continue

            except Exception as e:
                logger.exception("LLM call failed in plan phase")
                self.step("error", f"LLM error: {e}")
                self.state = "failed"
                break

            skill_loaded = action_input if action == "load_skill" else None
            step_info = self.step(action=f"{action}({action_input})", reasoning=reasoning, skill_loaded=skill_loaded)
            current_step_id = step_info.step_id

            if action == "complete":
                if not has_plan:
                    logger.error("LLM called 'complete' without creating a plan — agent failed")
                    self.state = "failed"
                    break
                obs = "Diagnosis concluded. Ready for dispatch."
            elif action == "load_skill":
                skill_name = action_input.split("/")[-1].replace(".md", "").strip()
                try:
                    content = self.load_skill(skill_name)
                    loaded_skills_content += f"\n--- SKILL: {skill_name} ---\n{content}\n"
                    obs = f"Loaded skill {skill_name} successfully."
                    self.loaded_skills_set.add(skill_name)
                    if skill_name == "create-plan":
                        has_plan = True
                        self._broadcast_plan_created()
                except Exception as e:
                    obs = f"Failed to load skill: {e}"
                self._broadcast_observation(current_step_id, obs)
                past_steps_list.append(f"Turn {plan_turn_count}: Action: {action}({action_input}) -> Observation: {obs}")
            else:
                obs = f"Unexpected action in plan phase: {action}"
                self._broadcast_observation(current_step_id, obs)
                past_steps_list.append(f"Turn {plan_turn_count}: Action: {action}({action_input}) -> Observation: {obs}")

            if has_plan:
                break

        if plan_turn_count >= max_plan_turns and not has_plan:
            logger.error("Plan phase reached max turns without creating a plan — agent failed")
            self.state = "failed"

        return loaded_skills_content, past_steps_str, past_steps_list

    def _run_execute_phase(self, incident_context: str, loaded_skills_content: str, past_steps_list: list[str]) -> list[str]:
        """Execute the diagnosis sub-loop.

        Follows the plan tasks sequentially (Routing → Diagnosis → Action),
        using DSPy tools for evidence gathering.  Runs up to 20 turns
        and stops when the agent calls ``complete`` or the state becomes
        ``blocked`` / ``failed``.
        """
        lm = config.get_lm()
        predictor = dspy.ChainOfThought(AgentStepSignature)
        execute_turn_count = 0
        max_execute_turns = 20
        empty_action_count = 0
        max_empty_actions = 3

        while execute_turn_count < max_execute_turns and self.state not in ["blocked", "completed", "failed"]:
            execute_turn_count += 1

            current_task = self.plan.get_current_task()
            if current_task:
                logger.info(f"Executing Task: {current_task.get('name')}")

            past_steps_str = "\n".join(past_steps_list) if past_steps_list else "None yet."

            current_task_info = ""
            current_task = self.plan.get_current_task()
            if current_task:
                current_task_info = (
                    f"CURRENT TASK: {current_task.get('name')} "
                    f"(Goal: Perform this task using available tools. "
                    f"You may need multiple steps for one task. "
                    f"Stay on this task until complete, then call "
                    f"update_task_status({{task_id: {current_task.get('id')}, status: 'completed'}})."
                )

            multi_step_note = (
                "One plan task may require multiple steps (DSPy turns). "
                "Stay on this task until its goal is achieved, then call "
                "update_task_status({task_id: N, status: 'completed'}) "
                "before moving to Task N+1. Do NOT skip tasks."
            )
            phase_instruction = (
                "CURRENT PHASE: EXECUTION. "
                "Follow the plan items sequentially. Use tools (query_pm_cards, hitl_check, etc.) to gather evidence. "
                f"{current_task_info} "
                f"{multi_step_note}"
            )
            full_prompt = f"{self.SYSTEM_PROMPT}\n\n{phase_instruction}"

            try:
                t0 = time.time()
                with dspy.context(lm=lm, cache=False):
                    result = predictor(
                        system_prompt=full_prompt,
                        incident_context=incident_context,
                        available_skills=self.skill_catalog,
                        loaded_skills=loaded_skills_content,
                        current_plan=json.dumps(self.plan.current_plan, indent=2) if self.plan.current_plan else "Plan lost.",
                        past_steps=past_steps_str
                    )
                elapsed = time.time() - t0
                if elapsed < 0.05:
                    logger.error("LLM execute response in %.1fms — TOO FAST, not a real API call! cache=False may be ignored by DSPy.", elapsed * 1000)
                else:
                    logger.info("LLM execute response in %.1fms", elapsed * 1000)

                reasoning, action, action_input = self._parse_llm_output(result)

                logger.info(f"Step action: {action}({action_input})")

                if not action:
                    empty_action_count += 1
                    logger.warning("Empty action from LLM in execute phase (attempt %d/%d). reasoning=%s action=%r",
                                   empty_action_count, max_empty_actions, reasoning[:200] if reasoning else "NONE", action)
                    if empty_action_count >= max_empty_actions:
                        logger.error("LLM returned empty action %d times — stopping agent", max_empty_actions)
                        self.state = "failed"
                        break
                    continue

            except Exception as e:
                logger.exception("LLM call failed in execute phase")
                self.step("error", f"LLM error: {e}")
                self.state = "failed"
                break

            skill_loaded = action_input if action == "load_skill" else None
            step_info = self.step(action=f"{action}({action_input})", reasoning=reasoning, skill_loaded=skill_loaded)
            current_step_id = step_info.step_id

            if action == "complete":
                summary = action_input
                obs = f"FINAL DIAGNOSTIC SUMMARY:\n{summary}"
                self._broadcast_observation(current_step_id, obs)
                self.state = "completed"
                past_steps_list.append(f"Turn {execute_turn_count}: Action: {action} -> Summary: {summary}")
                break
            elif action == "update_task_status":
                try:
                    try:
                        data = json.loads(action_input)
                    except json.JSONDecodeError:
                        data = ast.literal_eval(action_input)
                    t_id = data.get("task_id")
                    t_status = data.get("status", "completed")
                    self.update_plan_task(t_id, t_status)
                    obs = f"Task {t_id} status updated to {t_status}."
                except Exception as e:
                    obs = f"Failed to update task: {e}"
            elif action == "create_plan":
                self._parse_plan_from_template()
                obs = "Diagnostic plan created/reset based on template."
            elif action == "load_skill":
                skill_name = action_input.split("/")[-1].replace(".md", "").strip()
                try:
                    content = self.load_skill(skill_name)
                    loaded_skills_content += f"\n--- SKILL: {skill_name} ---\n{content}\n"
                    obs = f"Loaded skill {skill_name} successfully."
                    self.loaded_skills_set.add(skill_name)
                except Exception as e:
                    obs = f"Failed to load skill: {e}"
            elif action == "query_pm_cards":
                tool = QueryPMCardsTool()
                obs = str(tool(action_input))
            elif action == "query_sensor_stats":
                tool = QuerySensorStatsTool()
                obs = str(tool(action_input))
            elif action == "hitl_check":
                try:
                    try:
                        params = json.loads(action_input)
                    except json.JSONDecodeError:
                        params = ast.literal_eval(action_input)
                except Exception as exc:
                    logger.warning("hitl_check param JSON parse failed: %s", exc)
                    params = {"raw": action_input, "parse_error": str(exc)}
                hitl_action = params.get("action", action_input) if isinstance(params, dict) else str(action_input)
                res = self.hitl_check(action=hitl_action, parameters=params, reasoning=reasoning)
                obs = str(res)
            else:
                obs = f"Error: Unknown action: {action}"
                logger.error("LLM returned unknown action: %s — stopping agent", action)
                self.state = "failed"
                break

            self._broadcast_observation(current_step_id, obs)

            if action == "complete":
                pass
            else:
                past_steps_list.append(f"Turn {execute_turn_count}: Action: {action}({action_input}) -> Observation: {obs}")

            if self.state in ("blocked", "failed"):
                break

            if execute_turn_count >= max_execute_turns and self.state == "running":
                self.state = "failed"
                self._broadcast_observation("system", "ERROR: Maximum execution turns (20) reached. Agent failed to complete the plan.")
                break

        return past_steps_list

    def _infer_task_type(self, name: str) -> str:
        """Delegate task-type inference to the :class:`PlanManager`."""
        return self.plan._infer_task_type(name)

    def _broadcast_plan_created(self) -> None:
        """Emit a ``plan_created`` event with the current task list."""
        if self.plan.current_plan is not None:
            self.emitter.emit_plan({
                "type": "plan_created",
                "incident_id": self.current_incident.incident_id if self.current_incident else None,
                "tasks": self.plan.current_plan["tasks"],
            })

    def _parse_plan_from_template(self) -> None:
        """Parse the default plan template; sets state to ``failed`` on failure."""
        if not self.plan.parse_from_template():
            self.state = "failed"

    def _broadcast_plan_task_started(self, task_id: int) -> None:
        """Emit a ``plan_task_started`` event for the given task."""
        self.emitter.emit_plan_task({
            "type": "plan_task_started",
            "incident_id": self.current_incident.incident_id if self.current_incident else None,
            "task_id": task_id,
        })

    def update_plan_task(self, task_id: int | str, status: str):
        """Update the status of a plan task via :class:`PlanManager`."""
        self.plan.update_task(task_id, status)

    def _broadcast_plan_task_update(self, task_id: str, status: str):
        """Emit a plan-task status update event."""
        self.plan._broadcast_plan_task_update(task_id, status)

    def _broadcast_plan_task_completed(self, task_id: str):
        """Emit a plan-task completed event."""
        self.plan._broadcast_plan_task_completed(task_id)

    def run_incident(self, incident: DiagnosticIncident) -> dict[str, Any]:
        """Run the full Planning-First diagnostic pipeline for an incident.

        The pipeline is:

        1. Start a new session.
        2. **Plan phase** — Load ``create_plan`` skill and build a roadmap.
        3. **Execute phase** — Follow the plan tasks with DSPy tools.
        4. Return a result dict with session status and step history.

        Parameters
        ----------
        incident : DiagnosticIncident
            The anomaly incident to diagnose.

        Returns
        -------
        dict[str, Any]
            Keys: ``session_id``, ``incident_id``, ``status``,
            ``diagnose_skill_used``, ``steps``.
        """
        self.start_session(incident)

        # Build per-sensor MAE context (Silver keys are raw sensor names)
        ps_scores = incident.per_sensor_scores or {}
        v9_lines = []
        if ps_scores:
            sensors_in_order = ["wire_feed_speed", "wire_temperature", "welding_current"]
            for s in sensors_in_order:
                val = ps_scores.get(s, None)
                if val is not None:
                    marker = "  ← DOMINANT" if s == incident.sensor else ""
                    v9_lines.append(f"    {s}: {round(float(val), 4)} MAE{marker}")

        per_sensor_context = ""
        if v9_lines:
            per_sensor_context = "\nPer-sensor MAE scores:\n" + "\n".join(v9_lines)

        incident_context = (
            f"Robot: {incident.robot_code}, "
            f"Family: {incident.abnormal_family}, "
            f"Subtype: {incident.abnormal_subtype}, "
            f"Sensor: {incident.sensor}, "
            f"Score: {incident.anomaly_score}"
            f"{per_sensor_context}"
        )

        loaded_skills_content = ""
        past_steps_list = []
        diagnose_skill_used: str | None = None

        lm = config.get_lm()
        if lm is None:
            logger.error("No LM configured - cannot run agent")
            self.state = "failed"
            return {
                "session_id": self.session_id,
                "incident_id": incident.incident_id,
                "status": self.state,
                "diagnose_skill_used": None,
                "steps": [],
            }

        cfg = config.get_llm_config()
        logger.info(f"Using LLM: {cfg.provider}/{cfg.model} (Base: {cfg.api_base})")

        self.state = "running"

        loaded_skills_content, past_steps_str, past_steps_list = self._run_plan_phase(
            incident_context, loaded_skills_content, past_steps_list
        )

        if self.state in ["completed", "failed"]:
            return {
                "session_id": self.session_id,
                "incident_id": incident.incident_id,
                "status": self.state,
                "diagnose_skill_used": diagnose_skill_used,
                "steps": [s.step_id for s in self.steps],
            }

        past_steps_list = self._run_execute_phase(incident_context, loaded_skills_content, past_steps_list)

        if self.state == "running":
            self.state = "failed"

        return {
            "session_id": self.session_id,
            "incident_id": incident.incident_id,
            "status": self.state,
            "diagnose_skill_used": diagnose_skill_used,
            "steps": [s.step_id for s in self.steps],
        }

def create_agent() -> DiagnosticAgent:
    """Factory function that returns a new :class:`DiagnosticAgent` instance."""
    return DiagnosticAgent()
