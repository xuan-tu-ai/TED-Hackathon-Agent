from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, ClassVar

from dotenv import load_dotenv

load_dotenv()

import dspy
import json
import re
from .hitl import HitlResult, hitl_check as perform_hitl_check
from .llm_config import get_lm
from .session_trace import SessionTraceWriter
from .skills import LoadSkillTool, SKILLS_BASE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("diagnostic_agent")

AGENTS_MD = Path(__file__).resolve().parents[3] / "agent_knowledge" / "AGENTS.md"
HITL_RULES_MD = Path(__file__).resolve().parents[3] / "agent_knowledge" / "rules" / "hitl-rules.md"

@dataclass
class DiagnosticIncident:
    incident_id: str
    robot_code: str
    abnormal_family: str
    abnormal_subtype: str
    sensor: str
    anomaly_score: float
    severity_hint: str
    context: dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentStep:
    step_id: str
    action: str
    skill_loaded: str | None
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)

class AgentStepSignature(dspy.Signature):
    """You are a Painting Robot Diagnostic Agent. Your job is to analyze incidents and determine the root cause by invoking necessary skills and tools.
    Follow the rules in SYSTEM_PROMPT.
    If you haven't loaded 'routing' skill yet, your very first action MUST be load_skill with parameter 'routing'.
    When you determine the root cause, load 'create_plan' to create a diagnostic plan with atomic tasks.
    Follow the plan step-by-step. Only after ALL plan items are executed, output action='complete'.
    Do NOT guess. If you need to search past incidents, use query_pm_cards.
    If you need human approval, use hitl_check.
    When all plan items are done and HITL checkpoints cleared (if any), output action='complete' and action_input='summary'.
    """
    system_prompt = dspy.InputField(desc="The core AGENTS.md rules.")
    incident_context = dspy.InputField(desc="The current incident context.")
    loaded_skills = dspy.InputField(desc="The contents of the skills you have loaded so far.")
    past_steps = dspy.InputField(desc="Previous actions and their observations. Pay close attention to what you already did.")
    
    reasoning = dspy.OutputField(desc="Your thought process for what to do next based on the incident context and past steps.")
    action = dspy.OutputField(desc="The tool to call. Must be exactly one of: load_skill, query_pm_cards, verify_surface_defect, query_db, hitl_check, complete.")
    action_input = dspy.OutputField(desc="The argument for the tool. A string or JSON depending on the tool.")

class DiagnosticAgent:
    SYSTEM_PROMPT: ClassVar[str] = AGENTS_MD.read_text(encoding="utf-8")
    TIER2_RULES: ClassVar[str] = HITL_RULES_MD.read_text(encoding="utf-8")
    AVAILABLE_TOOLS: ClassVar[list[str]] = [
        "load_skill",
        "query_pm_cards",
        "verify_surface_defect",
        "query_db",
        "hitl_check",
    ]

    def __init__(self):
        self.load_skill_tool = LoadSkillTool()
        self.session_id: str | None = None
        self.current_incident: DiagnosticIncident | None = None
        self.steps: list[AgentStep] = []
        self.context: dict[str, Any] = {}
        self.state: str = "idle"
        self.hitl_pending: dict[str, Any] = {}
        self._trace_writer: SessionTraceWriter | None = None
        self.on_hitl_request: Callable[[dict[str, Any]], None] | None = None
        self.on_step: Callable[[dict[str, Any]], None] | None = None
        self.on_observation: Callable[[dict[str, Any]], None] | None = None
        self.on_plan: Callable[[dict[str, Any]], None] | None = None
        self.on_plan_task: Callable[[dict[str, Any]], None] | None = None
        self.current_skill: str | None = None
        self.current_plan: dict[str, Any] | None = None
        self.current_task_index: int = 0

    def start_session(self, incident: DiagnosticIncident) -> str:
        self.session_id = f"sess-{incident.incident_id}"
        self.current_incident = incident
        self.steps = []
        self.context = {}
        self.state = "idle"
        self.hitl_pending = {}
        self.current_skill = None
        self.current_plan = None
        self.current_task_index = 0
        self._trace_writer = SessionTraceWriter(incident.incident_id)
        logger.info(f"Session {self.session_id} started for incident {incident.incident_id}")
        return self.session_id

    def step(self, action: str, reasoning: str, skill_loaded: str | None = None) -> AgentStep:
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
        
        if self.on_step is not None:
            parts = action.split("(", 1)
            tool_name = parts[0]
            tool_input = parts[1][:-1] if len(parts) > 1 and action.endswith(")") else ""
            self.on_step({
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
        if self.on_observation is not None:
            self.on_observation({
                "type": "agent_step_update",
                "session_id": self.session_id,
                "incident_id": self.current_incident.incident_id if self.current_incident else None,
                "step_id": step_id,
                "observation": observation,
            })

    def load_skill(self, skill_name: str) -> str:
        content = self.load_skill_tool(skill_name)
        return content

    def hitl_check(self, action: str, parameters: dict[str, Any], reasoning: str) -> dict[str, Any]:
        if self.current_incident is None:
            return {"error": "No active incident"}

        result = perform_hitl_check(
            action=action,
            incident_id=self.current_incident.incident_id,
            robot_code=self.current_incident.robot_code,
            parameters=parameters,
            reasoning=reasoning,
        )

        if result.result == HitlResult.PERMANENTLY_BLOCKED:
            self.state = "blocked"
            return {
                "blocked": True,
                "permanently_blocked": True,
                "action": action,
                "reason": result.reason,
            }

        if result.result == HitlResult.BLOCKED:
            self.hitl_pending[action] = result.queue_id

            from ..simulation.state import get_simulation_state
            state = get_simulation_state()

            hitl_event = threading.Event()
            state.set_hitl_event(result.queue_id, hitl_event)

            self.state = "blocked"

            if self.on_hitl_request is not None:
                self.on_hitl_request({
                    "type": "hitl_request",
                    "queue_id": result.queue_id,
                    "incident_id": self.current_incident.incident_id,
                    "robot_code": self.current_incident.robot_code,
                    "action": action,
                    "parameters": parameters,
                    "severity": result.severity,
                    "reasoning": reasoning,
                    "context": {
                        "step_count": len(self.steps),
                        "skill_in_use": self.current_skill,
                        "incident_summary": (
                            f"Robot {self.current_incident.robot_code}, "
                            f"{self.current_incident.abnormal_family}/{self.current_incident.abnormal_subtype}, "
                            f"score={self.current_incident.anomaly_score}"
                        ),
                    },
                })

            hitl_event.wait(timeout=300)

            state.remove_hitl_event(result.queue_id)

            self.state = "running"

            return {
                "blocked": True,
                "permanently_blocked": False,
                "queue_id": result.queue_id,
                "action": action,
                "reason": result.reason,
                "severity": result.severity,
                "resolved": True,
            }

        return {"allowed": True, "action": action}

    def explore(self): pass
    def plan(self): pass
    def act(self): pass

    def _run_plan_phase(self, incident_context: str, loaded_skills_content: str, past_steps_list: list[str]) -> tuple[str, str, list[str]]:
        lm = get_lm()
        predictor = dspy.ChainOfThought(AgentStepSignature)

        plan_turn_count = 0
        max_plan_turns = 3
        has_routed = False
        has_diagnosed = False
        has_plan = False

        while plan_turn_count < max_plan_turns and self.state not in ["blocked", "completed", "failed"]:
            plan_turn_count += 1
            past_steps_str = "\n".join(past_steps_list) if past_steps_list else "None yet."

            try:
                with dspy.context(lm=lm):
                    result = predictor(
                        system_prompt=self.SYSTEM_PROMPT,
                        incident_context=incident_context,
                        loaded_skills=loaded_skills_content,
                        past_steps=past_steps_str
                    )
                reasoning = result.reasoning
                action = result.action.strip()
                action_input = result.action_input.strip()
            except Exception as e:
                logger.error(f"DSPy prediction error in plan phase: {e}")
                self.step("error", f"LLM error: {e}")
                self.state = "failed"
                break

            skill_loaded = action_input if action == "load_skill" else None
            self.step(action=f"{action}({action_input})", reasoning=reasoning, skill_loaded=skill_loaded)

            if action == "complete":
                if not has_plan:
                    self._apply_default_plan()
                self.state = "completed"
                break
            elif action == "load_skill":
                try:
                    content = self.load_skill(action_input)
                    loaded_skills_content += f"\n--- SKILL: {action_input} ---\n{content}\n"
                    obs = f"Loaded skill {action_input} successfully."
                    loaded_skill_name = action_input.strip()
                    if loaded_skill_name == "routing":
                        has_routed = True
                    elif loaded_skill_name not in ["routing", "create_plan"]:
                        has_diagnosed = True
                    elif loaded_skill_name == "create_plan":
                        has_plan = True
                        self._broadcast_plan_created()
                except Exception as e:
                    obs = f"Failed to load skill: {e}"
                self._broadcast_observation(self.steps[-1].step_id, obs)
                past_steps_list.append(f"Turn {plan_turn_count}: Action: {action}({action_input}) -> Observation: {obs}")
            else:
                obs = f"Unexpected action in plan phase: {action}"
                self._broadcast_observation(self.steps[-1].step_id, obs)
                past_steps_list.append(f"Turn {plan_turn_count}: Action: {action}({action_input}) -> Observation: {obs}")

            if has_routed and has_diagnosed and has_plan:
                break

        if plan_turn_count >= max_plan_turns and not has_plan:
            logger.warning("Plan phase reached max turns without plan, applying default")
            self._apply_default_plan()

        return loaded_skills_content, past_steps_str, past_steps_list

    def _run_execute_phase(self, loaded_skills_content: str, past_steps_list: list[str]) -> list[str]:
        lm = get_lm()
        predictor = dspy.ChainOfThought(AgentStepSignature)
        execute_turn_count = 0
        max_execute_turns = 10

        while execute_turn_count < max_execute_turns and self.state not in ["blocked", "completed", "failed"]:
            execute_turn_count += 1

            if self.current_plan and self.current_task_index < len(self.current_plan["tasks"]):
                current_task = self.current_plan["tasks"][self.current_task_index]
                task_name = current_task.get("name", f"Task {self.current_task_index + 1}")
                self._broadcast_plan_task_started(current_task.get("id", self.current_task_index + 1))

            past_steps_str = "\n".join(past_steps_list) if past_steps_list else "None yet."

            try:
                with dspy.context(lm=lm):
                    result = predictor(
                        system_prompt=self.SYSTEM_PROMPT,
                        incident_context="",
                        loaded_skills=loaded_skills_content,
                        past_steps=past_steps_str
                    )
                reasoning = result.reasoning
                action = result.action.strip()
                action_input = result.action_input.strip()
            except Exception as e:
                logger.error(f"DSPy prediction error in execute phase: {e}")
                self.step("error", f"LLM error: {e}")
                self.state = "failed"
                break

            skill_loaded = action_input if action == "load_skill" else None
            self.step(action=f"{action}({action_input})", reasoning=reasoning, skill_loaded=skill_loaded)

            if action == "complete":
                self.state = "completed"
                break
            elif action == "load_skill":
                try:
                    content = self.load_skill(action_input)
                    loaded_skills_content += f"\n--- SKILL: {action_input} ---\n{content}\n"
                    obs = f"Loaded skill {action_input} successfully."
                except Exception as e:
                    obs = f"Failed to load skill: {e}"
            elif action == "query_pm_cards":
                from .rag_tools import QueryPMCardsTool
                tool = QueryPMCardsTool()
                obs = str(tool(action_input))
            elif action == "query_db":
                obs = f"Queried DB with {action_input}. Returned 1 relevant row indicating recent anomaly."
            elif action == "verify_surface_defect":
                obs = f"Checked {action_input}. Found surface defects: orange_peel x14, thin_coat x5."
            elif action == "hitl_check":
                try:
                    params = json.loads(action_input)
                except Exception:
                    params = {"raw": action_input}
                res = self.hitl_check(action="approve_plan", parameters=params, reasoning=reasoning)
                obs = str(res)
            else:
                obs = f"Unknown action: {action}"

            self._broadcast_observation(self.steps[-1].step_id, obs)
            past_steps_list.append(f"Turn {execute_turn_count}: Action: {action}({action_input}) -> Observation: {obs}")

            if self.current_plan and self.current_task_index < len(self.current_plan["tasks"]):
                task_id = self.current_plan["tasks"][self.current_task_index].get("id", self.current_task_index + 1)
                self._broadcast_plan_task_completed(task_id)
                self.current_task_index += 1

        return past_steps_list

    def _apply_default_plan(self) -> None:
        default_plan = {
            "tasks": [
                {"id": 1, "name": "Load routing skill", "type": "skill", "status": "completed"},
                {"id": 2, "name": "Run diagnosis", "type": "diagnose", "status": "completed"},
                {"id": 3, "name": "HITL approval", "type": "hitl", "status": "pending"},
                {"id": 4, "name": "Dispatch action", "type": "action", "status": "pending"},
            ]
        }
        self.current_plan = default_plan
        self._broadcast_plan_created()

    def _infer_task_type(self, name: str) -> str:
        name_lower = name.lower()
        if "load" in name_lower or "skill" in name_lower:
            return "skill"
        elif "diagnos" in name_lower or "check" in name_lower or "query" in name_lower:
            return "diagnose"
        elif "hitl" in name_lower or "approval" in name_lower:
            return "hitl"
        elif "dispatch" in name_lower or "send" in name_lower or "formulate" in name_lower or "action" in name_lower:
            return "action"
        elif "evaluate" in name_lower or "determine" in name_lower:
            return "assess"
        else:
            return "diagnose"

    def _broadcast_plan_created(self) -> None:
        if self.current_plan is None:
            self._parse_plan_from_template()
        if self.on_plan is not None and self.current_plan is not None:
            self.on_plan({
                "type": "plan_created",
                "incident_id": self.current_incident.incident_id if self.current_incident else None,
                "tasks": self.current_plan["tasks"],
            })

    def _parse_plan_from_template(self) -> None:
        template_path = SKILLS_BASE / "create_plan" / "_template.md"
        if not template_path.exists():
            self._apply_default_plan()
            return

        content = template_path.read_text(encoding="utf-8")
        tasks = []
        task_id = 1
        for line in content.splitlines():
            if "- [ ]" in line or "- [x]" in line:
                match = re.search(r"\*\*(Item \d+:)\s*(.*?)\.", line)
                if match:
                    name = match.group(2).strip()
                    tasks.append({
                        "id": task_id,
                        "name": name,
                        "type": self._infer_task_type(name),
                        "status": "pending",
                    })
                    task_id += 1

        if not tasks:
            self._apply_default_plan()
            return

        self.current_plan = {"tasks": tasks}

    def _broadcast_plan_task_started(self, task_id: int) -> None:
        if self.on_plan_task is not None:
            self.on_plan_task({
                "type": "plan_task_started",
                "incident_id": self.current_incident.incident_id if self.current_incident else None,
                "task_id": task_id,
            })

    def _broadcast_plan_task_completed(self, task_id: int) -> None:
        if self.on_plan_task is not None:
            self.on_plan_task({
                "type": "plan_task_completed",
                "incident_id": self.current_incident.incident_id if self.current_incident else None,
                "task_id": task_id,
            })

    def run_incident(self, incident: DiagnosticIncident) -> dict[str, Any]:
        self.start_session(incident)

        incident_context = f"Family: {incident.abnormal_family}, Subtype: {incident.abnormal_subtype}, Sensor: {incident.sensor}, Score: {incident.anomaly_score}"

        loaded_skills_content = ""
        past_steps_list = []
        diagnose_skill_used: str | None = None

        lm = get_lm()
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

        past_steps_list = self._run_execute_phase(loaded_skills_content, past_steps_list)

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
    return DiagnosticAgent()
