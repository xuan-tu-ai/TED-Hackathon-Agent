from __future__ import annotations

import logging
import os
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
from .llm_config import get_lm, get_llm_config
from .rag_tools import QueryPMCardsTool
from .session_trace import SessionTraceWriter
from .skills import LoadSkillTool, SKILLS_BASE
from ..seed import create_session_factory

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
    """You are a Painting Robot Diagnostic Agent. Your job is to analyze incidents and determine the root cause.
    You MUST follow the 'Planning First' principle:
    1. Your very first action MUST be load_skill('create_plan').
    2. Once the plan is created, follow the plan tasks sequentially (Routing -> Diagnosis -> Action).
    3. Load subsequent skills (like 'routing' or 'diagnose_surge') as specified in your plan.
    4. Only after ALL plan items are executed, output action='complete'.
    """
    system_prompt = dspy.InputField(desc="The core AGENTS.md rules.")
    incident_context = dspy.InputField(desc="The current incident context.")
    loaded_skills = dspy.InputField(desc="The contents of the skills you have loaded so far.")
    available_skills = dspy.InputField(desc="List of all available skills and their metadata. Choose from these ONLY.")
    current_plan = dspy.InputField(desc="The current diagnostic plan and task statuses. Follow this roadmap.")
    past_steps = dspy.InputField(desc="Previous actions and their observations. Pay close attention to what you already did.")

    reasoning = dspy.OutputField(desc="Your thought process for what to do next based on the incident context and past steps.")
    action = dspy.OutputField(desc="The tool to call. Must be exactly one of: load_skill, query_pm_cards, verify_surface_defect, query_db, hitl_check, update_task_status, create_plan, complete.")
    action_input = dspy.OutputField(desc="The argument for the tool. \n- load_skill: 'skill_name'\n- update_task_status: '{\"task_id\": 1, \"status\": \"completed\"}'\n- create_plan: Use only to initialize or radically change the roadmap.\n- complete: 'summary_text'. Use ONLY when ALL plan tasks are 'completed'.")

class DiagnosticAgent:
    SYSTEM_PROMPT: ClassVar[str] = AGENTS_MD.read_text(encoding="utf-8")
    TIER2_RULES: ClassVar[str] = HITL_RULES_MD.read_text(encoding="utf-8")
    AVAILABLE_TOOLS: ClassVar[list[str]] = [
        "load_skill",
        "query_pm_cards",
        "verify_surface_defect",
        "query_db",
        "hitl_check",
        "update_task_status",
        "create_plan",
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
        self.loaded_skills_set: set[str] = set()
        self.skill_catalog: str = self._load_skill_catalog()

    def _load_skill_catalog(self) -> str:
        catalog = []
        skills_dir = SKILLS_BASE
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
        self.session_id = f"sess-{incident.incident_id}"
        self.current_incident = incident
        self.steps = []
        self.context = {}
        self.state = "idle"
        self.hitl_pending = {}
        self.current_skill = None
        self.current_plan = None
        self.current_task_index = 0
        self.loaded_skills_set = set()
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

            event_was_set = hitl_event.wait(timeout=300)

            state.remove_hitl_event(result.queue_id)

            if not event_was_set:
                self.state = "failed"
                return {
                    "blocked": True,
                    "permanently_blocked": False,
                    "queue_id": result.queue_id,
                    "action": action,
                    "reason": f"Timeout waiting for HITL approval on '{action}' (300s elapsed)",
                    "severity": result.severity,
                    "resolved": False,
                    "timeout": True,
                }

            from ..db.models import HitlQueue
            _db_url = os.getenv("DATABASE_URL") or "sqlite+pysqlite:///../data/demo.db"
            _session = create_session_factory(_db_url)()
            try:
                entry = _session.get(HitlQueue, result.queue_id)
                entry_status = entry.status if entry else None
            finally:
                _session.close()

            if entry is None or entry_status in ("REJECTED", "EXPIRED"):
                self.state = "failed"
                return {
                    "blocked": True,
                    "permanently_blocked": False,
                    "queue_id": result.queue_id,
                    "action": action,
                    "reason": f"HITL action '{action}' was {entry_status.lower() if entry_status else 'entry not found'}",
                    "severity": result.severity,
                    "resolved": False,
                }

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
        max_plan_turns = 5
        has_plan = False
        past_steps_str = "None yet."

        while plan_turn_count < max_plan_turns and self.state not in ["blocked", "completed", "failed"]:
            plan_turn_count += 1
            past_steps_str = "\n".join(past_steps_list) if past_steps_list else "None yet."

            phase_instruction = (
                "CURRENT PHASE: PLANNING. "
                "Your goal is to create a roadmap for this incident. Your very first action MUST be 'load_skill(\"create_plan\")'. "
                "Do NOT perform routing or diagnosis until the plan is created."
            )
            full_prompt = f"{self.SYSTEM_PROMPT}\n\n{phase_instruction}"

            logger.info(f"--- LLM Plan Turn {plan_turn_count} (Incident: {self.current_incident.incident_id if self.current_incident else 'unknown'}) ---")
            try:
                with dspy.context(lm=lm, cache=False):
                    result = predictor(
                        system_prompt=full_prompt,
                        incident_context=incident_context,
                        available_skills=self.skill_catalog,
                        loaded_skills=loaded_skills_content,
                        current_plan=json.dumps(self.current_plan, indent=2) if self.current_plan else "No plan created yet.",
                        past_steps=past_steps_str
                    )

                reasoning = getattr(result, 'reasoning', "No reasoning provided.")
                reasoning = re.sub(r"<think>.*?</think>", "", reasoning, flags=re.DOTALL).strip()

                raw_action = getattr(result, 'action', "").strip()
                raw_input = getattr(result, 'action_input', "").strip()

                if "{" in raw_input and "}" in raw_input:
                    try:
                        json_match = re.search(r"(\{.*\})", raw_input)
                        json_str = json_match.group(1) if json_match else None
                        if json_str:
                            data = json.loads(json_str)
                            if isinstance(data, dict):
                                raw_input = data.get("skill_name") or data.get("query") or str(data)
                    except: pass

                if not raw_action and "(" in raw_input:
                    match = re.match(r"(\w+)\((.*)\)", raw_input)
                    if match:
                        raw_action = match.group(1)
                        raw_input = match.group(2)

                action = raw_action
                action_input = raw_input.strip("'\" ")

                logger.info(f"Agent Thought: {reasoning[:80]}...")
                logger.info(f"Agent Action: {action}({action_input})")

                if not action:
                    logger.warning(f"Empty action from LLM. Raw result: {result}")
                    continue

            except Exception as e:
                logger.error(f"LLM Call Failed: {e}")
                self.step("error", f"LLM error: {e}")
                self.state = "failed"
                break

            skill_loaded = action_input if action == "load_skill" else None
            step_info = self.step(action=f"{action}({action_input})", reasoning=reasoning, skill_loaded=skill_loaded)
            current_step_id = step_info.step_id

            if action == "complete":
                obs = "Diagnosis concluded. Ready for dispatch."
                self._broadcast_observation(current_step_id, obs)
                if not has_plan:
                    self._apply_default_plan()
                self.state = "completed"
                break
            elif action == "load_skill":
                skill_name = action_input.split("/")[-1].replace(".md", "").strip()
                try:
                    content = self.load_skill(skill_name)
                    loaded_skills_content += f"\n--- SKILL: {skill_name} ---\n{content}\n"
                    obs = f"Loaded skill {skill_name} successfully."
                    self.loaded_skills_set.add(skill_name)
                    if skill_name == "create_plan":
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
            logger.warning("Plan phase reached max turns without plan, applying default")
            self._apply_default_plan()

        return loaded_skills_content, past_steps_str, past_steps_list

    def _run_execute_phase(self, incident_context: str, loaded_skills_content: str, past_steps_list: list[str]) -> list[str]:
        lm = get_lm()
        predictor = dspy.ChainOfThought(AgentStepSignature)
        execute_turn_count = 0
        max_execute_turns = 20

        while execute_turn_count < max_execute_turns and self.state not in ["blocked", "completed", "failed"]:
            execute_turn_count += 1

            if self.current_plan and self.current_task_index < len(self.current_plan["tasks"]):
                current_task = self.current_plan["tasks"][self.current_task_index]
                task_name = current_task.get("name", f"Task {self.current_task_index + 1}")
                logger.info(f"Executing Task: {task_name}")

            past_steps_str = "\n".join(past_steps_list) if past_steps_list else "None yet."

            current_task_info = ""
            if self.current_plan and self.current_task_index < len(self.current_plan["tasks"]):
                t = self.current_plan["tasks"][self.current_task_index]
                current_task_info = f"CURRENT TASK: {t.get('name')} (Goal: Perform this task using available tools)."

            phase_instruction = (
                "CURRENT PHASE: EXECUTION. "
                "Follow the plan items sequentially. Use tools (query_db, query_pm_cards, etc.) to gather evidence. "
                f"{current_task_info}"
            )
            full_prompt = f"{self.SYSTEM_PROMPT}\n\n{phase_instruction}"

            try:
                with dspy.context(lm=lm, cache=False):
                    result = predictor(
                        system_prompt=full_prompt,
                        incident_context=incident_context,
                        available_skills=self.skill_catalog,
                        loaded_skills=loaded_skills_content,
                        current_plan=json.dumps(self.current_plan, indent=2) if self.current_plan else "Plan lost.",
                        past_steps=past_steps_str
                    )

                reasoning = getattr(result, 'reasoning', "No reasoning provided.")
                raw_action = getattr(result, 'action', "").strip()
                raw_input = getattr(result, 'action_input', "").strip()

                if not raw_action and "(" in raw_input:
                    match = re.match(r"(\w+)\((.*)\)", raw_input)
                    if match:
                        raw_action = match.group(1)
                        raw_input = match.group(2)

                action = raw_action
                action_input = raw_input.strip("'\"")

                logger.info(f"Step Thought: {reasoning[:120]}...")
                logger.info(f"Step Action: {action}({action_input})")

                if not action:
                    logger.warning(f"Empty action in execute phase. Raw: {result}")
                    continue

            except Exception as e:
                logger.error(f"LLM Call Failed in Execute Phase: {e}")
                self.step("error", f"LLM error: {e}")
                self.state = "failed"
                break

            reasoning = re.sub(r"<think>.*?</think>", "", reasoning, flags=re.DOTALL).strip()

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
                    data = json.loads(action_input)
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
            elif action == "query_db":
                if self.current_incident:
                    obs = (
                        f"DB Result for {action_input}: Found 1 row. "
                        f"Robot {self.current_incident.robot_code} at {datetime.now().isoformat()} "
                        f"showed {self.current_incident.sensor} peak {self.current_incident.anomaly_score * 1.2:.2f} "
                        f"exceeding threshold 0.50."
                    )
                else:
                    obs = f"DB Result for {action_input}: No active incident."
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

            self._broadcast_observation(current_step_id, obs)

            if action == "complete":
                pass
            else:
                past_steps_list.append(f"Turn {execute_turn_count}: Action: {action}({action_input}) -> Observation: {obs}")

            if self.state in ("blocked", "failed"):
                break

            if self.current_plan and self.current_task_index < len(self.current_plan["tasks"]):
                task_id = self.current_plan["tasks"][self.current_task_index].get("id", self.current_task_index + 1)
                self._broadcast_plan_task_completed(task_id)
                self.current_task_index += 1

            if execute_turn_count >= max_execute_turns and self.state == "running":
                self.state = "failed"
                self._broadcast_observation("system", "ERROR: Maximum execution turns (20) reached. Agent failed to complete the plan.")
                break

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

    def update_plan_task(self, task_id: int | str, status: str):
        if not self.current_plan:
            return

        str_id = str(task_id)
        for idx, task in enumerate(self.current_plan.get("tasks", [])):
            if str(task.get("id")) == str_id:
                task["status"] = status
                self._broadcast_plan_task_update(str_id, status)

                if status == "completed" and idx == self.current_task_index:
                    self.current_task_index += 1
                break

    def _broadcast_plan_task_update(self, task_id: str, status: str):
        if self.on_step and self.current_incident:
            self.on_step({
                "type": "plan_task_update",
                "incident_id": self.current_incident.incident_id,
                "task_id": task_id,
                "status": status
            })

    def _broadcast_plan_task_completed(self, task_id: str):
        self._broadcast_plan_task_update(task_id, "completed")

    def run_incident(self, incident: DiagnosticIncident) -> dict[str, Any]:
        self.start_session(incident)

        incident_context = (
            f"Robot: {incident.robot_code}, "
            f"Family: {incident.abnormal_family}, "
            f"Subtype: {incident.abnormal_subtype}, "
            f"Sensor: {incident.sensor}, "
            f"Score: {incident.anomaly_score}"
        )

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

        cfg = get_llm_config()
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
    return DiagnosticAgent()
