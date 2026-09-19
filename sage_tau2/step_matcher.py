"""Step-level skill activation gate for τ² (ALFWorld ``SkillPreconditionMatcher`` analogue).

ALFWorld retrieves skills per-step: the candidate pool is fixed for the episode,
but which skills are *active* (injected into the prompt) is re-evaluated every
step based on the action history + observation. This module brings the same
per-step retrieval to τ².

τ² skill protocols are tool-call sequences, not ALFWorld action verbs, so the
step gate is tool-oriented:

* **Write-goal skills** (primary write tool in ``WRITE_TOOL_NAMES``): active
  until the agent has *successfully* called the skill's primary write tool.
  Once the write lands without error, the skill's goal is achieved and it is
  deactivated — the prompt slot frees up for the next pending skill.
* **Lookup-only skills** (no write tool in the protocol): stay active for the
  whole episode (they are diagnostic, not goal-terminating).
* **Multi-write skills** (e.g. ``send_payment_request`` + ``resume_line``):
  stay active until *all* their write tools have been successfully called.

The matcher reads the live conversation history (assistant ``tool_calls`` +
``ToolMessage`` error flags) so it reacts to what the agent actually did, not
to static task-id hints.
"""

from __future__ import annotations

from typing import Any

from sage_tau2.credit import injectable_skills
from sage_tau2.schemas import Tau2Skill
from sage_tau2.task_context import (
    WRITE_TOOL_NAMES,
    _normalized_scope,
    skill_agent_write_names,
)


class Tau2StepMatcher:
    """Per-step skill activation gate for τ² Executor skill injection."""

    def active_skills(
        self,
        candidate_pool: list[Tau2Skill],
        *,
        messages: list[Any],
        task_id: str = "",
        max_skills: int = 2,
        allow_provisional: bool = True,
        dedupe_writes: bool = True,
    ) -> list[Tau2Skill]:
        """Return skills from ``candidate_pool`` that are still *pending*.

        Parameters
        ----------
        candidate_pool:
            Episode-level filtered skills (domain + scope matched, not
            truncated by ``max_skills``). Fixed for the episode.
        messages:
            Live conversation history (pydantic models or dicts). Used to
            extract successful write-tool calls.
        task_id:
            Current episode task id (for coverage ranking).
        max_skills:
            Hard cap on returned skills (applied after step-level gating).
        """
        if not candidate_pool:
            return []

        successful_writes = self._successful_write_calls(messages)
        if not successful_writes:
            # Nothing achieved yet — all candidates are pending.
            pending = list(candidate_pool)
        else:
            pending = [
                skill
                for skill in candidate_pool
                if self._is_pending(skill, successful_writes)
            ]

        if not pending:
            return []

        # Re-rank the pending pool: coverage + credit + dedupe, then cap.
        return injectable_skills(
            pending,
            max_skills=max_skills,
            allow_provisional=allow_provisional,
            task_id=task_id,
            dedupe_writes=dedupe_writes,
        )

    # ------------------------------------------------------------------
    # Step-level gate
    # ------------------------------------------------------------------

    @staticmethod
    def _is_pending(
        skill: Tau2Skill,
        successful_writes: set[str],
    ) -> bool:
        """True when the skill's write goal has not been achieved yet."""
        writes = set(skill_agent_write_names(skill))
        if not writes:
            # Lookup-only / diagnostic skill: always pending (no terminal write).
            return True
        # Multi-write protocol: pending until *all* writes are done.
        achieved = writes & successful_writes
        return achieved != writes

    # ------------------------------------------------------------------
    # Conversation history parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _successful_write_calls(messages: list[Any]) -> set[str]:
        """Extract normalized names of *successful* write-tool calls.

        Walks the conversation history, pairing each assistant ``tool_call``
        with its ``ToolMessage`` result. A write is "successful" when the
        tool result has ``error=False`` (or no error flag at all).
        """
        write_norm = {_normalized_scope(n) for n in WRITE_TOOL_NAMES}
        # Map tool_call_id → name from assistant messages.
        call_id_to_name: dict[str, str] = {}
        for msg in messages:
            d = Tau2StepMatcher._msg_dict(msg)
            if d.get("role") != "assistant":
                continue
            for tc in d.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                name = str(
                    tc.get("name")
                    or (tc.get("function") or {}).get("name")
                    or ""
                ).strip()
                cid = str(tc.get("id") or tc.get("tool_call_id") or "").strip()
                if name and cid:
                    call_id_to_name[cid] = name

        successful: set[str] = set()
        for msg in messages:
            d = Tau2StepMatcher._msg_dict(msg)
            role = str(d.get("role") or "").lower()
            if role not in {"tool", "toolmessage"}:
                continue
            cid = str(
                d.get("id")
                or d.get("tool_call_id")
                or ""
            ).strip()
            name = str(d.get("name") or "").strip()
            if not name and cid in call_id_to_name:
                name = call_id_to_name[cid]
            if not name:
                continue
            norm = _normalized_scope(name)
            if norm not in write_norm:
                continue
            # error flag: False or absent = success.
            errored = d.get("error")
            if errored is True:
                continue
            successful.add(norm)
        return successful

    @staticmethod
    def _msg_dict(msg: Any) -> dict[str, Any]:
        if isinstance(msg, dict):
            return msg
        if hasattr(msg, "model_dump"):
            return msg.model_dump()
        if hasattr(msg, "dict"):
            return msg.dict()
        return {"role": getattr(msg, "role", None), "content": getattr(msg, "content", None)}
