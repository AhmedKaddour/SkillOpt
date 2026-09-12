from __future__ import annotations

import json
import os
import re
from pathlib import Path

from skillopt.model import chat_optimizer, chat_target


def _contains(text: str, term: str) -> bool:
    return term.casefold() in (text or "").casefold()


def _extract_json(text: str) -> dict:
    candidate = (text or "").strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.I)
        candidate = re.sub(r"\s*```$", "", candidate)
    try:
        data = json.loads(candidate)
        return data if isinstance(data, dict) else {}
    except Exception:
        match = re.search(r"\{.*\}", candidate, flags=re.S)
        if not match:
            return {}
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}


def _rubric_text(criteria: list[dict]) -> str:
    if not criteria:
        return "- Overall usefulness, correctness, clarity, and instruction adherence (weight 1)."
    lines: list[str] = []
    for idx, criterion in enumerate(criteria, start=1):
        name = str(criterion.get("name") or f"criterion_{idx}")
        description = str(criterion.get("description") or "")
        weight = float(criterion.get("weight", 1.0))
        lines.append(f"- {name} (weight {weight:g}): {description}")
    return "\n".join(lines)


def _judge(item: dict, prediction: str) -> tuple[float, str, dict]:
    required = item.get("required_terms") or []
    forbidden = item.get("forbidden_terms") or []
    missing_required = [term for term in required if not _contains(prediction, term)]
    present_forbidden = [term for term in forbidden if _contains(prediction, term)]

    system = (
        "You are a strict evaluation judge for an agent-skill benchmark. "
        "Evaluate only against the supplied task, reference context, and rubric. "
        "Do not reward unsupported additions. Return JSON only."
    )
    user = f"""TASK\n{item['task']}\n\nREFERENCE CONTEXT\n{item.get('reference_context', '')}\n\nRUBRIC\n{_rubric_text(item.get('criteria') or [])}\n\nCANDIDATE OUTPUT\n{prediction}\n\nReturn exactly this JSON shape:\n{{\n  \"score\": 0.0,\n  \"reason\": \"brief evidence-based reason\",\n  \"criterion_scores\": {{\"criterion name\": 0.0}}\n}}\n\nscore and each criterion score must be numbers from 0 to 1."""
    reply, _usage = chat_optimizer(
        system=system,
        user=user,
        max_completion_tokens=1200,
        stage="rubrictext_judge",
    )
    judged = _extract_json(reply)
    try:
        score = max(0.0, min(1.0, float(judged.get("score", 0.0))))
    except Exception:
        score = 0.0
    reason = str(judged.get("reason") or "Judge did not return a usable reason.")

    if missing_required:
        score = min(score, 0.49)
        reason += f" Missing required terms: {missing_required}."
    if present_forbidden:
        score = min(score, 0.49)
        reason += f" Forbidden terms present: {present_forbidden}."

    judged["missing_required"] = missing_required
    judged["present_forbidden"] = present_forbidden
    judged["score_after_deterministic_checks"] = score
    return score, reason, judged


def _rollout_one(
    item: dict,
    skill_content: str,
    *,
    prediction_dir: Path,
    max_completion_tokens: int,
) -> dict:
    system = skill_content
    user = item["task"]
    reference = item.get("reference_context", "").strip()
    if reference:
        user += f"\n\nReference context:\n{reference}"

    prediction, _usage = chat_target(
        system=system,
        user=user,
        max_completion_tokens=max_completion_tokens,
        stage="rubrictext_target",
    )
    soft, reason, judge_details = _judge(item, prediction)
    threshold = float(item.get("minimum_score", 0.80))
    hard = int(soft >= threshold)

    task_dir = prediction_dir / str(item["id"])
    task_dir.mkdir(parents=True, exist_ok=True)
    conversation = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": prediction},
    ]
    (task_dir / "conversation.json").write_text(
        json.dumps(conversation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (task_dir / "judge.json").write_text(
        json.dumps(judge_details, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "id": str(item["id"]),
        "hard": hard,
        "soft": soft,
        "predicted_answer": prediction,
        "task_description": item["task"],
        "question": item["task"],
        "reference_text": item.get("reference_context", ""),
        "task_type": item.get("task_type", "rubrictext"),
        "fail_reason": "" if hard else reason,
        "judge_reason": reason,
        "criterion_scores": judge_details.get("criterion_scores", {}),
        "target_system_prompt": system,
        "target_user_prompt": user,
        "n_turns": 1,
    }


def run_batch(
    *,
    items: list[dict],
    skill_content: str,
    out_root: str,
    workers: int = 1,
    max_completion_tokens: int = 4096,
) -> list[dict]:
    del workers  # Sequential by design initially: deterministic and easier to debug.
    os.makedirs(out_root, exist_ok=True)
    prediction_dir = Path(out_root, "predictions")
    results = [
        _rollout_one(
            item,
            skill_content,
            prediction_dir=prediction_dir,
            max_completion_tokens=max_completion_tokens,
        )
        for item in items
    ]
    Path(out_root, "rollouts.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return results
