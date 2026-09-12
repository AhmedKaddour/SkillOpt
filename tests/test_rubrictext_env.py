from __future__ import annotations

import json

from skillopt.envs.rubrictext import rollout
from skillopt.envs.rubrictext.dataloader import RubricTextDataLoader


def test_rubrictext_dataloader_normalizes_items(tmp_path):
    split_root = tmp_path / "split"
    for split in ("train", "val", "test"):
        split_dir = split_root / split
        split_dir.mkdir(parents=True)
        (split_dir / "items.json").write_text(
            json.dumps([
                {
                    "id": f"{split}-1",
                    "task": "Produce an answer",
                    "reference_context": "Only use this context",
                    "criteria": [{"name": "grounded", "description": "Stay grounded"}],
                }
            ]),
            encoding="utf-8",
        )

    loader = RubricTextDataLoader(split_dir=str(split_root), split_mode="split_dir")
    loader.setup({})

    assert loader.get_train_size() == 1
    assert loader._splits["val"][0]["minimum_score"] == 0.80
    assert loader._splits["test"][0]["task_type"] == "rubrictext"


def test_deterministic_guardrails_cap_judge_score(monkeypatch, tmp_path):
    monkeypatch.setattr(
        rollout,
        "chat_target",
        lambda **kwargs: ("The answer includes a forbidden phrase.", {}),
    )
    monkeypatch.setattr(
        rollout,
        "chat_optimizer",
        lambda **kwargs: (
            json.dumps({
                "score": 0.97,
                "reason": "Strong otherwise",
                "criterion_scores": {"quality": 0.97},
            }),
            {},
        ),
    )

    item = {
        "id": "guardrail-1",
        "task": "Write the answer",
        "reference_context": "Reference",
        "criteria": [{"name": "quality", "description": "Be useful"}],
        "required_terms": ["required phrase"],
        "forbidden_terms": ["forbidden phrase"],
        "task_type": "rubrictext",
        "minimum_score": 0.80,
    }

    results = rollout.run_batch(
        items=[item],
        skill_content="# Skill",
        out_root=str(tmp_path),
        max_completion_tokens=100,
    )

    result = results[0]
    assert result["hard"] == 0
    assert result["soft"] == 0.49
    assert "Missing required terms" in result["fail_reason"]
    assert "Forbidden terms present" in result["fail_reason"]
    assert (tmp_path / "predictions" / "guardrail-1" / "conversation.json").exists()
    assert (tmp_path / "predictions" / "guardrail-1" / "judge.json").exists()
