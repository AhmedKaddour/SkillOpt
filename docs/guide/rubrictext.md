# RubricText environment

`rubrictext` is a generic SkillOpt environment for improving Markdown skills whose target output is text and whose quality is best expressed as an explicit rubric.

It is intentionally domain-neutral. Keep private task sets, internal skills, customer data, and evaluation evidence outside a public SkillOpt fork and point the config to those files at runtime.

## Task schema

Each split contains an `items.json` array. A task supports:

```json
{
  "id": "task-001",
  "task_type": "example",
  "task": "What the target model should produce",
  "reference_context": "Facts the answer may rely on",
  "criteria": [
    {
      "name": "criterion_name",
      "description": "What good performance means",
      "weight": 2
    }
  ],
  "required_terms": [],
  "forbidden_terms": [],
  "minimum_score": 0.80
}
```

The target model receives the current skill as its system prompt and the task plus reference context as user input. The optimizer model then acts as a strict rubric judge and returns a score in `[0, 1]`. Required/forbidden term checks are deterministic guardrails on top of that score.

`hard=1` when the final score reaches `minimum_score`; `soft` is the final score. The target conversation is persisted so SkillOpt's normal reflection stage can derive bounded edits.

## Run

Use the dedicated launcher so the environment does not need to be added to SkillOpt's built-in registry:

```bash
python scripts/train_rubrictext.py --config /path/to/private/config.yaml
```

A starter config is available at `configs/rubrictext/default.yaml`. Private configs can inherit from it and override `env.skill_init` and `env.split_dir`.

## Data boundary

Do not place sensitive prompts or internal evaluation tasks in a public fork. Rubric judging sends task context and candidate output to the configured optimizer backend, so only provide data that is permitted to leave the local machine for that provider. Keep run outputs private when they contain internal prompts or model responses.

## Scoring guidance

Skill optimization is only as good as its scoring signal. Prefer criteria that are observable from the supplied reference context. Avoid vague aesthetic rubrics, hidden legal assumptions, or criteria that depend on knowledge not present in the task. Keep validation and test tasks held out from skill editing.
