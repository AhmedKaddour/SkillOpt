#!/usr/bin/env python3
"""Train a rubric-scored text skill without modifying SkillOpt's built-in registry.

This thin launcher keeps the generic RubricText environment isolated and makes
it usable with private datasets/configs outside this public repository.

Usage:
    python scripts/train_rubrictext.py --config /path/to/private/config.yaml
"""
from __future__ import annotations

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import scripts.train as train
from skillopt.envs.rubrictext.adapter import RubricTextAdapter


if __name__ == "__main__":
    train._ENV_REGISTRY["rubrictext"] = RubricTextAdapter
    train.main()
