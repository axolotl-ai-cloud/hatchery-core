# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Verify core does not import hosted — enforces the open-core boundary."""

from __future__ import annotations

import ast
from pathlib import Path


def test_core_does_not_import_hosted():
    """No production file in hatchery.core may import hatchery.hosted."""
    repo_root = Path(__file__).resolve().parent.parent
    core_pkg = repo_root / "hatchery" / "core"
    violations: list[str] = []

    for py in sorted(core_pkg.rglob("*.py")):
        rel = py.relative_to(repo_root)
        # Tests are allowed to import hosted for integration testing.
        if "tests" in rel.parts:
            continue
        try:
            tree = ast.parse(py.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("hatchery.hosted"):
                        violations.append(f"{rel}:{node.lineno}: import {alias.name}")
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("hatchery.hosted")
            ):
                violations.append(f"{rel}:{node.lineno}: from {node.module}")

    assert not violations, "Core imports hosted:\n" + "\n".join(violations)
