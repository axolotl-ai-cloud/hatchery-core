# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Minimal example of an in-process loss plugin.

Demonstrates the two ways operators can extend the server-side loss
registry:

1. **Entry-point packaging.** A third-party package ships
   ``scaled_cross_entropy`` (or similar) by adding a fragment like
   this to its ``pyproject.toml``::

       [project.entry-points."hatchery.losses"]
       scaled_cross_entropy = "hatchery.core.examples.loss_plugin_example:scaled_cross_entropy"

   On first dispatch through :func:`hatchery.core.losses.compute`, the
   registry will discover and register the entry point automatically.

2. **In-process registration.** Tests and operator-internal code that
   doesn't want to package a wheel can register a callable directly
   via :func:`hatchery.core.losses.register_loss`. The convenience
   helper :func:`install` below shows the pattern.

The plugin signature is::

    fn(inputs: LossInputs) -> Tensor | tuple[Tensor, dict]

i.e. the same contract as the built-in losses in
:mod:`hatchery.core.losses`.
"""

from __future__ import annotations

from typing import Any

from hatchery.core.losses import LossInputs, compute, register_loss

DEFAULT_SCALE = 0.5


def scaled_cross_entropy(inputs: LossInputs) -> Any:
    """Built-in cross-entropy multiplied by a constant scale factor.

    The scale is read from ``loss_fn_config["scale"]`` if present,
    falling back to :data:`DEFAULT_SCALE`. The intent is purely
    illustrative — it gives plugin authors a working reference for
    how to read ``loss_fn_config`` and delegate to another registered
    loss via the public :func:`hatchery.core.losses.compute` entry point.
    """
    cfg = inputs.loss_fn_config or {}
    scale = float(cfg.get("scale", DEFAULT_SCALE))
    return compute("cross_entropy", inputs) * scale


def install() -> None:
    """Register :func:`scaled_cross_entropy` under ``"scaled_cross_entropy"``.

    Re-runnable: passes ``override=True`` so repeated calls within a
    single test process silently overwrite an existing entry rather
    than raising. Real plugin packages should normally omit
    ``override=True`` — production code wants a hard failure if two
    plugins try to claim the same name.
    """
    register_loss("scaled_cross_entropy", scaled_cross_entropy, override=True)
