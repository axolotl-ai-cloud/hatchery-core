# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Re-export from ``hatchery.client`` for backward compatibility.

The client SDK now lives in the standalone ``hatchery-client`` package.
Existing ``from hatchery.core.client import HatcheryClient`` imports
continue to work via this module.
"""

from hatchery.client import (  # noqa: F401
    HatcheryClient,
    HatcheryClientError,
    RequestFailedError,
    SamplingClient,
    TrainingClient,
)

__all__ = [
    "HatcheryClient",
    "HatcheryClientError",
    "RequestFailedError",
    "SamplingClient",
    "TrainingClient",
]
