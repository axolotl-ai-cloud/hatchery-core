# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Backend-neutral types for speculative decoding request options and response metadata.

This module defines the protocol surface that hatchery-core exposes for
speculative decoding. No backend-specific (DFlash, ngram, etc.) logic lives
here — that belongs in extension packages such as hatchery-hosted.

Clients set ``SamplingParams.speculative_decoding`` to opt in or out of
speculative decoding on a per-request basis. The server-side policy resolver
(e.g. ``DFlashPolicyResolver`` in hatchery-hosted) consumes these options and
returns a ``SpeculativeDecodingMetadata`` describing what was actually used.

Sampling behavior is unchanged when ``speculative_decoding`` is ``None`` or
when the server has no policy configured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, Field

SPEC_BACKEND_DFLASH = "dflash"


class SpeculativeDecodingRequest(BaseModel):
    """Request-level speculative decoding preferences.

    All fields are optional. Omitting them (or setting ``enable=None``) defers
    to the server's configured default policy: if the server has a DFlash draft
    model configured, the request will use speculative decoding automatically.
    Setting ``enable=False`` explicitly disables speculative decoding for this
    request regardless of server defaults. Setting ``enable=True`` is equivalent
    to ``None`` for DFlash (server config still gates eligibility) but will
    trigger ``fallback_reason`` in the response metadata if DFlash is not
    configured, making the intent visible to the caller.
    """

    enable: Optional[bool] = None
    backend: Optional[str] = None
    max_draft_tokens: Optional[int] = Field(None, ge=1, le=64)
    strict: bool = False


@dataclass
class SpeculativeDecodingMetadata:
    """Metadata describing the speculative decoding used for a completed request.

    Attached to sampling responses when a policy resolver is active. All
    fields are ``None`` when speculative decoding was not evaluated.
    """

    requested_backend: Optional[str] = None
    used_backend: Optional[str] = None
    draft_model: Optional[str] = None
    max_draft_tokens: Optional[int] = None
    fallback_reason: Optional[str] = None
