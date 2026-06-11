"""Venvstacks launch module for the Hatchery Core gateway."""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.getenv("HATCHERY_GATEWAY_HOST", "0.0.0.0")
    port = int(os.getenv("HATCHERY_GATEWAY_PORT", os.getenv("PORT", "8420")))
    uvicorn.run(
        "hatchery.core.gateway:create_app",
        factory=True,
        host=host,
        port=port,
    )


if __name__ == "__main__":
    main()
