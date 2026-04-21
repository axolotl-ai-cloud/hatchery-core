# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Custom exceptions raised across the platform."""


class PlatformError(Exception):
    """Base class for platform exceptions."""


class SessionNotFoundError(PlatformError):
    """Raised when a session lookup fails."""


class SessionAccessError(PlatformError):
    """Raised when a user tries to access a session they don't own."""


class SessionStateError(PlatformError):
    """Raised when a session is in a state incompatible with the operation."""


class QuotaExceededError(PlatformError):
    """Raised when a user exceeds their resource quota."""


class JobTimeoutError(PlatformError):
    """Raised when a job does not complete before the timeout."""


class JobFailedError(PlatformError):
    """Raised when a job completes with a failure status."""


class AuthenticationError(PlatformError):
    """Raised when bearer token is invalid or missing."""


class AuthorizationError(PlatformError):
    """Raised when user lacks permission for the requested action."""


class ObjectStoreError(PlatformError):
    """Raised on object store I/O failures."""


class InsufficientBalanceError(PlatformError):
    """Raised when user's prepaid balance is too low for the requested operation."""
