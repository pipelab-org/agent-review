# Copyright 2026 Pipelock contributors
# SPDX-License-Identifier: Apache-2.0
"""Fixed, operator-safe failure types shared by the planner and the engine."""
from __future__ import annotations

try:
    from . import review_reliability as reliability
except ImportError:
    import review_reliability as reliability


class ReviewAbort(Exception):
    """Fixed, operator-safe failure message."""


class InputLimitAbort(ReviewAbort, reliability.Reasoned):
    """Even one bounded section cannot be framed within the depth's max_model_tokens allowance.

    Raised only where the engine sizes input before a pass launches. A request that
    exceeds the allowance after a pass ran is a different failure and stays a ReviewAbort.
    """

    reason = "input_limit"


class IncompleteReviewAbort(ReviewAbort, reliability.Reasoned):
    """A logical pass stopped before every required section completed.

    `progress` carries only counters (completed, total sections and the strategy name
    from the fixed DEPTHS vocabulary); never provider or model text. Nothing recorded
    before the stop is published as coverage.
    """

    reason = "sections_incomplete"

    def __init__(self, message, progress=None):
        super().__init__(message)
        self.progress = progress
