"""Typed errors raised by core code. UI pages catch these and show a friendly next step."""

from __future__ import annotations


class RiskLabError(Exception):
    """Base class for all Portfolio Risk Lab errors."""


class InsufficientDataError(RiskLabError):
    """Not enough history (or observations) to compute the requested quantity."""


class DataUnavailableError(RiskLabError):
    """Data could not be obtained from any source (live, cache, or bundle)."""


class PortfolioValidationError(RiskLabError, ValueError):
    """A portfolio definition is invalid (negative weights, empty, unknown tickers, ...)."""


class LLMUnavailableError(RiskLabError):
    """No LLM is configured, the call budget is exhausted, or the API failed."""
