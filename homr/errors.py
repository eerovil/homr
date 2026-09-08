"""Failures that must not be presented as complete recognition results."""


class IncompleteRecognitionError(RuntimeError):
    """A requested reading is incomplete and must not be exported as a full score."""
