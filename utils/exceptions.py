"""
FinScope AI - Custom Exceptions

Structured exception hierarchy for consistent error handling.
"""


class FinScopeError(Exception):
    """Base exception for FinScope AI."""

    def __init__(self, message: str, code: str = "FINSCOPE_ERROR"):
        self.message = message
        self.code = code
        super().__init__(self.message)


class DataValidationError(FinScopeError):
    """Raised when input data fails validation."""

    def __init__(self, message: str):
        super().__init__(message, code="DATA_VALIDATION_ERROR")


class ModelNotFoundError(FinScopeError):
    """Raised when a model artifact is not found."""

    def __init__(self, model_name: str):
        super().__init__(f"Model '{model_name}' not found", code="MODEL_NOT_FOUND")


class ModelInferenceError(FinScopeError):
    """Raised when model inference fails."""

    def __init__(self, message: str):
        super().__init__(message, code="MODEL_INFERENCE_ERROR")


class AuthenticationError(FinScopeError):
    """Raised for authentication failures."""

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, code="AUTHENTICATION_ERROR")


class DatabaseError(FinScopeError):
    """Raised for database operation failures."""

    def __init__(self, message: str):
        super().__init__(message, code="DATABASE_ERROR")
