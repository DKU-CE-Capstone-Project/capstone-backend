"""Stable public errors without internal exception details."""
import uuid


class V2Error(Exception):
    def __init__(self, status: int, code: str, message: str, retryable: bool = False, details: dict | None = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.retryable = retryable
        self.details = details or {}

    def envelope(self) -> dict:
        return {"error": {"code": self.code, "message": self.message,
                          "retryable": self.retryable, "details": self.details},
                "request_id": str(uuid.uuid4())}
