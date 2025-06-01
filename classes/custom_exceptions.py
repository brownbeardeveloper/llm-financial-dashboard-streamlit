class ChatError(Exception):
    """Base class for OpenAI chat-related errors."""


class MissingOpenAIKeyError(ChatError):
    """Raised when API key is missing."""


class MessageValidationError(ChatError):
    """Raised when message format to OpenAI API is invalid."""


class OpenAIErrorParser:
    """Utility to parse OpenAI exceptions and return user-friendly error messages."""

    @staticmethod
    def get_user_friendly_message(error: Exception) -> str:
        error_str = str(error).lower()

        if hasattr(error, "status_code"):
            code = error.status_code
        elif "401" in error_str or "incorrect api key" in error_str:
            code = 401
        elif "429" in error_str or "rate limit" in error_str:
            code = 429
        elif "500" in error_str:
            code = 500
        elif "503" in error_str:
            code = 503
        elif "timeout" in error_str:
            code = "timeout"
        elif "overloaded" in error_str:
            code = "overloaded"
        else:
            code = "unknown"

        return {
            401: "Invalid API key (Error: 401)",
            429: "Rate limit exceeded (Error: 429)",
            500: "Unexpected error occurred (Error: 500)",
            503: "Service temporarily unavailable (Error: 503)",
            "timeout": "Request timed out (Error: timeout)",
            "overloaded": "AI service is busy (Error: overloaded)",
        }.get(code, f"Unexpected error occurred (Error: {code})")
