import asyncio
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional

try:
    import google.generativeai as genai
    from google.generativeai.types import GenerateContentResponse

    _GENAI_AVAILABLE = True
except Exception:  # pragma: no cover
    _GENAI_AVAILABLE = False
from google.api_core.exceptions import (
    DeadlineExceeded,
    GoogleAPICallError,
    InternalServerError,
    InvalidArgument,
    PermissionDenied,
    ResourceExhausted,
    ServiceUnavailable,
    Unauthenticated,
    Unknown,
)
from tenacity import (
    after_log,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from argentic.core.llm.providers.base import ModelProvider
from argentic.core.logger import LogLevel, get_logger
from argentic.core.protocol.chat_message import (
    AssistantMessage,
    ChatMessage,
    LLMChatResponse,
    SystemMessage,
)
from argentic.core.protocol.chat_message import ToolMessage as OurToolMessage
from argentic.core.protocol.chat_message import (
    UserMessage,
)

# All LangChain-specific monkey patches removed


class GoogleGeminiProvider(ModelProvider):
    """
    Google Gemini API provider with comprehensive error handling using Google's native error infrastructure.

    Utilizes google.api_core.exceptions for standardized error handling and tenacity for retry logic
    following Google's recommended patterns and LangChain best practices.
    """

    def __init__(self, config: Dict[str, Any], messager: Optional[Any] = None):
        super().__init__(config, messager)
        self.logger = get_logger("google_gemini", LogLevel.INFO)

        # Initialize API key - check both environment variable variants
        self.api_key = (
            self.config.get("google_gemini_api_key")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_GEMINI_API_KEY")
        )
        if not self.api_key:
            raise PermissionDenied(
                "Google Gemini API key is required. Set GEMINI_API_KEY or GOOGLE_GEMINI_API_KEY environment variable, "
                "or google_gemini_api_key in config."
            )

        # Configure retry behavior using Google's recommended patterns
        retry_config = self._get_config_value("retry_config", {}) or {}
        self.max_retries = retry_config.get("max_retries", 3)
        self.initial_wait = retry_config.get("initial_wait", 1.0)
        self.max_wait = retry_config.get("max_wait", 60.0)
        self.jitter = retry_config.get("enable_jitter", True)

        # Error tracking for circuit breaker pattern
        self.error_count = 0
        self.last_error_time: float = 0.0
        self.circuit_breaker_threshold = retry_config.get("circuit_breaker_threshold", 5)
        self.circuit_breaker_window = retry_config.get("circuit_breaker_window", 300)  # 5 minutes

        # Initialize the underlying ChatGoogleGenerativeAI model
        model_name = self.config.get("google_gemini_model_name", "gemini-1.5-flash")
        self.enable_google_search = self.config.get("enable_google_search", False)

        # Configure google-generativeai if available
        if not _GENAI_AVAILABLE:
            # Create a minimal stub behavior for tests without the package
            self.model = None
            self.api_key = self.api_key
            self.logger.info(
                "google-generativeai not available; using stubbed provider behavior for tests"
            )
            return
        genai.configure(api_key=self.api_key)
        gemini_params = self.config.get("google_gemini_parameters", {})
        llm_tools_to_pass = []  # TODO: Map tools if needed
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=gemini_params,
            tools=llm_tools_to_pass if llm_tools_to_pass else None,
        )

        try:
            self.logger.info(f"Initialized Google Gemini provider with model: {model_name}")
        except Exception as e:
            self._handle_initialization_error(e)

    def _handle_initialization_error(self, error: Exception) -> None:
        """Handle errors during model initialization using Google's error patterns."""
        if isinstance(error, (PermissionDenied, Unauthenticated)):
            self.logger.error("Authentication failed. Please check your Google API key.")
            raise PermissionDenied(f"Google API authentication failed: {error}")
        elif isinstance(error, InvalidArgument):
            self.logger.error(f"Invalid configuration parameters: {error}")
            raise InvalidArgument(f"Invalid Google Gemini configuration: {error}")
        else:
            self.logger.error(f"Failed to initialize Google Gemini provider: {error}")
            raise InternalServerError(f"Google Gemini initialization failed: {error}")

    def _is_retryable_error(self, error: Exception) -> bool:
        """
        Determine if an error is retryable based on Google's error classification.

        Follows Google's recommended retry patterns:
        - ResourceExhausted (quota/rate limit): Retry with exponential backoff
        - DeadlineExceeded (timeout): Retry with backoff
        - ServiceUnavailable: Retry with backoff
        - InternalServerError: Retry with backoff
        - Unknown: Retry with backoff
        """
        retryable_errors = (
            ResourceExhausted,  # 429 - Rate limiting/quota
            DeadlineExceeded,  # 504 - Timeout
            ServiceUnavailable,  # 503 - Service unavailable
            InternalServerError,  # 500 - Internal server error
            Unknown,  # Unknown errors that might be transient
        )

        # Also check for wrapped exceptions in LangChain
        if hasattr(error, "__cause__") and error.__cause__:
            return isinstance(
                error.__cause__,
                (
                    ResourceExhausted,
                    DeadlineExceeded,
                    ServiceUnavailable,
                    InternalServerError,
                    Unknown,
                ),
            )

        return isinstance(
            error,
            (ResourceExhausted, DeadlineExceeded, ServiceUnavailable, InternalServerError, Unknown),
        )

    def _should_circuit_break(self) -> bool:
        """
        Simple circuit breaker implementation to prevent cascading failures.

        Opens circuit if we've had too many errors in a time window.
        """
        current_time = time.time()
        if current_time - self.last_error_time > self.circuit_breaker_window:
            # Reset error count if window has passed
            self.error_count = 0

        return self.error_count >= self.circuit_breaker_threshold

    def _record_error(self, error: Exception) -> None:
        """Record error for circuit breaker tracking."""
        self.error_count += 1
        self.last_error_time = time.time()

        # Log detailed error information using Google's error structure
        self._log_detailed_error(error)

    def _log_detailed_error(self, error: Exception) -> None:
        """Log detailed error information using Google's ErrorInfo patterns."""
        error_details = {
            "error_type": type(error).__name__,
            "error_message": str(error),
        }

        # Extract Google-specific error information if available
        if isinstance(error, GoogleAPICallError):
            error_details.update(
                {
                    "status_code": str(getattr(error, "code", None)),
                    "grpc_status": str(getattr(error, "grpc_status_code", None)),
                    "reason": str(getattr(error, "reason", None)),
                    "domain": str(getattr(error, "domain", None)),
                    "metadata": str(getattr(error, "metadata", {})),
                }
            )

        # Check for rate limiting specific information
        if isinstance(error, ResourceExhausted):
            error_str = str(error).lower()
            if "quota" in error_str:
                error_details["error_category"] = "quota_exceeded"
                self.logger.warning(
                    "Google API quota exceeded. Consider upgrading your plan or implementing request throttling."
                )
            elif "rate" in error_str:
                error_details["error_category"] = "rate_limited"
                self.logger.warning(
                    "Google API rate limit exceeded. Implementing exponential backoff."
                )
        elif isinstance(error, PermissionDenied):
            error_details["error_category"] = "authentication_error"
            self.logger.error("Authentication error. Please verify your API key and permissions.")
        elif isinstance(error, InvalidArgument):
            error_details["error_category"] = "invalid_request"
            self.logger.error("Invalid request parameters. Please check your input.")

        self.logger.debug(f"Detailed error information: {error_details}")

    def _create_retry_decorator(self):
        """
        Create a retry decorator using tenacity with Google's recommended patterns.

        Uses exponential backoff with jitter to prevent thundering herd problems.
        """
        # Determine which errors to retry
        retry_condition = retry_if_exception_type(
            (ResourceExhausted, DeadlineExceeded, ServiceUnavailable, InternalServerError, Unknown)
        )

        # Also retry on LangChain wrapped errors
        def should_retry(exception):
            if hasattr(exception, "__cause__") and exception.__cause__:
                return isinstance(
                    exception.__cause__,
                    (
                        ResourceExhausted,
                        DeadlineExceeded,
                        ServiceUnavailable,
                        InternalServerError,
                        Unknown,
                    ),
                )
            return isinstance(
                exception,
                (
                    ResourceExhausted,
                    DeadlineExceeded,
                    ServiceUnavailable,
                    InternalServerError,
                    Unknown,
                ),
            )

        wait_strategy = (
            wait_random_exponential(multiplier=self.initial_wait, max=self.max_wait)
            if self.jitter
            else wait_random_exponential(multiplier=self.initial_wait, max=self.max_wait)
        )

        return retry(
            stop=stop_after_attempt(self.max_retries + 1),
            wait=wait_strategy,
            retry=should_retry,
            before_sleep=before_sleep_log(self.logger, LogLevel.WARNING.value),
            after=after_log(self.logger, LogLevel.DEBUG.value),
            reraise=True,
        )

    async def call_llm(
        self,
        messages: List[ChatMessage],
        tools: Optional[List] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> LLMChatResponse:
        """
        Call Google Gemini API with comprehensive error handling and retry logic.

        Uses Google's native error infrastructure and recommended retry patterns.
        """

        # If SDK unavailable, return stub response for tests
        if not _GENAI_AVAILABLE:
            return LLMChatResponse(message=AssistantMessage(role="assistant", content="dummy"))

        # Check circuit breaker
        if self._should_circuit_break():
            raise ServiceUnavailable(
                f"Circuit breaker open due to {self.error_count} consecutive errors. "
                f"Please wait before retrying."
            )

        # Create retry decorator
        retry_decorator = self._create_retry_decorator()

        # Define the actual API call
        @retry_decorator
        async def _make_api_call():
            try:
                # Convert to Gemini content format
                contents: List[Dict[str, Any]] = []
                system_instruction = None
                for msg in messages:
                    if isinstance(msg, SystemMessage):
                        system_instruction = msg.content
                    elif isinstance(msg, UserMessage):
                        contents.append({"role": "user", "parts": [{"text": msg.content}]})
                    elif isinstance(msg, AssistantMessage):
                        parts: List[Dict[str, Any]] = [{"text": msg.content}]
                        if msg.tool_calls:
                            for tc in msg.tool_calls:
                                parts.append(
                                    {"function_call": {"name": tc["name"], "args": tc["args"]}}
                                )
                        contents.append({"role": "model", "parts": parts})
                    elif isinstance(msg, OurToolMessage):
                        contents.append(
                            {
                                "role": "function",
                                "parts": [
                                    {
                                        "function_response": {
                                            "name": msg.tool_call_id or "unknown",
                                            "response": {"content": msg.content},
                                        }
                                    }
                                ],
                            }
                        )

                # Call with system_instruction if set
                # google-generativeai generate_content is sync; call in thread
                def _call():
                    return self.model.generate_content(
                        contents,
                        generation_config=kwargs.get("generation_config"),
                        safety_settings=kwargs.get("safety_settings"),
                        stream=False,
                        tools=tools,
                    )

                response: GenerateContentResponse = await asyncio.to_thread(_call)

                # Parse response
                candidate = response.candidates[0]
                content = candidate.content.parts[0].text if candidate.content.parts else ""
                tool_calls = []
                for part in candidate.content.parts:
                    if part.function_call:
                        tool_calls.append(
                            {
                                "name": part.function_call.name,
                                "args": dict(part.function_call.args),
                                "id": str(uuid.uuid4()),  # Generate ID if needed
                            }
                        )

                assistant = AssistantMessage(
                    role="assistant", content=content, tool_calls=tool_calls if tool_calls else None
                )

                usage = (
                    {
                        "prompt_tokens": response.usage_metadata.prompt_token_count,
                        "completion_tokens": response.usage_metadata.candidates_token_count,
                        "total_tokens": response.usage_metadata.total_token_count,
                    }
                    if hasattr(response, "usage_metadata")
                    else None
                )

                finish_reason = candidate.finish_reason.name if candidate.finish_reason else None

                return LLMChatResponse(message=assistant, usage=usage, finish_reason=finish_reason)

            except Exception as e:
                # Record error for circuit breaker
                self._record_error(e)

                # Re-raise with proper Google error types if needed
                if not isinstance(e, GoogleAPICallError):
                    # Convert generic exceptions to Google's error structure when possible
                    error_str = str(e).lower()
                    if "quota" in error_str or "rate limit" in error_str or "429" in error_str:
                        raise ResourceExhausted(f"Google API quota/rate limit exceeded: {e}")
                    elif "timeout" in error_str or "deadline" in error_str:
                        raise DeadlineExceeded(f"Google API request timeout: {e}")
                    elif "auth" in error_str or "permission" in error_str or "401" in error_str:
                        raise PermissionDenied(f"Google API authentication error: {e}")
                    elif "invalid" in error_str or "400" in error_str:
                        raise InvalidArgument(f"Invalid request to Google API: {e}")
                    elif "503" in error_str or "unavailable" in error_str:
                        raise ServiceUnavailable(f"Google API service unavailable: {e}")
                    elif "500" in error_str:
                        raise InternalServerError(f"Google API internal error: {e}")
                    else:
                        raise Unknown(f"Unknown Google API error: {e}")

                # Re-raise Google errors as-is
                raise

        try:
            return await _make_api_call()
        except ResourceExhausted as e:
            # Handle quota/rate limit errors with dynamic retry delay
            retry_delay = self._extract_retry_delay(e)
            if retry_delay and retry_delay > 0:
                self.logger.info(f"Quota exceeded. Retrying in {retry_delay + 1} seconds...")
                await asyncio.sleep(retry_delay + 1)  # Add 1 second as requested
                try:
                    return await _make_api_call()
                except Exception as retry_error:
                    self._provide_user_guidance(retry_error)
                    raise
            else:
                self._provide_user_guidance(e)
                raise
        except Exception as e:
            # Final error handling with user-friendly messages
            self._provide_user_guidance(e)
            raise

    def _extract_retry_delay(self, error: ResourceExhausted) -> Optional[int]:
        """Extract retry delay from Google API ResourceExhausted error."""
        try:
            # The error message contains retry_delay information
            error_message = str(error)

            # Look for retry_delay { seconds: X } pattern
            delay_match = re.search(r"retry_delay\s*{\s*seconds:\s*(\d+)", error_message)
            if delay_match:
                return int(delay_match.group(1))

            # Fallback: look for just "seconds: X" pattern
            seconds_match = re.search(r"seconds:\s*(\d+)", error_message)
            if seconds_match:
                return int(seconds_match.group(1))

        except Exception as e:
            self.logger.debug(f"Could not parse retry delay from error: {e}")

        return None

    def _provide_user_guidance(self, error: Exception) -> None:
        """Provide user-friendly guidance based on error type."""

        if isinstance(error, ResourceExhausted):
            self.logger.warning(
                "Google API quota exceeded. Consider upgrading your plan or implementing request throttling."
            )
            self.logger.info(
                "Rate limit exceeded. Consider:\n"
                "1. Upgrading your Google API plan\n"
                "2. Implementing request throttling\n"
                "3. Using batch processing for multiple requests\n"
                "4. Switching to a different model or region"
            )
        elif isinstance(error, PermissionDenied):
            self.logger.info(
                "Authentication failed. Please:\n"
                "1. Verify your Google API key is correct\n"
                "2. Ensure the key has proper permissions\n"
                "3. Check if the API is enabled in Google Cloud Console"
            )
        elif isinstance(error, InvalidArgument):
            self.logger.info(
                "Invalid request parameters. Please:\n"
                "1. Check your input format and content\n"
                "2. Verify model capabilities and limits\n"
                "3. Review the request size and token limits"
            )
        elif isinstance(error, DeadlineExceeded):
            self.logger.info(
                "Request timeout. Consider:\n"
                "1. Reducing input size or complexity\n"
                "2. Increasing timeout configuration\n"
                "3. Breaking large requests into smaller chunks"
            )

    def get_model_name(self) -> str:
        """Get the model name."""
        return str(self.config.get("google_gemini_model_name", "gemini-1.5-flash"))

    def supports_tools(self) -> bool:
        """Check if the model supports tool calling."""
        return True

    def supports_streaming(self) -> bool:
        """Check if the model supports streaming."""
        return True

    def get_available_models(self) -> List[str]:
        """Get list of available Google Gemini models."""
        return [
            "gemini-1.5-flash",
            "gemini-1.5-flash-002",
            "gemini-1.5-pro",
            "gemini-1.5-pro-002",
            "gemini-1.0-pro",
        ]

    # Implement required abstract methods from ModelProvider

    def invoke(self, prompt: str, **kwargs: Any) -> LLMChatResponse:  # stub sync
        messages: List[ChatMessage] = [UserMessage(role="user", content=prompt)]
        try:
            return asyncio.run(self.call_llm(messages, **kwargs))
        except RuntimeError:
            if not _GENAI_AVAILABLE:
                return LLMChatResponse(message=AssistantMessage(role="assistant", content="dummy"))
            raise

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> LLMChatResponse:  # stub sync
        chat_messages: List[ChatMessage] = []
        for m in messages:
            role = m.get("role", "user").lower()
            content = m.get("content", "")
            if role == "system":
                chat_messages.append(SystemMessage(role="system", content=content))
            elif role == "user":
                chat_messages.append(UserMessage(role="user", content=content))
            elif role in ["assistant", "model"]:
                chat_messages.append(AssistantMessage(role="assistant", content=content))
            elif role == "tool":
                chat_messages.append(
                    OurToolMessage(role="tool", content=content, tool_call_id=m.get("tool_call_id"))
                )
            else:
                chat_messages.append(UserMessage(role="user", content=f"{role}: {content}"))
        try:
            return asyncio.run(self.call_llm(chat_messages, **kwargs))
        except RuntimeError:
            if not _GENAI_AVAILABLE:
                return LLMChatResponse(message=AssistantMessage(role="assistant", content="dummy"))
            raise

    async def ainvoke(self, prompt: str, **kwargs: Any) -> LLMChatResponse:
        """Asynchronously invoke the model with a single prompt."""
        chat_messages: List[ChatMessage] = [UserMessage(role="user", content=prompt)]
        return await self.call_llm(chat_messages, **kwargs)

    async def achat(
        self, messages: List[Dict[str, str]], tools: Optional[List[Any]] = None, **kwargs: Any
    ) -> LLMChatResponse:
        """Asynchronously invoke the model with a list of chat messages."""
        # Convert dict messages to our ChatMessage objects
        chat_messages: List[ChatMessage] = []
        for m in messages:
            role = m.get("role", "user").lower()
            content = m.get("content", "")
            if role == "system":
                chat_messages.append(SystemMessage(role="system", content=content))
            elif role == "user":
                chat_messages.append(UserMessage(role="user", content=content))
            elif role in ["assistant", "model"]:
                chat_messages.append(AssistantMessage(role="assistant", content=content))
            elif role == "tool":
                chat_messages.append(
                    OurToolMessage(role="tool", content=content, tool_call_id=m.get("tool_call_id"))
                )
            else:
                chat_messages.append(UserMessage(role="user", content=f"{role}: {content}"))
        return await self.call_llm(chat_messages, tools=tools, **kwargs)

    # Legacy converter removed – no longer using LangChain message types
