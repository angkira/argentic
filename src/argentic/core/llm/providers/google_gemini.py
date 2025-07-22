import os
import re
import time
import asyncio
from typing import Any, Dict, List, Optional, Union
from functools import wraps

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

# Use Google's existing error infrastructure
import google.api_core.exceptions as google_exceptions
from google.api_core.exceptions import (
    GoogleAPICallError,
    ResourceExhausted,
    InvalidArgument,
    PermissionDenied,
    InternalServerError,
    DeadlineExceeded,
    ServiceUnavailable,
    BadRequest,
    NotFound,
    Unauthenticated,
    Unknown,
)

# Use tenacity for retry logic as recommended by Google and LangChain
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
    before_sleep_log,
    after_log,
)

from argentic.core.llm.providers.base import ModelProvider
from argentic.core.logger import get_logger, LogLevel

# Removed google.generativeai imports as they are no longer used
# import google.generativeai as genai
# from google.generativeai import types

# --- Patch for langchain_google_genai finish_reason bug ---------------------------------
# In some Gemini responses the `candidate.finish_reason` field is returned as a plain
# integer instead of the expected enum. LangChain 0.1.x accesses the `.name` attribute
# unconditionally, which crashes with `AttributeError: 'int' object has no attribute 'name'`.
# We monkey-patch the internal helper to coerce ints to a dummy enum-like object.

try:
    from langchain_google_genai import chat_models as _gchat

    if hasattr(_gchat, "_response_to_result"):
        _orig_resp_to_res = _gchat._response_to_result

        def _coerce_finish_reason(obj):
            """Recursively coerce finish_reason ints to enum-like objects."""
            # Handle the object itself
            if hasattr(obj, "finish_reason"):
                fr = getattr(obj, "finish_reason")
                if isinstance(fr, int):

                    class _DummyEnum(int):
                        @property
                        def name(self):
                            return str(self)

                    setattr(obj, "finish_reason", _DummyEnum(fr))

            # Recursively handle nested objects
            # Check for candidates
            if hasattr(obj, "candidates"):
                candidates = getattr(obj, "candidates")
                if candidates:
                    for c in candidates:
                        _coerce_finish_reason(c)

            # Check for parts
            if hasattr(obj, "parts"):
                parts = getattr(obj, "parts")
                if parts:
                    for p in parts:
                        _coerce_finish_reason(p)

            # Check for other common nested structures
            for attr_name in ["generations", "generation", "content", "response"]:
                if hasattr(obj, attr_name):
                    attr_value = getattr(obj, attr_name)
                    if attr_value is not None:
                        if isinstance(attr_value, (list, tuple)):
                            for item in attr_value:
                                if hasattr(item, "__dict__"):  # Only recurse if it's an object
                                    _coerce_finish_reason(item)
                        elif hasattr(attr_value, "__dict__"):  # Only recurse if it's an object
                            _coerce_finish_reason(attr_value)

        def _safe_response_to_result(response):  # type: ignore
            try:
                return _orig_resp_to_res(response)
            except AttributeError as e:
                # Coerce ints to enum-like objects then retry once
                if "'int' object has no attribute 'name'" in str(e):
                    _coerce_finish_reason(response)
                    try:
                        return _orig_resp_to_res(response)
                    except AttributeError as e2:
                        # If it still fails, try to provide a more helpful error
                        raise AttributeError(f"Failed to fix finish_reason enum issue: {e2}") from e
                else:
                    # Re-raise if it's a different AttributeError
                    raise

        _gchat._response_to_result = _safe_response_to_result  # type: ignore
except Exception as e:
    # If patching fails we proceed silently – worst case the original bug remains.
    pass
# ---------------------------------------------------------------------------------------


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

        # Configure langchain-google-genai with error handling
        langchain_config = {
            "model": model_name,
            "google_api_key": self.api_key,
            "max_retries": 0,  # We handle retries ourselves for better control
            "timeout": retry_config.get("request_timeout", 60),
        }

        # Add optional parameters from google_gemini_parameters section
        gemini_params = self.config.get("google_gemini_parameters", {})
        if "temperature" in gemini_params:
            langchain_config["temperature"] = gemini_params["temperature"]
        if "max_output_tokens" in gemini_params:
            langchain_config["max_output_tokens"] = gemini_params["max_output_tokens"]
        if "top_k" in gemini_params:
            langchain_config["top_k"] = gemini_params["top_k"]
        if "top_p" in gemini_params:
            langchain_config["top_p"] = gemini_params["top_p"]

        try:
            self.model = ChatGoogleGenerativeAI(**langchain_config)
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
        messages: List[BaseMessage],
        tools: Optional[List] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> BaseMessage:
        """
        Call Google Gemini API with comprehensive error handling and retry logic.

        Uses Google's native error infrastructure and recommended retry patterns.
        """

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
                # Gemini API requires a specific turn structure.
                # If the last message is a ToolMessage, it must be followed by a user role.
                # We adapt the message history to comply with this.
                adapted_messages = list(messages)
                if adapted_messages and isinstance(adapted_messages[-1], ToolMessage):
                    # The result from a tool is now presented as a user's observation
                    tool_message = adapted_messages.pop()
                    tool_name = getattr(
                        tool_message, "name", getattr(tool_message, "tool_call_id", "unknown_tool")
                    )
                    user_feedback_content = f"Tool '{tool_name}' returned:\n{tool_message.content}"

                    # We need to find the AIMessage that initiated the tool call
                    # and ensure the history is clean before adding the user feedback
                    if adapted_messages and isinstance(adapted_messages[-1], AIMessage):
                        # This AIMessage contained the tool_call, we keep it.
                        pass

                    adapted_messages.append(HumanMessage(content=user_feedback_content))

                llm_tools_to_pass = []

                # Add LangChain tools for function calling if provided
                if tools:
                    llm_tools_to_pass.extend(tools)

                # Bind all tools to the model
                model_to_invoke = self.model
                if llm_tools_to_pass:
                    model_to_invoke = self.model.bind_tools(llm_tools_to_pass)

                if asyncio.iscoroutinefunction(model_to_invoke.ainvoke):
                    response = await model_to_invoke.ainvoke(adapted_messages, **kwargs)
                else:
                    # Run sync call in thread executor to avoid blocking
                    loop = asyncio.get_running_loop()

                    def sync_call():
                        return model_to_invoke.invoke(adapted_messages, **kwargs)

                    response = await loop.run_in_executor(None, sync_call)  # type: ignore

                # Reset error count on successful call
                self.error_count = 0
                return response

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

    def invoke(self, prompt: str, **kwargs: Any) -> BaseMessage:
        """Synchronously invoke the model with a single prompt."""
        messages: List[BaseMessage] = [HumanMessage(content=prompt)]
        # Convert async call to sync using asyncio.run to avoid blocking
        import asyncio

        try:
            return asyncio.run(self.call_llm(messages, **kwargs))
        except RuntimeError:
            # If we're already in an event loop, use the async version directly
            # This should be handled by the caller using run_in_executor
            raise RuntimeError(
                "Cannot use invoke() from within an async context. Use ainvoke() instead, "
                "or call this method from run_in_executor()."
            )

    async def ainvoke(self, prompt: str, **kwargs: Any) -> BaseMessage:
        """Asynchronously invoke the model with a single prompt."""
        messages: List[BaseMessage] = [HumanMessage(content=prompt)]
        return await self.call_llm(messages, **kwargs)

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> BaseMessage:
        """Synchronously invoke the model with a list of chat messages."""
        # Convert dict messages to BaseMessage objects
        langchain_messages = self._convert_dict_messages_to_langchain(messages)

        # Convert async call to sync using asyncio.run to avoid blocking
        import asyncio

        try:
            return asyncio.run(self.call_llm(langchain_messages, **kwargs))
        except RuntimeError:
            # If we're already in an event loop, use the async version directly
            # This should be handled by the caller using run_in_executor
            raise RuntimeError(
                "Cannot use chat() from within an async context. Use achat() instead, "
                "or call this method from run_in_executor()."
            )

    async def achat(
        self, messages: List[Dict[str, str]], tools: Optional[List[Any]] = None, **kwargs: Any
    ) -> BaseMessage:
        """Asynchronously invoke the model with a list of chat messages."""
        # Convert dict messages to BaseMessage objects
        langchain_messages = self._convert_dict_messages_to_langchain(messages)
        return await self.call_llm(langchain_messages, tools=tools, **kwargs)

    def _convert_dict_messages_to_langchain(
        self, messages: List[Dict[str, str]]
    ) -> List[BaseMessage]:
        """Convert dictionary messages to LangChain BaseMessage objects."""
        langchain_messages = []

        for msg in messages:
            role = msg.get("role", "user").lower()
            content = msg.get("content", "")

            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role in ["assistant", "model"]:
                langchain_messages.append(AIMessage(content=content))
            elif role == "tool":
                tool_call_id = msg.get("tool_call_id")
                if tool_call_id:
                    langchain_messages.append(
                        ToolMessage(content=content, tool_call_id=tool_call_id)
                    )
                else:
                    # Fallback to HumanMessage if no tool_call_id
                    langchain_messages.append(HumanMessage(content=f"Tool output: {content}"))
            else:
                # Unknown role, treat as user message
                langchain_messages.append(HumanMessage(content=f"{role}: {content}"))

        return langchain_messages
