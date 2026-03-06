"""
Image preprocessing bridge from student images to retriever-ready text.

This module analyzes student-uploaded images (graphs, equations,
handwritten work, screenshots) and converts them into structured
signals that the text-first retrieval pipeline can use.

What it does:
1. Takes an image and optional student text.
2. Sends both to GPT-4o Vision.
3. Gets back a clean retrieval query plus image metadata.
4. Returns a safe default result if model output is malformed.

How this is achieved:
- ``ImagePreprocessor.process_image`` builds a multimodal chat payload
  and calls the vision model using a strict JSON-oriented system prompt.
- ``_load_image_as_base64`` and ``_get_image_media_type`` handle
  defensive input normalization so callers can pass multiple image forms.
- ``_parse_response`` validates and sanitizes model output into
  ``ImageQueryResult``, including markdown-JSON extraction and defaults.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from openai import OpenAI

# Type alias for detected image types
ImageType = Literal["graph", "diagram", "handwritten", "screenshot", "equation", "unknown"]
VALID_IMAGE_TYPES = frozenset(
    {"graph", "diagram", "handwritten", "screenshot", "equation", "unknown"}
)
IMAGE_URL_PREFIXES = ("http://", "https://")
SUFFIX_TO_MEDIA_TYPE = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


@dataclass
class ImageQueryResult:
    """Result from image preprocessing."""

    query: str
    """Text query suitable for the retriever."""

    detected_type: ImageType
    """Type of educational content detected in the image."""

    confidence: float
    """Confidence score from 0-1 for the detection."""

    latex_content: list[str] = field(default_factory=list)
    """List of LaTeX strings extracted from the image."""

    key_features: list[str] = field(default_factory=list)
    """Notable features such as intercepts, asymptotes, or labeled elements."""

    likely_topic: str | None = None
    """Best-guess topic to guide retrieval (e.g., derivatives, limits)."""


class ImagePreprocessor:
    """
    Convert images to text queries using GPT-4o Vision.

    This preprocessor analyzes educational images and generates text queries
    that can be fed into the TeachingPackRetriever for finding relevant
    learning objectives and content.

    Example usage:
        preprocessor = ImagePreprocessor()
        result = preprocessor.process_image("path/to/graph.png", "What is this showing?")
        # result.query can now be passed to the retriever
    """

    VISION_MODEL = "gpt-4o"

    # System prompt optimized for retrieval-oriented educational image analysis.
    SYSTEM_PROMPT = """You are an educational image-to-query converter for a math tutoring retrieval system.

Your job is not to write a full visual report. Your job is to convert a
student image into a short, accurate, retrieval-ready representation of the
underlying math concept.

Priorities:
1. Identify the most likely learning objective or math topic.
2. Produce a concise search query that would retrieve the right tutoring content.
3. Extract visible math expressions as LaTeX when they are clearly readable.
4. Return only the most important visual features that help disambiguate the topic.

Rules:
- Prefer canonical textbook topic names over vague visual descriptions.
- Use the student's text prompt as extra context when it helps clarify intent.
- Do not invent unreadable text, labels, or equations.
- If the image is ambiguous, return a broader topic, fewer details, and lower confidence.
- Return at most 3 key_features.
- Return valid JSON only. No markdown. No extra commentary.

Field guidance:
- detected_type: one of graph, equation, handwritten, diagram, screenshot, unknown.
- latex_content: list only clearly readable expressions in LaTeX; otherwise [].
- key_features: short phrases describing the most useful visual clues.
- likely_topic: one main textbook-style topic name, or "unknown" if unclear.
- query: a short retrieval query using canonical math wording, not a full sentence.
- confidence: a number from 0.0 to 1.0.

Confidence rubric:
- 0.85-1.0: topic is very clear and strongly supported by visible evidence.
- 0.60-0.84: likely topic is clear but some details are uncertain.
- 0.30-0.59: only a broad topic can be inferred.
- 0.00-0.29: image is unclear or does not provide enough information.

OUTPUT JSON:
{
    "detected_type": "graph|equation|handwritten|diagram|screenshot|unknown",
    "latex_content": ["\\\\frac{d}{dx}(x^2)", "..."],
    "key_features": ["x-intercepts visible", "worked derivative steps"],
    "likely_topic": "derivatives|limits|trigonometric equations|complex numbers|unknown",
    "query": "short retrieval query using canonical math topic wording",
    "confidence": 0.0
}"""

    def __init__(self, model: str = "gpt-4o") -> None:
        """Initialize model + OpenAI client."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable not set.")

        self._model = model
        self._client: Any = OpenAI(api_key=api_key)

    def _load_image_as_base64(self, image: str | bytes | Path) -> str:
        """Normalize image input to either URL or base64 payload."""
        # Raw bytes -> base64 payload.
        if isinstance(image, bytes):
            return base64.b64encode(image).decode("utf-8")

        # Normalize to string for URL/path/base64 checks.
        image_str = str(image) if isinstance(image, Path) else image
        if not isinstance(image_str, str):
            raise ValueError(f"Unsupported image type: {type(image)}")

        # URLs are forwarded as-is to the model.
        if image_str.startswith(IMAGE_URL_PREFIXES):
            return image_str

        # Existing local file -> read and encode.
        path = Path(image_str)
        if path.is_file():
            return base64.b64encode(path.read_bytes()).decode("utf-8")
        if path.exists():
            raise ValueError(f"Path is not a file: {image_str}")

        # Fallback: treat long non-path strings as base64 payloads.
        if self._try_decode_base64(image_str) is not None:
            return image_str

        raise ValueError(f"Image file not found: {image_str}")

    def _get_image_media_type(self, image: str | bytes | Path) -> str:
        """Infer media type from suffix or image bytes; default to JPEG."""
        # Raw bytes -> detect directly from signature.
        if isinstance(image, bytes):
            return self._media_type_from_bytes(image)

        # Normalize to string for suffix checks.
        image_str = str(image) if isinstance(image, Path) else image
        if not isinstance(image_str, str):
            return "image/jpeg"

        lower = image_str.lower().split("?", 1)[0]

        # Prefer file/URL suffix when available.
        for suffix, media_type in SUFFIX_TO_MEDIA_TYPE.items():
            if lower.endswith(suffix):
                return media_type

        # If this is base64 text, inspect decoded bytes.
        decoded = self._try_decode_base64(image_str)
        if decoded is not None:
            return self._media_type_from_bytes(decoded)

        return "image/jpeg"

    @staticmethod
    def _media_type_from_bytes(image_bytes: bytes) -> str:
        """Infer media type from common image file signatures."""
        if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if image_bytes.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if image_bytes.startswith((b"GIF87a", b"GIF89a")):
            return "image/gif"
        if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
            return "image/webp"
        return "image/jpeg"

    def process_image(
        self,
        image: str | bytes | Path,
        user_prompt: str | None = None,
    ) -> ImageQueryResult:
        """Analyze an image and return a retrieval-ready result."""
        # Clean prompt text once so whitespace-only input is treated as empty.
        prompt_text = (user_prompt or "").strip() or None

        # Build the text instruction that gives the model student context.
        text_part = {
            "type": "text",
            "text": (
                f"Student's question: {prompt_text}\n\n"
                "Analyze the following image:"
                if prompt_text
                else "Analyze the following educational image:"
            ),
        }

        # Normalize the image into either a direct URL or a base64 data URL.
        image_data = self._load_image_as_base64(image)
        image_url = image_data
        if not image_data.startswith(("http://", "https://")):
            media_type = self._get_image_media_type(image)
            image_url = f"data:{media_type};base64,{image_data}"

        # Build the multimodal user message expected by the vision model.
        content: list[dict[str, Any]] = [
            text_part,
            {
                "type": "image_url",
                "image_url": {"url": image_url},
            },
        ]

        # Call the vision model and request JSON-shaped output.
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                max_tokens=500,
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            raw_response = response.choices[0].message.content or ""
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {e}") from e

        # Parse the structured response into the typed result object.
        return self._parse_response(raw_response, prompt_text)

    def _parse_response(
        self, raw_response: str, user_prompt: str | None
    ) -> ImageQueryResult:
        """Parse model JSON into ImageQueryResult with safe defaults."""
        try:
            # Parse structured JSON response from the vision model.
            data = json.loads(raw_response)

            # Normalize fields so malformed model output degrades safely.
            detected_type = str(data.get("detected_type", "unknown")).strip().lower()
            if detected_type not in VALID_IMAGE_TYPES:
                detected_type = "unknown"

            confidence = self._coerce_confidence(data.get("confidence"))
            query = str(
                data.get("query") or user_prompt or "educational content"
            ).strip()
            likely_topic = data.get("likely_topic")
            if isinstance(likely_topic, str):
                likely_topic = likely_topic.strip() or None
            else:
                likely_topic = None

            return ImageQueryResult(
                query=query,
                detected_type=detected_type,  # type: ignore
                confidence=confidence,
                latex_content=self._coerce_string_list(data.get("latex_content")),
                key_features=self._coerce_string_list(data.get("key_features")),
                likely_topic=likely_topic,
            )

        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            # Fallback: return a conservative best-effort result.
            return ImageQueryResult(
                query=(
                    user_prompt
                    or raw_response[:200]
                    if raw_response
                    else "educational image"
                ),
                detected_type="unknown",
                confidence=0.3,
                latex_content=[],
                key_features=[],
                likely_topic=None,
            )

    @staticmethod
    def _try_decode_base64(value: str) -> bytes | None:
        """Decode base64 text; return None when not valid base64."""
        if len(value) <= 100:
            return None
        try:
            return base64.b64decode(value, validate=True)
        except Exception:
            return None

    @staticmethod
    def _coerce_confidence(value: Any) -> float:
        """Convert model confidence to a clamped float in [0.0, 1.0]."""
        try:
            score = float(value if value is not None else 0.5)
        except (TypeError, ValueError):
            score = 0.5
        return max(0.0, min(1.0, score))

    @staticmethod
    def _coerce_string_list(value: Any) -> list[str]:
        """Normalize model list fields into a clean list of strings."""
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]


def process_image_query(
    image: str | bytes | Path,
    user_prompt: str | None = None,
    model: str = "gpt-4o",
) -> str:
    """
    Convenience function to process an image and return just the query string.

    This is the main integration point for the coach/retriever to use.

    Args:
        image: Image to analyze (path, URL, bytes, or base64).
        user_prompt: Optional context from the student.
        model: Vision model to use.

    Returns:
        Text query suitable for the retriever.

    Example:
        query = process_image_query("homework.jpg", "I don't understand this graph")
        plan = retriever.retrieve_plan(query, subject="calculus", ...)
    """
    preprocessor = ImagePreprocessor(model=model)
    result = preprocessor.process_image(image, user_prompt)
    return result.query

