"""
Image Preprocessor for Adaptive Learning System.

This module provides GPT-4o vision-based image understanding to convert
student-uploaded images (graphs, handwritten math, screenshots) into
text queries suitable for the existing retriever pipeline.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

# Type alias for detected image types
ImageType = Literal["graph", "diagram", "handwritten", "screenshot", "equation", "unknown"]


@dataclass
class ImageQueryResult:
    """Result from image preprocessing."""

    query: str
    """Text query suitable for the retriever."""

    detected_type: ImageType
    """Type of educational content detected in the image."""

    confidence: float
    """Confidence score from 0-1 for the detection."""

    extracted_text: Optional[str]
    """Any visible text or mathematical expressions extracted."""

    raw_analysis: Optional[str]
    """Full analysis from the vision model (for debugging)."""


class ImagePreprocessor:
    """
    Convert images to text queries using GPT-4o vision.

    This preprocessor analyzes educational images and generates text queries
    that can be fed into the TeachingPackRetriever for finding relevant
    learning objectives and content.

    Example usage:
        preprocessor = ImagePreprocessor()
        result = preprocessor.process_image("path/to/graph.png", "What is this showing?")
        # result.query can now be passed to the retriever
    """

    VISION_MODEL = "gpt-4o"

    # System prompt optimized for educational content analysis
    SYSTEM_PROMPT = """You are an expert educational content analyzer. Your task is to analyze images that students upload and generate search queries to find relevant learning materials.

When analyzing an image:
1. IDENTIFY the type: graph, diagram, handwritten work, screenshot, equation, or unknown
2. EXTRACT any visible text, mathematical expressions (use LaTeX notation), or labels
3. DESCRIBE what the image is showing in educational terms
4. GENERATE a concise search query (1-2 sentences) that would help find relevant learning objectives

Focus on mathematical and scientific content. Be specific about:
- Mathematical concepts (derivatives, integrals, limits, etc.)
- Graph types (linear, quadratic, exponential, trigonometric, etc.)
- Problem-solving techniques shown
- Any formulas or equations visible

Respond in JSON format:
{
    "detected_type": "graph|diagram|handwritten|screenshot|equation|unknown",
    "confidence": 0.0-1.0,
    "extracted_text": "any visible text or LaTeX math",
    "description": "what the image shows",
    "query": "search query for finding relevant learning objectives"
}"""

    def __init__(self, model: str = "gpt-4o") -> None:
        """
        Initialize the ImagePreprocessor.

        Args:
            model: OpenAI vision model to use. Defaults to gpt-4o.

        Raises:
            RuntimeError: If OPENAI_API_KEY environment variable is not set.
        """
        self._model = model
        self._client: Any = None

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable not set. "
                "Image preprocessing requires OpenAI API access."
            )

        try:
            from openai import OpenAI

            self._client = OpenAI(api_key=api_key)
        except ImportError as e:
            raise RuntimeError(
                "openai package is required for image preprocessing. "
                "Install with: pip install openai"
            ) from e

    def _load_image_as_base64(self, image: Union[str, bytes, Path]) -> str:
        """
        Convert image input to base64-encoded string.

        Args:
            image: Can be:
                - A file path (str or Path)
                - A URL (str starting with http:// or https://)
                - Raw bytes
                - Already base64-encoded string

        Returns:
            Base64-encoded image string.

        Raises:
            ValueError: If image format is not supported or file not found.
        """
        # Handle Path objects
        if isinstance(image, Path):
            image = str(image)

        # Handle bytes
        if isinstance(image, bytes):
            return base64.b64encode(image).decode("utf-8")

        # Handle string inputs
        if isinstance(image, str):
            # Check if it's a URL
            if image.startswith(("http://", "https://")):
                # Return URL directly - GPT-4o can handle URLs
                return image

            # Check if it's already base64 (simple heuristic)
            if len(image) > 100 and not os.path.exists(image):
                # Likely already base64
                try:
                    base64.b64decode(image)
                    return image
                except Exception:
                    pass

            # Treat as file path
            path = Path(image)
            if not path.exists():
                raise ValueError(f"Image file not found: {image}")

            if not path.is_file():
                raise ValueError(f"Path is not a file: {image}")

            # Read and encode file
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

        raise ValueError(f"Unsupported image type: {type(image)}")

    def _get_image_media_type(self, image: Union[str, bytes, Path]) -> str:
        """
        Determine the media type of the image.

        Args:
            image: Image input (path, URL, bytes, or base64).

        Returns:
            Media type string (e.g., "image/png", "image/jpeg").
        """
        if isinstance(image, Path):
            image = str(image)

        if isinstance(image, str):
            lower = image.lower()
            if lower.endswith(".png"):
                return "image/png"
            elif lower.endswith((".jpg", ".jpeg")):
                return "image/jpeg"
            elif lower.endswith(".gif"):
                return "image/gif"
            elif lower.endswith(".webp"):
                return "image/webp"

        # Default to jpeg for unknown types
        return "image/jpeg"

    def process_image(
        self,
        image: Union[str, bytes, Path],
        user_prompt: Optional[str] = None,
    ) -> ImageQueryResult:
        """
        Analyze an image and generate a text query for the retriever.

        Args:
            image: The image to analyze. Can be:
                - A file path (str or Path)
                - A URL (str starting with http:// or https://)
                - Raw bytes
                - Base64-encoded string
            user_prompt: Optional text prompt from the student providing context
                about what they're asking.

        Returns:
            ImageQueryResult containing the generated query and analysis.

        Raises:
            ValueError: If the image format is invalid.
            RuntimeError: If the API call fails.
        """
        # Build the user message content
        content: list[Dict[str, Any]] = []

        # Add user's text prompt if provided
        if user_prompt:
            content.append({
                "type": "text",
                "text": f"Student's question: {user_prompt}\n\nAnalyze the following image:",
            })
        else:
            content.append({
                "type": "text",
                "text": "Analyze the following educational image:",
            })

        # Handle image input
        image_data = self._load_image_as_base64(image)

        if image_data.startswith(("http://", "https://")):
            # URL-based image
            content.append({
                "type": "image_url",
                "image_url": {"url": image_data},
            })
        else:
            # Base64-encoded image
            media_type = self._get_image_media_type(image)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{image_data}",
                },
            })

        # Call the vision API
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                max_tokens=500,
                temperature=0.3,  # Lower temperature for more consistent output
            )

            raw_response = response.choices[0].message.content or ""

        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {e}") from e

        # Parse the JSON response
        return self._parse_response(raw_response, user_prompt)

    def _parse_response(
        self, raw_response: str, user_prompt: Optional[str]
    ) -> ImageQueryResult:
        """
        Parse the vision model's JSON response into an ImageQueryResult.

        Args:
            raw_response: Raw text response from the model.
            user_prompt: Original user prompt for fallback query generation.

        Returns:
            Parsed ImageQueryResult.
        """
        try:
            # Try to extract JSON from the response
            # Sometimes the model wraps JSON in markdown code blocks
            json_str = raw_response
            if "```json" in raw_response:
                start = raw_response.find("```json") + 7
                end = raw_response.find("```", start)
                json_str = raw_response[start:end].strip()
            elif "```" in raw_response:
                start = raw_response.find("```") + 3
                end = raw_response.find("```", start)
                json_str = raw_response[start:end].strip()

            data = json.loads(json_str)

            detected_type = data.get("detected_type", "unknown")
            if detected_type not in (
                "graph",
                "diagram",
                "handwritten",
                "screenshot",
                "equation",
                "unknown",
            ):
                detected_type = "unknown"

            confidence = float(data.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))  # Clamp to 0-1

            return ImageQueryResult(
                query=data.get("query", user_prompt or "educational content"),
                detected_type=detected_type,  # type: ignore
                confidence=confidence,
                extracted_text=data.get("extracted_text"),
                raw_analysis=raw_response,
            )

        except (json.JSONDecodeError, KeyError, TypeError):
            # Fallback: use raw response as query if JSON parsing fails
            return ImageQueryResult(
                query=user_prompt or raw_response[:200] if raw_response else "educational image",
                detected_type="unknown",
                confidence=0.3,
                extracted_text=None,
                raw_analysis=raw_response,
            )


def process_image_query(
    image: Union[str, bytes, Path],
    user_prompt: Optional[str] = None,
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

