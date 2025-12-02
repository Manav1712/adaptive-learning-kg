"""
Unit tests for the ImagePreprocessor module.
"""

import base64
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.workflow_demo.image_preprocessor import (
    ImagePreprocessor,
    ImageQueryResult,
    process_image_query,
)

# Patch target for OpenAI - it's imported inside the __init__ method
OPENAI_PATCH_TARGET = "openai.OpenAI"


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client with a vision response."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps({
        "detected_type": "graph",
        "confidence": 0.9,
        "extracted_text": "f(x) = x^2",
        "description": "A parabola showing quadratic function",
        "query": "quadratic functions and parabolas",
    })
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def sample_image_bytes():
    """Create minimal valid PNG bytes for testing."""
    # Minimal 1x1 PNG image
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )


@pytest.fixture
def temp_image_file(sample_image_bytes):
    """Create a temporary image file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(sample_image_bytes)
        path = f.name
    yield path
    # Cleanup
    if os.path.exists(path):
        os.unlink(path)


# -----------------------------------------------------------------------------
# Initialization Tests
# -----------------------------------------------------------------------------


@pytest.mark.unit
def test_image_preprocessor_requires_api_key(monkeypatch):
    """ImagePreprocessor should raise RuntimeError if OPENAI_API_KEY is not set."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY environment variable not set"):
        ImagePreprocessor()


@pytest.mark.unit
def test_image_preprocessor_initializes_with_api_key(monkeypatch, mock_openai_client):
    """ImagePreprocessor should initialize successfully with API key."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    
    with patch(OPENAI_PATCH_TARGET) as mock_openai:
        mock_openai.return_value = mock_openai_client
        preprocessor = ImagePreprocessor()
        assert preprocessor._client is not None
        assert preprocessor._model == "gpt-4o"


@pytest.mark.unit
def test_image_preprocessor_custom_model(monkeypatch, mock_openai_client):
    """ImagePreprocessor should accept custom model parameter."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    
    with patch(OPENAI_PATCH_TARGET) as mock_openai:
        mock_openai.return_value = mock_openai_client
        preprocessor = ImagePreprocessor(model="gpt-4o-mini")
        assert preprocessor._model == "gpt-4o-mini"


# -----------------------------------------------------------------------------
# Image Loading Tests
# -----------------------------------------------------------------------------


@pytest.mark.unit
def test_load_image_from_file_path(monkeypatch, mock_openai_client, temp_image_file):
    """Should load and encode image from file path."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    
    with patch(OPENAI_PATCH_TARGET) as mock_openai:
        mock_openai.return_value = mock_openai_client
        preprocessor = ImagePreprocessor()
        
        result = preprocessor._load_image_as_base64(temp_image_file)
        
        # Should return valid base64
        assert isinstance(result, str)
        decoded = base64.b64decode(result)
        assert len(decoded) > 0


@pytest.mark.unit
def test_load_image_from_path_object(monkeypatch, mock_openai_client, temp_image_file):
    """Should load image from Path object."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    
    with patch(OPENAI_PATCH_TARGET) as mock_openai:
        mock_openai.return_value = mock_openai_client
        preprocessor = ImagePreprocessor()
        
        result = preprocessor._load_image_as_base64(Path(temp_image_file))
        
        assert isinstance(result, str)
        base64.b64decode(result)  # Should not raise


@pytest.mark.unit
def test_load_image_from_bytes(monkeypatch, mock_openai_client, sample_image_bytes):
    """Should encode raw bytes to base64."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    
    with patch(OPENAI_PATCH_TARGET) as mock_openai:
        mock_openai.return_value = mock_openai_client
        preprocessor = ImagePreprocessor()
        
        result = preprocessor._load_image_as_base64(sample_image_bytes)
        
        assert isinstance(result, str)
        decoded = base64.b64decode(result)
        assert decoded == sample_image_bytes


@pytest.mark.unit
def test_load_image_from_url(monkeypatch, mock_openai_client):
    """Should return URL directly for http/https URLs."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    
    with patch(OPENAI_PATCH_TARGET) as mock_openai:
        mock_openai.return_value = mock_openai_client
        preprocessor = ImagePreprocessor()
        
        url = "https://example.com/image.png"
        result = preprocessor._load_image_as_base64(url)
        
        assert result == url


@pytest.mark.unit
def test_load_image_file_not_found(monkeypatch, mock_openai_client):
    """Should raise ValueError for non-existent file."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    
    with patch(OPENAI_PATCH_TARGET) as mock_openai:
        mock_openai.return_value = mock_openai_client
        preprocessor = ImagePreprocessor()
        
        with pytest.raises(ValueError, match="Image file not found"):
            preprocessor._load_image_as_base64("/nonexistent/path/image.png")


# -----------------------------------------------------------------------------
# Media Type Detection Tests
# -----------------------------------------------------------------------------


@pytest.mark.unit
def test_get_image_media_type_png(monkeypatch, mock_openai_client):
    """Should detect PNG media type."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    
    with patch(OPENAI_PATCH_TARGET) as mock_openai:
        mock_openai.return_value = mock_openai_client
        preprocessor = ImagePreprocessor()
        
        assert preprocessor._get_image_media_type("image.png") == "image/png"
        assert preprocessor._get_image_media_type("IMAGE.PNG") == "image/png"


@pytest.mark.unit
def test_get_image_media_type_jpeg(monkeypatch, mock_openai_client):
    """Should detect JPEG media type."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    
    with patch(OPENAI_PATCH_TARGET) as mock_openai:
        mock_openai.return_value = mock_openai_client
        preprocessor = ImagePreprocessor()
        
        assert preprocessor._get_image_media_type("image.jpg") == "image/jpeg"
        assert preprocessor._get_image_media_type("image.jpeg") == "image/jpeg"


@pytest.mark.unit
def test_get_image_media_type_default(monkeypatch, mock_openai_client):
    """Should default to jpeg for unknown types."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    
    with patch(OPENAI_PATCH_TARGET) as mock_openai:
        mock_openai.return_value = mock_openai_client
        preprocessor = ImagePreprocessor()
        
        assert preprocessor._get_image_media_type("image.unknown") == "image/jpeg"
        assert preprocessor._get_image_media_type(b"bytes") == "image/jpeg"


# -----------------------------------------------------------------------------
# Process Image Tests
# -----------------------------------------------------------------------------


@pytest.mark.unit
def test_process_image_success(monkeypatch, mock_openai_client, temp_image_file):
    """Should successfully process an image and return ImageQueryResult."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    
    with patch(OPENAI_PATCH_TARGET) as mock_openai:
        mock_openai.return_value = mock_openai_client
        preprocessor = ImagePreprocessor()
        
        result = preprocessor.process_image(temp_image_file, "What is this graph showing?")
        
        assert isinstance(result, ImageQueryResult)
        assert result.query == "quadratic functions and parabolas"
        assert result.detected_type == "graph"
        assert result.confidence == 0.9
        assert result.extracted_text == "f(x) = x^2"


@pytest.mark.unit
def test_process_image_without_prompt(monkeypatch, mock_openai_client, temp_image_file):
    """Should process image without user prompt."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    
    with patch(OPENAI_PATCH_TARGET) as mock_openai:
        mock_openai.return_value = mock_openai_client
        preprocessor = ImagePreprocessor()
        
        result = preprocessor.process_image(temp_image_file)
        
        assert isinstance(result, ImageQueryResult)
        assert result.query == "quadratic functions and parabolas"


@pytest.mark.unit
def test_process_image_with_url(monkeypatch, mock_openai_client):
    """Should process image from URL."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    
    with patch(OPENAI_PATCH_TARGET) as mock_openai:
        mock_openai.return_value = mock_openai_client
        preprocessor = ImagePreprocessor()
        
        result = preprocessor.process_image(
            "https://example.com/graph.png",
            "Help me understand this",
        )
        
        assert isinstance(result, ImageQueryResult)
        # Verify the API was called with URL format
        call_args = mock_openai_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        user_content = messages[1]["content"]
        assert any(
            item.get("type") == "image_url" and 
            item.get("image_url", {}).get("url") == "https://example.com/graph.png"
            for item in user_content
        )


@pytest.mark.unit
def test_process_image_api_failure(monkeypatch, mock_openai_client, temp_image_file):
    """Should raise RuntimeError on API failure."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    
    with patch(OPENAI_PATCH_TARGET) as mock_openai:
        mock_openai.return_value = mock_openai_client
        mock_openai_client.chat.completions.create.side_effect = Exception("API Error")
        
        preprocessor = ImagePreprocessor()
        
        with pytest.raises(RuntimeError, match="OpenAI API call failed"):
            preprocessor.process_image(temp_image_file)


# -----------------------------------------------------------------------------
# Response Parsing Tests
# -----------------------------------------------------------------------------


@pytest.mark.unit
def test_parse_response_valid_json(monkeypatch, mock_openai_client):
    """Should parse valid JSON response."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    
    with patch(OPENAI_PATCH_TARGET) as mock_openai:
        mock_openai.return_value = mock_openai_client
        preprocessor = ImagePreprocessor()
        
        response = json.dumps({
            "detected_type": "equation",
            "confidence": 0.85,
            "extracted_text": "\\int x^2 dx",
            "query": "integration of polynomial functions",
        })
        
        result = preprocessor._parse_response(response, None)
        
        assert result.detected_type == "equation"
        assert result.confidence == 0.85
        assert result.extracted_text == "\\int x^2 dx"
        assert result.query == "integration of polynomial functions"


@pytest.mark.unit
def test_parse_response_json_in_code_block(monkeypatch, mock_openai_client):
    """Should extract JSON from markdown code block."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    
    with patch(OPENAI_PATCH_TARGET) as mock_openai:
        mock_openai.return_value = mock_openai_client
        preprocessor = ImagePreprocessor()
        
        response = """Here's my analysis:
```json
{
    "detected_type": "diagram",
    "confidence": 0.75,
    "extracted_text": null,
    "query": "vector diagrams in physics"
}
```
"""
        
        result = preprocessor._parse_response(response, None)
        
        assert result.detected_type == "diagram"
        assert result.confidence == 0.75
        assert result.query == "vector diagrams in physics"


@pytest.mark.unit
def test_parse_response_invalid_json_fallback(monkeypatch, mock_openai_client):
    """Should fallback gracefully on invalid JSON."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    
    with patch(OPENAI_PATCH_TARGET) as mock_openai:
        mock_openai.return_value = mock_openai_client
        preprocessor = ImagePreprocessor()
        
        response = "This is not valid JSON, just a description of a graph."
        
        result = preprocessor._parse_response(response, "my original question")
        
        assert result.detected_type == "unknown"
        assert result.confidence == 0.3
        assert result.query == "my original question"
        assert result.raw_analysis == response


@pytest.mark.unit
def test_parse_response_clamps_confidence(monkeypatch, mock_openai_client):
    """Should clamp confidence to 0-1 range."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    
    with patch(OPENAI_PATCH_TARGET) as mock_openai:
        mock_openai.return_value = mock_openai_client
        preprocessor = ImagePreprocessor()
        
        # Test confidence > 1
        response = json.dumps({
            "detected_type": "graph",
            "confidence": 1.5,
            "query": "test query",
        })
        result = preprocessor._parse_response(response, None)
        assert result.confidence == 1.0
        
        # Test confidence < 0
        response = json.dumps({
            "detected_type": "graph",
            "confidence": -0.5,
            "query": "test query",
        })
        result = preprocessor._parse_response(response, None)
        assert result.confidence == 0.0


@pytest.mark.unit
def test_parse_response_unknown_type_fallback(monkeypatch, mock_openai_client):
    """Should fallback to 'unknown' for invalid detected_type."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    
    with patch(OPENAI_PATCH_TARGET) as mock_openai:
        mock_openai.return_value = mock_openai_client
        preprocessor = ImagePreprocessor()
        
        response = json.dumps({
            "detected_type": "invalid_type",
            "confidence": 0.8,
            "query": "test query",
        })
        
        result = preprocessor._parse_response(response, None)
        
        assert result.detected_type == "unknown"


# -----------------------------------------------------------------------------
# Convenience Function Tests
# -----------------------------------------------------------------------------


@pytest.mark.unit
def test_process_image_query_function(monkeypatch, mock_openai_client, temp_image_file):
    """process_image_query convenience function should return query string."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    
    with patch(OPENAI_PATCH_TARGET) as mock_openai:
        mock_openai.return_value = mock_openai_client
        
        query = process_image_query(temp_image_file, "What is this?")
        
        assert isinstance(query, str)
        assert query == "quadratic functions and parabolas"


@pytest.mark.unit
def test_process_image_query_custom_model(monkeypatch, mock_openai_client, temp_image_file):
    """process_image_query should accept custom model parameter."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    
    with patch(OPENAI_PATCH_TARGET) as mock_openai:
        mock_openai.return_value = mock_openai_client
        
        query = process_image_query(temp_image_file, model="gpt-4o-mini")
        
        assert isinstance(query, str)


# -----------------------------------------------------------------------------
# ImageQueryResult Tests
# -----------------------------------------------------------------------------


@pytest.mark.unit
def test_image_query_result_dataclass():
    """ImageQueryResult should be a proper dataclass."""
    result = ImageQueryResult(
        query="test query",
        detected_type="graph",
        confidence=0.9,
        extracted_text="y = mx + b",
        raw_analysis="full response",
    )
    
    assert result.query == "test query"
    assert result.detected_type == "graph"
    assert result.confidence == 0.9
    assert result.extracted_text == "y = mx + b"
    assert result.raw_analysis == "full response"


@pytest.mark.unit
def test_image_query_result_optional_fields():
    """ImageQueryResult should allow None for optional fields."""
    result = ImageQueryResult(
        query="test query",
        detected_type="unknown",
        confidence=0.5,
        extracted_text=None,
        raw_analysis=None,
    )
    
    assert result.extracted_text is None
    assert result.raw_analysis is None

