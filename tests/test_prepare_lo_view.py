"""
Tests for prepare_lo_view.py
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.experiments_manual.prepare_lo_view import (
    PrepareConfig,
    apply_sampling,
    build_outputs,
    collect_strings_recursively,
    extract_image_urls_from_text,
    generate_content_ids,
    load_config,
    normalize_row_text_and_images,
    strip_markdown_images,
    try_parse_json,
)


class TestPrepareLoView:
    """Test cases for prepare_lo_view functionality"""

    def test_load_config_defaults(self):
        """Test default config when no file exists"""
        config = load_config(None)
        assert config.concept_glob == "data/raw/concept*.csv"
        assert config.output_lo_index == "data/processed/lo_index.csv"
        assert config.sample_max_los is None

    def test_load_config_from_file(self, tmp_path):
        """Test loading config from YAML file"""
        config_file = tmp_path / "config.yaml"
        config_data = {
            "input_paths": {
                "concept": "custom/concept.csv",
                "example": "custom/example.csv",
            },
            "output_paths": {
                "lo_index": "custom/lo_index.csv",
                "content_items": "custom/content_items.csv",
            },
            "sample": {
                "max_los": 50,
                "max_content_per_lo": 3,
            },
        }
        config_file.write_text(json.dumps(config_data, indent=2))
        
        config = load_config(str(config_file))
        assert config.concept_glob == "custom/concept.csv"
        assert config.output_lo_index == "custom/lo_index.csv"
        assert config.sample_max_los == 50
        assert config.sample_max_content_per_lo == 3

    def test_extract_image_urls_from_text(self):
        """Test image URL extraction from various text formats"""
        # Markdown images
        text1 = "Here's an image ![alt text](https://example.com/image1.jpg) and another ![alt](https://example.com/image2.png)"
        urls1 = extract_image_urls_from_text(text1)
        assert "https://example.com/image1.jpg" in urls1
        assert "https://example.com/image2.png" in urls1
        assert len(urls1) == 2

        # Plain URLs
        text2 = "Check this out: https://example.com/photo.jpg and https://example.com/diagram.png"
        urls2 = extract_image_urls_from_text(text2)
        assert "https://example.com/photo.jpg" in urls2
        assert "https://example.com/diagram.png" in urls2

        # Mixed content
        text3 = "![graph](https://example.com/graph.jpg) and plain text https://example.com/chart.png"
        urls3 = extract_image_urls_from_text(text3)
        assert len(urls3) == 2
        assert all("https://example.com" in url for url in urls3)

        # No images
        text4 = "Just plain text with no images"
        urls4 = extract_image_urls_from_text(text4)
        assert urls4 == []

    def test_strip_markdown_images(self):
        """Test markdown image removal"""
        text = "Here's text ![alt](https://example.com/img.jpg) and more text"
        stripped = strip_markdown_images(text)
        assert "![alt](https://example.com/img.jpg)" not in stripped
        assert "Here's text" in stripped
        assert "and more text" in stripped

    def test_try_parse_json(self):
        """Test JSON parsing with fallback"""
        # Valid JSON
        valid_json = '{"key": "value", "nested": {"num": 42}}'
        parsed = try_parse_json(valid_json)
        assert parsed == {"key": "value", "nested": {"num": 42}}

        # Invalid JSON
        invalid_json = '{"key": "value", "unclosed": true'
        parsed = try_parse_json(invalid_json)
        assert parsed is None

        # Plain text
        plain_text = "Just some text"
        parsed = try_parse_json(plain_text)
        assert parsed is None

    def test_collect_strings_recursively(self):
        """Test recursive string collection from nested structures"""
        # Nested dict
        nested_dict = {
            "key1": "value1",
            "nested": {
                "key2": "value2",
                "list": ["item1", "item2"]
            },
            "list": [{"key3": "value3"}, "item4"]
        }
        strings = collect_strings_recursively(nested_dict)
        expected = ["value1", "value2", "item1", "item2", "value3", "item4"]
        assert all(s in strings for s in expected)

        # Simple string
        simple = "just a string"
        strings = collect_strings_recursively(simple)
        assert strings == ["just a string"]

    def test_normalize_row_text_and_images(self):
        """Test text normalization and image extraction"""
        # JSON content
        json_content = '{"problem": "Solve this", "solution": "![graph](https://example.com/graph.jpg) Answer: 42"}'
        text, urls = normalize_row_text_and_images(json_content)
        assert "Solve this" in text
        assert "Answer: 42" in text
        assert "https://example.com/graph.jpg" in urls

        # Plain text with images
        plain_content = "Here's a problem ![diagram](https://example.com/diagram.png) solve it"
        text, urls = normalize_row_text_and_images(plain_content)
        assert "Here's a problem" in text
        assert "solve it" in text
        assert "https://example.com/diagram.png" in urls

    def test_generate_content_ids(self):
        """Test content ID generation"""
        df = pd.DataFrame({
            "lo_id": ["100", "100", "100", "101", "101"],
            "content_type": ["concept", "concept", "example", "concept", "try_it"],
        })
        content_ids = generate_content_ids(df)
        expected_ids = ["100_concept_1", "100_concept_2", "100_example_1", "101_concept_1", "101_try_it_1"]
        assert list(content_ids) == expected_ids

    def test_apply_sampling(self):
        """Test sampling functionality"""
        df = pd.DataFrame({
            "lo_id": ["100", "100", "100", "101", "101", "102"],
            "content_type": ["concept", "example", "try_it", "concept", "example", "concept"],
            "raw_content": ["text1", "text2", "text3", "text4", "text5", "text6"]
        })

        # Test max_los sampling
        sampled_los = apply_sampling(df, max_los=2, max_content_per_lo=None)
        assert len(sampled_los["lo_id"].unique()) == 2
        assert "100" in sampled_los["lo_id"].values
        assert "101" in sampled_los["lo_id"].values

        # Test max_content_per_lo sampling
        sampled_content = apply_sampling(df, max_los=None, max_content_per_lo=1)
        # Should get 1 content per (lo_id, content_type) combination
        # LO 100: concept, example, try_it (3) + LO 101: concept, example (2) + LO 102: concept (1) = 6
        assert len(sampled_content) == 6

        # Test both
        sampled_both = apply_sampling(df, max_los=2, max_content_per_lo=1)
        assert len(sampled_both["lo_id"].unique()) == 2
        # LO 100: concept, example, try_it (3) + LO 101: concept, example (2) = 5
        assert len(sampled_both) == 5

    def test_build_outputs(self):
        """Test output building from unified dataframe"""
        df = pd.DataFrame({
            "lo_id": ["100", "100", "101"],
            "raw_content": [
                '{"problem": "Solve this", "![graph](https://example.com/graph.jpg)"}',
                '{"example": "Here is how", "![diagram](https://example.com/diagram.png)"}',
                '{"try_it": "Practice this"}'
            ],
            "type": ["concept", "example", "try_it"],
            "book": ["Calculus", "Calculus", "Calculus"],
            "learning_objective": ["LO1", "LO1", "LO2"],
            "unit": ["Unit1", "Unit1", "Unit2"],
            "chapter": ["Ch1", "Ch1", "Ch2"]
        })

        lo_index, content_items = build_outputs(df, PrepareConfig())

        # Check lo_index
        assert len(lo_index) == 2  # 2 unique LOs
        assert "100" in lo_index["lo_id"].values
        assert "101" in lo_index["lo_id"].values

        # Check content_items
        assert len(content_items) == 3
        assert "content_id" in content_items.columns
        assert "text" in content_items.columns
        assert "image_urls" in content_items.columns
        assert "lo_id_parent" in content_items.columns

        # Check content_id format
        content_ids = content_items["content_id"].tolist()
        assert "100_concept_1" in content_ids
        assert "100_example_1" in content_ids
        assert "101_try_it_1" in content_ids

    def test_integration_with_sample_data(self):
        """Test with actual sample data structure"""
        # Create sample data similar to what we expect from raw CSVs
        sample_data = pd.DataFrame({
            "lo_id": ["1867", "1867", "1868"],
            "raw_content": [
                '{"problem": "Solve trigonometric equation", "![unit circle](https://example.com/unit_circle.jpg)"}',
                '{"example": "Here is how to solve it", "![solution](https://example.com/solution.jpg)"}',
                '{"concept": "Basic trigonometry"}'
            ],
            "type": ["try_it", "example", "concept"],
            "book": ["Calculus", "Calculus", "Calculus"],
            "learning_objective": ["Solving Trigonometric Equations", "Solving Trigonometric Equations", "Basic Trigonometry"],
            "unit": ["Trigonometry", "Trigonometry", "Trigonometry"],
            "chapter": ["2", "2", "1"]
        })

        lo_index, content_items = build_outputs(sample_data, PrepareConfig())

        # Verify outputs
        assert len(lo_index) == 2  # 2 unique LOs
        assert len(content_items) == 3  # 3 content items

        # Check that image URLs were extracted
        image_urls_col = content_items["image_urls"].tolist()
        assert any("unit_circle.jpg" in urls for urls in image_urls_col)
        assert any("solution.jpg" in urls for urls in image_urls_col)

        # Check that text was cleaned
        text_col = content_items["text"].tolist()
        assert any("Solve trigonometric equation" in text for text in text_col)
        assert any("Basic trigonometry" in text for text in text_col)


if __name__ == "__main__":
    pytest.main([__file__])
