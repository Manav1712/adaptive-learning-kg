"""
Tests for discover_content_links.py
"""

import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.experiments_manual.discover_content_links import (
    DiscoveryConfig,
    build_lo_metadata,
    generate_candidates_for_row,
    load_config,
    relation_for_content_type,
    tokenize,
    write_candidates,
)


class TestDiscoverContentLinks:
    """Test cases for content→LO candidate generation"""

    def test_load_config_defaults(self):
        """
        Tests default configuration loading.
        
        Args:
            None (uses default config)
            
        Returns:
            DiscoveryConfig with default values
            
        Behavior:
            - Returns defaults when no config file provided
            - Uses expected default paths and settings
        """
        config = load_config(None)
        assert config.input_lo_index == "data/processed/lo_index.csv"
        assert config.restrict_same_unit is True
        assert config.lexical_top_k == 5
        assert config.relation_concept == "explained_by"

    def test_load_config_from_file(self, tmp_path):
        """
        Tests configuration loading from YAML file.
        
        Args:
            tmp_path: Pytest temporary directory fixture
            
        Returns:
            None (asserts config values match file)
            
        Behavior:
            - Reads YAML config and maps to DiscoveryConfig fields
            - Handles nested config structure with pruning/relations
        """
        config_file = tmp_path / "config.yaml"
        config_data = {
            "output_paths": {
                "lo_index": "custom/lo_index.csv",
                "content_items": "custom/content_items.csv",
            },
            "output_candidates": "custom/candidates.csv",
            "pruning": {
                "same_unit": False,
                "same_chapter": True,
                "lexical_top_k": 10,
                "lexical_min_overlap": 2,
            },
            "relations": {
                "concept": "custom_explained_by",
                "example": "custom_exemplified_by",
                "try_it": "custom_practiced_by",
            },
        }
        config_file.write_text(json.dumps(config_data, indent=2))
        
        config = load_config(str(config_file))
        assert config.input_lo_index == "custom/lo_index.csv"
        assert config.output_candidates == "custom/candidates.csv"
        assert config.restrict_same_unit is False
        assert config.restrict_same_chapter is True
        assert config.lexical_top_k == 10
        assert config.lexical_min_overlap == 2
        assert config.relation_concept == "custom_explained_by"

    def test_tokenize(self):
        """
        Tests text tokenization into keywords.
        
        Args:
            Various text inputs
            
        Returns:
            List of lowercase alphanumeric tokens
            
        Behavior:
            - Extracts alphanumeric words
            - Converts to lowercase
            - Handles empty/None inputs gracefully
        """
        # Normal text
        tokens1 = tokenize("Solving Trigonometric Equations with Identities")
        expected1 = ["solving", "trigonometric", "equations", "with", "identities"]
        assert tokens1 == expected1

        # Text with punctuation and numbers
        tokens2 = tokenize("Chapter 2.3: Advanced Topics (Part A)")
        expected2 = ["chapter", "advanced", "topics", "part"]
        assert tokens2 == expected2

        # Empty/None inputs
        assert tokenize("") == []
        assert tokenize(None) == []
        assert tokenize("   ") == []

        # Special characters only
        assert tokenize("!@#$%^&*()") == []

    def test_relation_for_content_type(self):
        """
        Tests content type to relation mapping.
        
        Args:
            content_type: String content type
            config: DiscoveryConfig with relation mappings
            
        Returns:
            Relation string based on content type
            
        Behavior:
            - Maps concept/example/try_it to configured relations
            - Handles case variations and unknown types
        """
        config = DiscoveryConfig()
        
        assert relation_for_content_type("concept", config) == "explained_by"
        assert relation_for_content_type("example", config) == "exemplified_by"
        assert relation_for_content_type("try_it", config) == "practiced_by"
        
        # Case variations
        assert relation_for_content_type("CONCEPT", config) == "explained_by"
        assert relation_for_content_type("Example", config) == "exemplified_by"
        
        # Unknown types default to try_it relation
        assert relation_for_content_type("unknown", config) == "practiced_by"
        assert relation_for_content_type("", config) == "practiced_by"

    def test_build_lo_metadata(self):
        """
        Tests LO metadata preparation with tokenization.
        
        Args:
            lo_df: DataFrame with LO data
            
        Returns:
            DataFrame with added lo_tokens column
            
        Behavior:
            - Adds tokenized learning objectives as lo_tokens
            - Handles various text formats in learning_objective column
        """
        lo_df = pd.DataFrame({
            "lo_id": ["100", "101", "102"],
            "learning_objective": [
                "Solving Linear Equations",
                "Graphing Functions",
                "Advanced Calculus Topics"
            ],
            "unit": ["Algebra", "Functions", "Calculus"],
            "chapter": ["1", "2", "3"],
            "book": ["Math", "Math", "Math"]
        })
        
        lo_meta = build_lo_metadata(lo_df)
        
        assert "lo_tokens" in lo_meta.columns
        assert lo_meta.loc[0, "lo_tokens"] == ["solving", "linear", "equations"]
        assert lo_meta.loc[1, "lo_tokens"] == ["graphing", "functions"]
        assert lo_meta.loc[2, "lo_tokens"] == ["advanced", "calculus", "topics"]

    def test_generate_candidates_for_row_same_unit(self):
        """
        Tests candidate generation with same-unit restriction.
        
        Args:
            content_row: Content item data
            lo_meta: LO metadata with tokens
            config: Config with same_unit=True
            
        Returns:
            List of (lo_id, reason) tuples
            
        Behavior:
            - Only includes LOs from same unit as content's parent
            - Adds lexical matches if configured
        """
        lo_meta = pd.DataFrame({
            "lo_id": ["100", "101", "102", "103"],
            "learning_objective": ["Linear Equations", "Quadratic Functions", "Trigonometry", "Linear Systems"],
            "unit": ["Algebra", "Algebra", "Trigonometry", "Algebra"],
            "chapter": ["1", "1", "2", "1"],
            "lo_tokens": [
                ["linear", "equations"],
                ["quadratic", "functions"],
                ["trigonometry"],
                ["linear", "systems"]
            ]
        })
        
        content_row = pd.Series({
            "content_id": "100_concept_1",
            "content_type": "concept",
            "unit": "Algebra",
            "chapter": "1",
            "text": "Linear equations are fundamental to algebra"
        })
        
        config = DiscoveryConfig(restrict_same_unit=True, lexical_top_k=3)
        candidates = generate_candidates_for_row(content_row, lo_meta, config)
        
        # Should get same-unit LOs (100, 101, 103) plus lexical matches
        lo_ids = [c[0] for c in candidates]
        assert "100" in lo_ids  # same unit
        assert "101" in lo_ids  # same unit
        assert "103" in lo_ids  # same unit
        assert "102" not in lo_ids  # different unit, but might appear as lexical

    def test_generate_candidates_for_row_lexical(self):
        """
        Tests lexical candidate generation based on keyword overlap.
        
        Args:
            content_row: Content with specific keywords
            lo_meta: LOs with overlapping keywords
            config: Config with lexical matching enabled
            
        Returns:
            List including lexical matches with high overlap
            
        Behavior:
            - Finds LOs with keyword overlap above threshold
            - Ranks by overlap count and takes top-K
        """
        lo_meta = pd.DataFrame({
            "lo_id": ["100", "101", "102"],
            "learning_objective": ["Trigonometric Functions", "Linear Algebra", "Solving Equations"],
            "unit": ["Math", "Math", "Math"],
            "chapter": ["1", "2", "3"],
            "lo_tokens": [
                ["trigonometric", "functions"],
                ["linear", "algebra"],
                ["solving", "equations"]
            ]
        })
        
        content_row = pd.Series({
            "content_id": "test_content",
            "content_type": "concept",
            "unit": "Math",
            "chapter": "1",
            "text": "Solving trigonometric equations requires understanding functions"
        })
        
        config = DiscoveryConfig(
            restrict_same_unit=False,
            lexical_top_k=2,
            lexical_min_overlap=1
        )
        candidates = generate_candidates_for_row(content_row, lo_meta, config)
        
        # Should prioritize LO 100 (trigonometric, functions) and LO 102 (solving, equations)
        lo_ids = [c[0] for c in candidates]
        assert "100" in lo_ids  # trigonometric + functions overlap
        assert "102" in lo_ids  # solving + equations overlap

    def test_write_candidates(self):
        """
        Tests full candidate writing process.
        
        Args:
            content_df: DataFrame of content items
            lo_meta: LO metadata
            config: Discovery configuration
            
        Returns:
            DataFrame of candidate edges
            
        Behavior:
            - Generates candidates for all content items
            - Writes CSV with proper columns and relations
            - Returns the written DataFrame
        """
        content_df = pd.DataFrame({
            "content_id": ["100_concept_1", "101_example_1"],
            "content_type": ["concept", "example"],
            "unit": ["Algebra", "Algebra"],
            "chapter": ["1", "1"],
            "text": ["Linear equations", "Example of quadratic function"]
        })
        
        lo_meta = pd.DataFrame({
            "lo_id": ["100", "101"],
            "learning_objective": ["Linear Equations", "Quadratic Functions"],
            "unit": ["Algebra", "Algebra"],
            "chapter": ["1", "1"],
            "lo_tokens": [["linear", "equations"], ["quadratic", "functions"]]
        })
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "candidates.csv")
            config = DiscoveryConfig(
                output_candidates=output_path,
                restrict_same_unit=True,
                lexical_top_k=0  # disable lexical for simpler test
            )
            
            result_df = write_candidates(content_df, lo_meta, config)
            
            # Check output exists and has correct structure
            assert os.path.exists(output_path)
            assert len(result_df) > 0
            assert "source_lo_id" in result_df.columns
            assert "target_content_id" in result_df.columns
            assert "proposed_relation" in result_df.columns
            assert "reason" in result_df.columns
            
            # Check relations are correct
            concept_rows = result_df[result_df["target_content_id"] == "100_concept_1"]
            example_rows = result_df[result_df["target_content_id"] == "101_example_1"]
            
            assert all(concept_rows["proposed_relation"] == "explained_by")
            assert all(example_rows["proposed_relation"] == "exemplified_by")

    def test_integration_with_sample_data(self):
        """
        Tests end-to-end candidate generation with realistic data.
        
        Args:
            Sample data mimicking real LO and content structure
            
        Returns:
            None (verifies candidate generation works correctly)
            
        Behavior:
            - Tests with multiple content types and units
            - Verifies candidate filtering and relation assignment
            - Checks that reasonable number of candidates generated
        """
        # Create sample data similar to what we'd get from prepare step
        lo_df = pd.DataFrame({
            "lo_id": ["1867", "1868", "1869"],
            "learning_objective": [
                "Solving Trigonometric Equations",
                "Trigonometric Identities",
                "Linear Algebra Basics"
            ],
            "unit": ["Trigonometry", "Trigonometry", "Algebra"],
            "chapter": ["2", "2", "1"],
            "book": ["Precalculus", "Precalculus", "Algebra"]
        })
        
        content_df = pd.DataFrame({
            "content_id": ["1867_concept_1", "1867_example_1", "1869_try_it_1"],
            "content_type": ["concept", "example", "try_it"],
            "lo_id_parent": ["1867", "1867", "1869"],
            "text": [
                "Trigonometric equations require solving for unknown angles",
                "Example: solve sin(x) = 0.5 for x",
                "Practice: solve 2x + 3 = 7"
            ],
            "unit": ["Trigonometry", "Trigonometry", "Algebra"],
            "chapter": ["2", "2", "1"],
            "image_urls": ["[]", "[]", "[]"]
        })
        
        lo_meta = build_lo_metadata(lo_df)
        config = DiscoveryConfig(restrict_same_unit=True, lexical_top_k=2)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "test_candidates.csv")
            config.output_candidates = output_path
            
            result_df = write_candidates(content_df, lo_meta, config)
            
            # Verify reasonable number of candidates
            assert len(result_df) >= 3  # At least one per content item
            
            # Verify trigonometry content gets trigonometry LOs
            trig_candidates = result_df[
                result_df["target_content_id"].isin(["1867_concept_1", "1867_example_1"])
            ]
            trig_lo_ids = set(trig_candidates["source_lo_id"])
            assert "1867" in trig_lo_ids or "1868" in trig_lo_ids
            
            # Verify relation types
            concept_relations = result_df[
                result_df["target_content_id"] == "1867_concept_1"
            ]["proposed_relation"].unique()
            assert "explained_by" in concept_relations
            
            example_relations = result_df[
                result_df["target_content_id"] == "1867_example_1"
            ]["proposed_relation"].unique()
            assert "exemplified_by" in example_relations
            
            try_it_relations = result_df[
                result_df["target_content_id"] == "1869_try_it_1"
            ]["proposed_relation"].unique()
            assert "practiced_by" in try_it_relations


if __name__ == "__main__":
    pytest.main([__file__])
