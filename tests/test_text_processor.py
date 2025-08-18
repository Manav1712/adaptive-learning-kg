"""
Test suite for StreamingTextProcessor

Tests the core functionality of the streaming text processor including
line cleaning, formatting fixes, and file processing capabilities.
"""
import os
import tempfile
import pytest
from unittest.mock import patch, mock_open
from src.processing.text_processor import StreamingTextProcessor


class TestStreamingTextProcessor:
    """Test cases for StreamingTextProcessor following SWE conventions"""
    
    @pytest.fixture(autouse=True)
    def setup_processor(self):
        """Fixture to set up fresh processor instance for each test"""
        self.processor = StreamingTextProcessor()
    
    def test_clean_line_normalizes_whitespace(self):
        """Test that _clean_line properly normalizes whitespace"""
        # Arrange
        input_line = "  hello   world  "
        expected_output = "hello world"
        
        # Act
        result = self.processor._clean_line(input_line)
        
        # Assert
        assert result == expected_output
    
    def test_clean_line_handles_empty_input(self):
        """Test that _clean_line handles empty and whitespace-only input"""
        # Test empty line
        assert self.processor._clean_line("   \t  \n") == ""
        
        # Test completely empty string
        assert self.processor._clean_line("") == ""
    
    def test_clean_line_preserves_single_words(self):
        """Test that _clean_line preserves single words without modification"""
        # Arrange
        input_line = "hello"
        expected_output = "hello"
        
        # Act
        result = self.processor._clean_line(input_line)
        
        # Assert
        assert result == expected_output
    
    @pytest.mark.parametrize("input_text,expected_output", [
        ("x ^ 2", "x^2"),
        ("a + b", "a+b"), 
        ("y - 3", "y-3"),
        ("z * 4", "z*4"),
    ])
    def test_fix_broken_math_notation(self, input_text, expected_output):
        """Test that _fix_broken_formatting fixes broken mathematical notation"""
        # Act
        result = self.processor._fix_broken_formatting(input_text)
        
        # Assert
        assert result == expected_output
    
    @pytest.mark.parametrize("input_text,expected_output", [
        ("3 . 14", "3.14"),
        ("0 . 5", "0.5"),
        ("123 . 456", "123.456"),
    ])
    def test_fix_broken_decimal_numbers(self, input_text, expected_output):
        """Test that _fix_broken_formatting fixes broken decimal numbers"""
        # Act
        result = self.processor._fix_broken_formatting(input_text)
        
        # Assert
        assert result == expected_output
    
    def test_fix_multiple_consecutive_dots(self):
        """Test that _fix_broken_formatting normalizes multiple consecutive dots"""
        # Arrange
        input_text = "end...."
        expected_output = "end..."
        
        # Act
        result = self.processor._fix_broken_formatting(input_text)
        
        # Assert
        assert result == expected_output
    
    def test_process_file_integration(self):
        """Integration test for complete file processing workflow"""
        # Arrange
        input_content = """Chapter 1 Functions


A function f consists of inputs.
A function f consists of inputs.

Example: f ( x ) = x ^ 2
   
The value 3 . 14 is important.

"""
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_input:
            temp_input.write(input_content)
            temp_input_path = temp_input.name
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_output:
            temp_output_path = temp_output.name
        
        try:
            # Act
            stats = self.processor.process_file(temp_input_path, temp_output_path)
            
            # Assert - Statistics
            assert stats['lines_processed'] > 0, "Should process at least one line"
            assert stats['lines_cleaned'] > 0, "Should output at least one clean line"
            assert stats['empty_lines_removed'] > 0, "Should remove empty lines"
            
            # Assert - Content verification
            with open(temp_output_path, 'r', encoding='utf-8') as f:
                output_content = f.read()
            
            assert "Chapter 1 Functions" in output_content, "Should preserve chapter title"
            assert "f ( x ) = x^2" in output_content, "Should fix mathematical notation"
            assert "3.14" in output_content, "Should fix decimal numbers"
            
            # Assert - No duplicate lines
            lines = [line.strip() for line in output_content.strip().split('\n') if line.strip()]
            unique_lines = list(set(lines))
            assert len(lines) == len(unique_lines), "Should remove duplicate lines"
            
        finally:
            # Cleanup
            if os.path.exists(temp_input_path):
                os.unlink(temp_input_path)
            if os.path.exists(temp_output_path):
                os.unlink(temp_output_path)
    
    def test_stats_reset_initializes_correctly(self):
        """Test that _reset_stats properly initializes statistics"""
        # Arrange - Set some values first
        self.processor.lines_processed = 100
        self.processor.lines_cleaned = 80
        
        # Act
        self.processor._reset_stats()
        
        # Assert
        stats = self.processor._get_stats()
        assert stats['lines_processed'] == 0
        assert stats['lines_cleaned'] == 0
        assert stats['empty_lines_removed'] == 0
        assert stats['reduction_percent'] == 100.0  # When lines_cleaned=0, reduction is 100%
    
    def test_stats_calculation_with_sample_data(self):
        """Test that statistics are calculated correctly with sample data"""
        # Arrange
        self.processor._reset_stats()
        self.processor.lines_processed = 100
        self.processor.lines_cleaned = 80
        self.processor.empty_lines_removed = 20
        
        # Act
        stats = self.processor._get_stats()
        
        # Assert
        assert stats['lines_processed'] == 100
        assert stats['lines_cleaned'] == 80
        assert stats['empty_lines_removed'] == 20
        assert stats['reduction_percent'] == 20.0  # (100-80)/100 * 100 = 20%
    
    def test_clean_line_with_special_characters(self):
        """Test that _clean_line handles special characters correctly"""
        # Arrange
        input_text = "!@#$%^&*()"
        expected_output = "!@#$%^&*()"
        
        # Act
        result = self.processor._clean_line(input_text)
        
        # Assert
        assert result == expected_output
    
    def test_clean_line_with_very_long_input(self):
        """Test that _clean_line handles very long input without crashing"""
        # Arrange
        long_line = "word " * 1000
        
        # Act
        result = self.processor._clean_line(long_line)
        
        # Assert
        assert isinstance(result, str), "Should return a string"
        assert len(result) < len(long_line), "Should compress multiple spaces"
        assert result.count("word") == 1000, "Should preserve all words but compress spaces"


class TestModuleImport:
    """Test module import functionality"""
    
    def test_streaming_text_processor_can_be_imported(self):
        """Test that StreamingTextProcessor can be imported and instantiated"""
        # Act
        from src.processing.text_processor import StreamingTextProcessor
        processor = StreamingTextProcessor()
        
        # Assert
        assert processor is not None
        assert isinstance(processor, StreamingTextProcessor)


if __name__ == "__main__":
    # Run tests with pytest if available, otherwise run simple test
    try:
        pytest.main([__file__])
    except ImportError:
        # Simple test runner if pytest not available
        test_instance = TestStreamingTextProcessor()
        test_instance.setup_method()
        
        print("Running basic tests...")
        test_instance.test_clean_line_basic()
        test_instance.test_fix_broken_formatting()
        test_instance.test_stats_tracking()
        test_instance.test_edge_cases()
        test_import()
        print("All basic tests passed!")
