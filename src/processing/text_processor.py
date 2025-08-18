"""
Streaming text processor for large OpenStax files (50K+ lines)
Performs minimal cleaning to fix broken formatting and prepare for chunking experiments
"""
import re
import os
from typing import Iterator, Optional


class StreamingTextProcessor:
    """
    Processes large text files line-by-line to avoid memory issues.
    Performs minimal cleaning focused on fixing formatting issues.
    """
    
    def __init__(self):
        self.lines_processed = 0
        self.lines_cleaned = 0
        self.empty_lines_removed = 0
        
    def process_file(self, input_path: str, output_path: str, 
                    progress_interval: int = 1000) -> dict:
        """
        Process a large text file with minimal cleaning.
        
        Args:
            input_path: Path to input file
            output_path: Path to output file
            progress_interval: Print progress every N lines
            
        Returns:
            dict: Processing statistics
        """
        self._reset_stats()
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"Processing {input_path} -> {output_path}")
        
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            previous_line = ""
            
            for line_num, line in enumerate(infile, 1):
                self.lines_processed += 1
                
                # Clean the current line
                cleaned_line = self._clean_line(line)
                
                # Skip if line becomes empty after cleaning
                if not cleaned_line.strip():
                    self.empty_lines_removed += 1
                    continue
                
                # Skip consecutive duplicate lines (common formatting issue)
                if cleaned_line.strip() == previous_line.strip():
                    continue
                
                # Write cleaned line
                outfile.write(cleaned_line + '\n')
                previous_line = cleaned_line
                self.lines_cleaned += 1
                
                # Progress reporting
                if line_num % progress_interval == 0:
                    print(f"  Processed {line_num:,} lines...")
        
        stats = self._get_stats()
        print(f"✅ Processing complete: {stats}")
        return stats
    
    def _clean_line(self, line: str) -> str:
        """
        Clean a single line with minimal processing.
        
        Args:
            line: Raw line from file
            
        Returns:
            str: Cleaned line
        """
        # Remove leading/trailing whitespace
        line = line.strip()
        
        # Normalize whitespace (multiple spaces/tabs become single space)
        line = re.sub(r'\s+', ' ', line)
        
        # Fix common broken formatting patterns
        line = self._fix_broken_formatting(line)
        
        return line
    
    def _fix_broken_formatting(self, line: str) -> str:
        """
        Fix common formatting issues in OpenStax documents.
        
        Args:
            line: Line to fix
            
        Returns:
            str: Line with formatting fixes
        """
        # Fix broken mathematical notation (spaces in formulas)
        line = re.sub(r'(\w)\s+(\^|\+|\-|\*|\/)\s+(\w)', r'\1\2\3', line)
        
        # Fix broken decimal numbers (e.g., "3 . 14" -> "3.14")
        line = re.sub(r'(\d)\s+\.\s+(\d)', r'\1.\2', line)
        
        # Fix broken parentheses (spaces before/after)
        line = re.sub(r'\s+\(', ' (', line)
        line = re.sub(r'\)\s+', ') ', line)
        
        # Fix multiple consecutive periods/dots
        line = re.sub(r'\.{3,}', '...', line)
        
        # Remove trailing periods on section headers (common OCR artifact)
        if re.match(r'^\d+\.\d+\s+[A-Z]', line):
            line = re.sub(r'\.$', '', line)
        
        return line
    
    def _reset_stats(self):
        """Reset processing statistics."""
        self.lines_processed = 0
        self.lines_cleaned = 0
        self.empty_lines_removed = 0
    
    def _get_stats(self) -> dict:
        """Get processing statistics."""
        return {
            'lines_processed': self.lines_processed,
            'lines_cleaned': self.lines_cleaned,
            'empty_lines_removed': self.empty_lines_removed,
            'reduction_percent': round(
                (1 - self.lines_cleaned / max(self.lines_processed, 1)) * 100, 2
            )
        }


def process_calculus_files():
    """
    Convenience function to process common Calculus files.
    Add your specific file paths here.
    """
    processor = StreamingTextProcessor()
    
    # Example file processing
    files_to_process = [
        {
            'input': 'Calculus_Volume_1_cleaned.txt',
            'output': 'data/processed/calculus_v1_minimal_clean.txt'
        }
        # Add more files as needed
    ]
    
    results = {}
    for file_config in files_to_process:
        if os.path.exists(file_config['input']):
            stats = processor.process_file(
                file_config['input'], 
                file_config['output']
            )
            results[file_config['input']] = stats
        else:
            print(f"❌ File not found: {file_config['input']}")
    
    return results


if __name__ == "__main__":
    # Run processing when script is executed directly
    results = process_calculus_files()
    
    print("\n📊 Processing Summary:")
    for filename, stats in results.items():
        print(f"  {filename}: {stats['lines_cleaned']:,} lines "
              f"({stats['reduction_percent']}% reduction)")
