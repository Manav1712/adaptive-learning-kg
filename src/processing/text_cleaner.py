"""
Text cleaner for OpenStax content
"""
import re
from typing import List, Tuple

class OpenStaxTextCleaner:
    """Cleans OpenStax text files by removing garbage and formatting artifacts"""
    
    def __init__(self):
        # Patterns to remove
        self.remove_patterns = [
            # Page numbers and headers
            r'^Chapter \d+ \| .*\n',
            r'^\d+\n',
            r'^Chapter \d+\n',
            
            # Figure references
            r'Figure \d+\.\d+[^.]*\n',
            
            # URLs and links
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            r'Visit this applet link[^.]*\.',
            
            # Extra whitespace
            r'\n\s*\n\s*\n',  # Multiple empty lines
            r'^\s+$',  # Lines with only whitespace
            
            # OpenStax footer
            r'This OpenStax book is available for free at[^\n]*\n',
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = [re.compile(pattern, re.MULTILINE) for pattern in self.remove_patterns]
    
    def clean_text(self, text: str) -> str:
        """Clean the text by removing garbage patterns"""
        cleaned = text
        
        # Remove all garbage patterns
        for pattern in self.compiled_patterns:
            cleaned = pattern.sub('', cleaned)
        
        # Additional cleaning steps
        cleaned = self._clean_whitespace(cleaned)
        cleaned = self._clean_math_notation(cleaned)
        
        return cleaned
    
    def _clean_whitespace(self, text: str) -> str:
        """Clean up whitespace issues"""
        # Replace multiple newlines with double newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove leading/trailing whitespace from lines
        lines = [line.strip() for line in text.split('\n')]
        
        # Remove empty lines at start and end
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        
        return '\n'.join(lines)
    
    def _clean_math_notation(self, text: str) -> str:
        """Clean up mathematical notation artifacts"""
        # Fix broken set notation
        text = re.sub(r'⎧\s*\n\s*⎨\s*\n\s*⎩', '{', text)
        text = re.sub(r'⎫\s*\n\s*⎬\s*\n\s*⎭', '}', text)
        
        # Fix broken piecewise functions
        text = re.sub(r'⎧([^⎫]*)⎫', r'{\1}', text)
        
        return text
    
    def extract_sections(self, text: str) -> List[Tuple[str, str]]:
        """Extract clean sections from the text"""
        # Split by chapter headers
        chapter_pattern = r'(\d+\.\d+\s+[^\n]+)'
        sections = re.split(chapter_pattern, text)
        
        # Group sections with their content
        section_data = []
        for i in range(1, len(sections), 2):
            if i + 1 < len(sections):
                section_title = sections[i].strip()
                section_content = sections[i + 1].strip()
                if section_title and section_content:
                    section_data.append((section_title, section_content))
        
        return section_data
    
    def clean_and_extract(self, text: str) -> List[Tuple[str, str]]:
        """Clean text and extract sections"""
        cleaned_text = self.clean_text(text)
        return self.extract_sections(cleaned_text)
