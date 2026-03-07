"""
Shared JSON parsing utilities for LLM responses.

Provides robust JSON parsing that handles common LLM response issues:
- Markdown code blocks
- Invalid escape sequences
- Invalid hex sequences (\\x, \\u)
- Fallback extraction
"""

import json
from typing import Any, Dict


def coerce_json(raw: str) -> Dict[str, Any]:
    """
    Parse JSON from LLM response, handling markdown code blocks and invalid escape sequences.
    
    This function attempts multiple strategies to parse JSON from LLM responses:
    1. Direct parsing
    2. Markdown code block extraction
    3. Escape sequence repair
    4. JSON boundary extraction
    
    Args:
        raw: Raw string response from LLM (may contain markdown or invalid escapes).
        
    Returns:
        Parsed JSON dictionary.
        
    Raises:
        json.JSONDecodeError: If JSON cannot be parsed even after repair attempts.
    """
    raw = raw.strip()
    
    # Extract JSON from markdown code blocks
    if raw.startswith("```"):
        lines = raw.split("\n")
        if len(lines) > 1:
            start_idx = 1
            end_idx = len(lines)
            for i, line in enumerate(lines[1:], start=1):
                if line.strip() == "```":
                    end_idx = i
                    break
            raw = "\n".join(lines[start_idx:end_idx])
    
    # Try parsing directly first
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        # Fix all invalid escape sequences by escaping lone backslashes
        fixed = _fix_invalid_escapes(raw)
        
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            # Last resort: try to find JSON object boundaries
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(raw[start:end+1])
                except json.JSONDecodeError:
                    pass
            
            # If all else fails, raise the original error
            raise e


def _fix_invalid_escapes(s: str) -> str:
    r"""
    Fix invalid escape sequences in a string by doubling lone backslashes.
    
    Valid JSON escapes are: \" \\ \/ \b \f \n \r \t \uXXXX
    Everything else (like \s \c \x without proper hex) needs escaping.
    """
    result = []
    i = 0
    n = len(s)

    while i < n:
        if s[i] != "\\":
            result.append(s[i])
            i += 1
            continue

        # A trailing backslash is invalid JSON, so escape it.
        if i == n - 1:
            result.append("\\\\")
            i += 1
            continue

        next_char = s[i + 1]

        # Keep standard JSON escapes, including escaped backslashes, as-is.
        if next_char in '"\\/bfnrt':
            result.append("\\" + next_char)
            i += 2
            continue

        # Keep valid unicode escapes as-is.
        if next_char == "u" and i + 5 < n:
            hex_part = s[i + 2:i + 6]
            if all(c in "0123456789abcdefABCDEF" for c in hex_part):
                result.append(s[i:i + 6])
                i += 6
                continue

        # Any other backslash needs to be escaped before JSON parsing.
        result.append("\\\\" + next_char)
        i += 2

    return ''.join(result)
