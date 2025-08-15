#!/usr/bin/env python3
"""
Script to clean the Calculus Volume 1 text file
"""
from src.processing.text_cleaner import OpenStaxTextCleaner

def main():
    # Read the Calculus text file
    with open('Calculus_Volume_1.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Original text length: {len(text)} characters")
    print(f"Original text lines: {len(text.splitlines())}")
    
    # Clean the text
    cleaner = OpenStaxTextCleaner()
    cleaned_text = cleaner.clean_text(text)
    
    print(f"Cleaned text length: {len(cleaned_text)} characters")
    print(f"Cleaned text lines: {len(cleaned_text.splitlines())}")
    
    # Save cleaned text only
    with open('Calculus_Volume_1_cleaned.txt', 'w', encoding='utf-8') as f:
        f.write(cleaned_text)
    
    print(f"\nCleaned text saved to: Calculus_Volume_1_cleaned.txt")

if __name__ == "__main__":
    main()
