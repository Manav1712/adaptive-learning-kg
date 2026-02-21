"""
CLI entrypoint that runs the Coach → Retriever → Tutor workflow end-to-end.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Optional

# Default session memory file (in project root by default)
SESSION_MEMORY_FILE = os.path.join(
    Path(__file__).resolve().parents[2], "session_memory.json"
)

# Add parent directory to path for imports when running directly
if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

try:
    from src.workflow_demo.coach import CoachAgent
    from src.workflow_demo.retriever import TeachingPackRetriever
    from src.workflow_demo.demo_profiles import get_active_profile, get_profile_name
except ImportError:
    # Fallback to relative imports when run as module
    from .coach import CoachAgent
    from .retriever import TeachingPackRetriever
    from .demo_profiles import get_active_profile, get_profile_name


def build_demo_coach() -> CoachAgent:
    """
    Convenience helper that wires the retriever and coach together.

    Inputs:
        None.

    Outputs:
        CoachAgent instance ready for conversation.
    """

    print("Initializing retriever (this may take a moment on first run)...")
    try:
        retriever = TeachingPackRetriever()
        print("Retriever initialized successfully!")
    except Exception as e:
        print(f"Error initializing retriever: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print("Initializing coach...")
    try:
        coach = CoachAgent(retriever=retriever, session_memory_path=SESSION_MEMORY_FILE)
        
        # Load the active demo profile
        profile = get_active_profile()
        profile_name = get_profile_name()
        coach.student_profile.update(profile)
        coach.session_memory.student_profile.update(profile)
        coach.session_memory.save()
        
        print(f"Coach initialized successfully!")
        print(f"  Session memory: {SESSION_MEMORY_FILE}")
        print(f"  Active profile: {profile_name}")
        print(f"  (To switch profiles, edit ACTIVE_PROFILE in demo_profiles.py)\n")
        return coach
    except Exception as e:
        print(f"Error initializing coach: {e}")
        import traceback
        traceback.print_exc()
        raise


def _looks_like_image_url(text: str) -> bool:
    return bool(re.match(r"^https?://.+\.(png|jpe?g|gif|webp)$", text, re.IGNORECASE))


def _looks_like_image_path(text: str) -> bool:
    """Check if text looks like an image file path and exists."""
    if not text:
        return False
    
    # Check if it has an image extension
    path = Path(text)
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
        return False
    
    # Try as-is (absolute or relative to current dir)
    if path.exists():
        return True
    
    # Try relative to project root (where this script is located)
    project_root = Path(__file__).resolve().parents[2]
    project_relative_path = project_root / text
    if project_relative_path.exists():
        return True
    
    return False


def detect_image_input(user_input: str) -> tuple[Optional[str], str]:
    """
    Detect if the user's message references an image (file path or URL).

    Returns:
        (image_ref, remaining_text) - image_ref is resolved to absolute path if it's a file path
    """
    tokens = user_input.split()
    image_ref: Optional[str] = None
    matched_token: Optional[str] = None
    
    for token in tokens:
        if _looks_like_image_url(token):
            image_ref = token
            matched_token = token
            break
        elif _looks_like_image_path(token):
            # Resolve to absolute path for file paths
            path = Path(token)
            if path.exists():
                image_ref = str(path.resolve())
                matched_token = token
            else:
                # Try relative to project root
                project_root = Path(__file__).resolve().parents[2]
                project_relative_path = project_root / token
                if project_relative_path.exists():
                    image_ref = str(project_relative_path.resolve())
                    matched_token = token
            if image_ref:
                break

    if not image_ref:
        return None, user_input

    # Remove the matched token from remaining text
    remaining_tokens = [tok for tok in tokens if tok != matched_token]
    remaining_text = " ".join(remaining_tokens).strip()
    return image_ref, remaining_text


def interactive_loop(coach: CoachAgent) -> None:
    """
    Simple REPL loop mirroring the notebook UX.

    Inputs:
        coach: CoachAgent to handle each user turn.

    Outputs:
        None. Prints assistant responses until the user exits.
    """

    import sys

    print("Adaptive Learning Coach Demo\nType 'quit' to exit.\n")
    sys.stdout.flush()

    greeting = coach.initial_greeting()
    if greeting:
        print(f"Assistant: {greeting}\n")
        sys.stdout.flush()
    
    while True:
        try:
            sys.stdout.write("You: ")
            sys.stdout.flush()
            user_text = input().strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEnding demo. Goodbye!")
            break

        if not user_text:
            continue
        if user_text.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        try:
            image_ref, text_only = detect_image_input(user_text)
            if image_ref:
                print(f"[DEBUG] Image detected in run_demo: {image_ref}")
            if image_ref:
                response = coach.process_multimodal_turn(text_only, image_ref)
            else:
                response = coach.process_turn(user_text)
            if response:
                print(f"Assistant: {response}\n")
                sys.stdout.flush()
        except Exception as e:
            print(f"Error processing turn: {e}")
            import traceback

            traceback.print_exc()
            sys.stdout.flush()


def main() -> None:
    """
    Parse CLI args and start the interactive demo loop.
    """

    parser = argparse.ArgumentParser(description="Run the workflow_demo CLI.")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch an interactive shell (default behavior).",
    )
    args = parser.parse_args()
    coach = build_demo_coach()

    interactive_loop(coach)


if __name__ == "__main__":
    main()

