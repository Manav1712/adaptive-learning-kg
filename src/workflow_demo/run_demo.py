"""
CLI entrypoint that runs the Coach → Retriever → Tutor workflow end-to-end.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports when running directly
if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

try:
    from src.workflow_demo.coach import CoachAgent
    from src.workflow_demo.retriever import TeachingPackRetriever
except ImportError:
    # Fallback to relative imports when run as module
    from .coach import CoachAgent
    from .retriever import TeachingPackRetriever


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
        coach = CoachAgent(retriever=retriever)
        print("Coach initialized successfully!\n")
        return coach
    except Exception as e:
        print(f"Error initializing coach: {e}")
        import traceback
        traceback.print_exc()
        raise


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

