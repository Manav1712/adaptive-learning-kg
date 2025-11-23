"""
CLI entrypoint that runs the Coach → Retriever → Tutor workflow end-to-end.
"""

from __future__ import annotations

import argparse
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

    retriever = TeachingPackRetriever()
    return CoachAgent(retriever=retriever)


def interactive_loop(coach: CoachAgent) -> None:
    """
    Simple REPL loop mirroring the notebook UX.

    Inputs:
        coach: CoachAgent to handle each user turn.

    Outputs:
        None. Prints assistant responses until the user exits.
    """

    print("Adaptive Learning Coach Demo\nType 'quit' to exit.\n")
    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEnding demo. Goodbye!")
            break

        if not user_text:
            continue
        if user_text.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        response = coach.process_turn(user_text)
        print(f"Coach: {response}\n")


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

