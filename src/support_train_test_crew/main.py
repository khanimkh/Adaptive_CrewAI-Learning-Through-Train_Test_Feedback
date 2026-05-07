import argparse
import json

from .train_and_test import run_train_test_cycle


def main() -> None:
    """
    CLI entry point for running the full train/test cycle from the terminal.

    Parses command-line arguments, runs the train/test cycle via
    run_train_test_cycle(), and prints the result as formatted JSON.

    CLI Arguments
    -------------
    --iterations : int, default 1
        Number of train/test iterations to run. Each iteration executes one
        full crew cycle and prompts for feedback once per task (4 prompts total).

    --model : str, default "gpt-4o-mini"
        OpenAI model name passed to crew.test() for evaluation scoring.
        Example: "gpt-4o", "gpt-4o-mini".

    --sample-size : int, default 5
        Number of support ticket rows to include in the analysis input.
        Higher values give the agents more context but cost more tokens.

    --feedback : str, default ""
        Optional pre-written feedback text injected into task prompts before
        training begins. Used when --no-prompt-feedback is set.

    --no-prompt-feedback : flag
        If set, skips the interactive terminal feedback prompt and uses only
        the text provided via --feedback. Useful for scripted/CI runs.

    Output
    ------
    Prints a JSON object to stdout with keys:
        before_test_status, train_status, after_test_status, kickoff_status,
        user_feedback, report_path, training_file_path, charts, final_output,
        before_test_output, after_test_output, train_output.
    """
    parser = argparse.ArgumentParser(description="Run support CrewAI train and test cycle")
    parser.add_argument("--iterations", type=int, default=1, help="Number of train/test iterations")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model name for crew.test")
    parser.add_argument("--sample-size", type=int, default=5, help="How many sample rows to frame in inputs")
    parser.add_argument("--feedback", type=str, default="", help="Optional feedback text used during training")
    parser.add_argument(
        "--no-prompt-feedback",
        action="store_true",
        help="Skip terminal feedback prompt and use only --feedback text",
    )
    args = parser.parse_args()

    result = run_train_test_cycle(
        iterations=args.iterations,
        model=args.model,
        sample_size=args.sample_size,
        terminal_feedback=args.feedback,
        prompt_feedback=not args.no_prompt_feedback,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
