
#!/usr/bin/env python3
"""Run a single explanation through the Task 2 evaluator."""

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "task2"))

from src.evaluator import load_knowledge, evaluate_one


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lesson-id", required=True, choices=["L1", "L2", "L3"])
    parser.add_argument("--explanation", required=True)
    parser.add_argument("--explanation-id", default="DEMO_001")
    args = parser.parse_args()

    row = {
        "lesson_id": args.lesson_id,
        "explanation_text": args.explanation,
        "explanation_id": args.explanation_id,
    }

    result = evaluate_one(row, load_knowledge())
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
