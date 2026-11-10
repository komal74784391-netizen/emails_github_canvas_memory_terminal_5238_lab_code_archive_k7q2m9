"""Replay a trace through the scheduler."""

import json
import random


def main(path: str) -> None:
    with open(path, "r", encoding="utf-8") as handle:
        trace = json.load(handle)
    random.seed(3)
    served = sum(len(step.get("served", [])) for step in trace)
    print(f"requests={served}")


if __name__ == "__main__":
    main("benchmarks/trace.json")
