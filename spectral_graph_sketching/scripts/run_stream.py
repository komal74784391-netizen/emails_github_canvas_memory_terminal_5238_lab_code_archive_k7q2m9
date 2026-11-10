"""Run the detector over a synthetic stream."""

import argparse
import random


def main(events: int) -> None:
    random.seed(11)
    hits = 0
    for _ in range(events):
        if random.random() < 0.01:
            hits += 1
    print(f"processed={events} anomalies={hits}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=int, default=5000)
    args = parser.parse_args()
    main(args.events)
