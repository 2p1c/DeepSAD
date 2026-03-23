"""Entry point for Deep SVDD training."""

from __future__ import annotations

import argparse

from agents.pipeline_agent import PipelineAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Deep SVDD for ultrasound anomaly detection.")
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the pipeline config file.",
    )
    args = parser.parse_args()

    agent = PipelineAgent()
    agent.run_train(args.config)


if __name__ == "__main__":
    main()
