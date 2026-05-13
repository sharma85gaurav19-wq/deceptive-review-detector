"""Main entry point for deceptive-review-detector pipeline."""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from src import data_generator
from src import ablation
from src import cross_domain
from src import evaluate
from src import explain
from src import plots
from src import threshold_analysis
from src import train

LOG_PATH = Path("outputs/logs/run.log")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
LOGGER = logging.getLogger(__name__)


def _log_and_print(message: str) -> None:
    LOGGER.info(message)
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}")


def main(args: argparse.Namespace) -> None:
    if args.generate_data:
        _log_and_print("Generating synthetic datasets...")
        data_generator.generate_synthetic_datasets()
    if args.train:
        _log_and_print("Training models...")
        train.train_all(args.dataset)
    if args.evaluate:
        _log_and_print("Evaluating models...")
        evaluate.evaluate_all(args.dataset)
    if args.ablation:
        _log_and_print("Running ablation study...")
        ablation.run_ablation()
    if args.cross_domain:
        _log_and_print("Running cross-domain evaluation...")
        cross_domain.run_cross_domain()
    if args.threshold:
        _log_and_print("Running threshold analysis...")
        threshold_analysis.run_threshold_analysis()
    if args.explain:
        _log_and_print("Generating LIME explanations...")
        explain.generate_lime_examples()
    if args.plots:
        _log_and_print("Generating plots...")
        plots.generate_all_plots()
    if args.all:
        _log_and_print("Starting full pipeline: generation, training, evaluation, experiments, explanations, plots...")
        data_generator.generate_synthetic_datasets()
        train.train_all(args.dataset)
        evaluate.evaluate_all(args.dataset)
        ablation.run_ablation()
        cross_domain.run_cross_domain()
        threshold_analysis.run_threshold_analysis()
        explain.generate_lime_examples()
        plots.generate_all_plots()
    _log_and_print("Pipeline completed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the deceptive review detector pipeline.")
    parser.add_argument("--generate-data", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--cross-domain", action="store_true")
    parser.add_argument("--threshold", action="store_true")
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dataset", choices=["amazon", "yelp", "both"], default="both")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
