"""CLI entry point for the company performance prediction pipeline."""

import argparse


def cmd_fetch() -> None:
    from src.data.fetch import run_fetch

    run_fetch()


def cmd_features() -> None:
    from src.features.build_features import build_features

    build_features()


def cmd_train() -> None:
    from src.models.train import train_models

    train_models()


def cmd_predict() -> None:
    from src.models.predict import run_predict

    run_predict()


def cmd_all() -> None:
    cmd_fetch()
    cmd_features()
    cmd_train()
    cmd_predict()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Company performance prediction MLOps pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("fetch", help="Fetch raw data snapshots from live APIs")
    subparsers.add_parser("features", help="Engineer features from raw data")
    subparsers.add_parser("train", help="Train and compare models")
    subparsers.add_parser("predict", help="Run on-demand batch predictions")
    subparsers.add_parser("all", help="Run fetch → features → train → predict")

    args = parser.parse_args()

    commands = {
        "fetch": cmd_fetch,
        "features": cmd_features,
        "train": cmd_train,
        "predict": cmd_predict,
        "all": cmd_all,
    }
    commands[args.command]()


if __name__ == "__main__":
    main()
