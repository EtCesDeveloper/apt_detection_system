"""
Entry point for the APT detection project.

Usage:
    python main.py train       # full training pipeline
    python main.py evaluate    # evaluate the best saved model
    python main.py visualize   # generate plots (history, confusion matrix, ROC, etc.)
    python main.py summary     # just print the model summary (sanity check)
"""

import sys
from src import config
from src.model import build_multitask_model


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "train":
        from src.train import main as train_main
        train_main()
    elif cmd == "evaluate":
        from src.evaluate import main as eval_main
        eval_main()
    elif cmd == "visualize":
        from src.visualize import main as viz_main
        viz_main()
    elif cmd == "summary":
        m = build_multitask_model(
            timesteps=config.TIMESTEPS,
            n_features=config.N_FEATURES,
        )
        m.summary()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
