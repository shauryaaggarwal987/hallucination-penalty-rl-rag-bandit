import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))


def generate_plots(episodes_csv):
    episodes_csv = Path(episodes_csv)

    if not episodes_csv.exists():
        raise FileNotFoundError(f"No episode log found at: {episodes_csv}")

    df = pd.read_csv(episodes_csv)

    if df.empty:
        raise ValueError("Episode log is empty.")

    # A scientific learning curve must represent one isolated run.
    if "run_id" in df.columns and df["run_id"].nunique() != 1:
        raise ValueError(
            "The supplied CSV contains multiple run_ids. "
            "Generate a learning curve from one isolated run only."
        )

    if "episode" not in df.columns:
        raise ValueError("The CSV does not contain an 'episode' column.")

    df = df.sort_values("episode").reset_index(drop=True)

    # Ensure the episode numbering is unique inside the run.
    if df["episode"].duplicated().any():
        raise ValueError(
            "Duplicate episode numbers found. "
            "The file may contain multiple appended runs."
        )

    df["reward_ma"] = df["chosen_reward"].rolling(
        window=20,
        min_periods=1
    ).mean()

    output_dir = episodes_csv.parent

    # Reward curve
    plt.figure(figsize=(10, 6))

    plt.plot(
        df["episode"],
        df["chosen_reward"],
        label="Per-Episode Reward",
        alpha=0.3
    )

    plt.plot(
        df["episode"],
        df["reward_ma"],
        label="Reward (20-Episode Moving Average)",
        linewidth=2
    )

    plt.xlabel("Episode Within Isolated Run")
    plt.ylabel("Reward")
    plt.title("Reward Trajectory During Training")
    plt.legend()
    plt.grid(True)

    reward_path = output_dir / "reward_curve.png"
    plt.savefig(reward_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {reward_path}")

    # Action counts
    plt.figure(figsize=(10, 6))

    counts = (
        df["chosen_action"]
        .value_counts()
        .reindex(["A0", "A1", "A2", "A3", "A4"], fill_value=0)
    )

    counts.plot(kind="bar")

    plt.xlabel("Action")
    plt.ylabel("Selection Count")
    plt.title("Action Selection Frequency in Isolated Training Run")
    plt.grid(axis="y")

    counts_path = output_dir / "action_counts.png"
    plt.savefig(counts_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {counts_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to episodes.csv from ONE isolated experimental run"
    )
    args = parser.parse_args()

    generate_plots(args.csv)