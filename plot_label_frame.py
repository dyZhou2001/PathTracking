"""Plot one sample from CARLA dataset labels.jsonl.

Usage:
  python plot_label_frame.py --labels dataset/run_xxx/labels.jsonl --index 0
  python plot_label_frame.py --labels dataset/run_xxx/labels.jsonl --frame 69958

Output:
  Saves a PNG next to labels.jsonl by default.
"""

import argparse
import json
from pathlib import Path


def _read_jsonl_line(labels_path: Path, index: int):
    with labels_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line)
    raise IndexError(f"index out of range: {index}")


def _read_jsonl_by_frame(labels_path: Path, frame: int):
    with labels_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if int(obj.get("frame", -1)) == int(frame):
                return obj
    raise ValueError(f"frame not found: {frame}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, help="Path to labels.jsonl")
    parser.add_argument("--index", type=int, default=None, help="0-based line index")
    parser.add_argument("--frame", type=int, default=None, help="Frame id")
    parser.add_argument("--out", default=None, help="Output png path")
    args = parser.parse_args()

    labels_path = Path(args.labels)
    if not labels_path.exists():
        raise FileNotFoundError(str(labels_path))

    if (args.index is None) == (args.frame is None):
        raise SystemExit("Please provide exactly one of --index or --frame")

    if args.frame is not None:
        sample = _read_jsonl_by_frame(labels_path, args.frame)
    else:
        sample = _read_jsonl_line(labels_path, args.index)

    frame = int(sample.get("frame", -1))
    episode_step = int(sample.get("episode_step", -1))

    future = sample.get("future_route_vehicle", []) or []
    target = sample.get("target_point_vehicle", None)

    xs = [float(p["x"]) for p in future if "x" in p and "y" in p]
    ys = [float(p["y"]) for p in future if "x" in p and "y" in p]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5), dpi=140)

    # Vehicle at origin in its own frame
    ax.scatter([0.0], [0.0], s=60, c="black", label="vehicle (0,0)")

    if xs:
        ax.plot(xs, ys, "-o", markersize=3, linewidth=1.5, label="future_route_vehicle")

    if isinstance(target, dict) and "x" in target and "y" in target:
        tx, ty = float(target["x"]), float(target["y"])
        ax.scatter([tx], [ty], s=80, c="#00aa00", marker="x", label="target_point_vehicle")

    ax.axhline(0.0, color="#cccccc", linewidth=1)
    ax.axvline(0.0, color="#cccccc", linewidth=1)

    ax.set_title(f"Local path in vehicle frame | frame={frame} step={episode_step}")
    ax.set_xlabel("x (meters, forward)")
    ax.set_ylabel("y (meters, right)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best")

    out_path = Path(args.out) if args.out else labels_path.parent / f"plot_frame_{frame}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    print(str(out_path))


if __name__ == "__main__":
    main()
