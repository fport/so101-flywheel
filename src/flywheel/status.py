"""flywheel-status: episodes recorded so far, per task sentence (called by scripts/status.sh)."""

import collections
import sys

from .task import POCKETS, task


def main():
    from lerobot.datasets import LeRobotDatasetMetadata

    repo_id, root, target = sys.argv[1], sys.argv[2], int(sys.argv[3])
    meta = LeRobotDatasetMetadata(repo_id, root=root)
    if meta.episodes is None:
        sys.exit("0 episodes so far. Start with: scripts/record.sh 1 ok")
    done = collections.Counter(t for tasks in meta.episodes["tasks"] for t in tasks)
    total = 0
    for bin_ in ("ok", "reject"):
        for pocket in range(1, POCKETS + 1):
            sentence = task(pocket, bin_)
            n = done.pop(sentence, 0)
            total += n
            todo = "" if n >= target else f"scripts/record.sh {pocket} {bin_} {target - n}"
            print(f"  {'✓' if n >= target else ' '} {n:3d}/{target}  {sentence}   {todo}")
    for sentence, n in done.items():
        print(f"  ? {n:3d}      {sentence}")
    print(f"\n  {total}/{2 * POCKETS * target} episodes · {meta.total_frames / meta.fps / 60:.1f} min · {root}")


if __name__ == "__main__":
    main()
