"""Episodes recorded so far, per task sentence:  scripts/status.sh"""

import collections
import sys

from lerobot.datasets import LeRobotDatasetMetadata

repo_id, root, target = sys.argv[1], sys.argv[2], int(sys.argv[3])
meta = LeRobotDatasetMetadata(repo_id, root=root)
if meta.episodes is None:
    sys.exit("0 episodes so far. Start with: scripts/record.sh 1 ok")
done = collections.Counter(t for tasks in meta.episodes["tasks"] for t in tasks)
total = 0
for bin_ in ("OK", "REJECT"):
    for pocket in range(1, 7):
        task = f"put the part from pocket {pocket} in the {bin_} bin"
        n = done.pop(task, 0)
        total += n
        print(f"  {'✓' if n >= target else ' '} {n:3d}/{target}  {task}   {'' if n >= target else f'scripts/record.sh {pocket} {bin_.lower()} {target - n}'}")
for task, n in done.items():
    print(f"  ? {n:3d}      {task}")
print(f"\n  {total}/{12 * target} episodes · {meta.total_frames / meta.fps / 60:.1f} min · {root}")
