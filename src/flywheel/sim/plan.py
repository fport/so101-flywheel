"""Balanced plan for the scripted sim run (`flywheel-sim batch`).

To break shortcut memorization each part is picked twice: once into the correct bin per the work order, once
deliberately into the opposite bin (command-following probe, `probe`). So the demos can't tie "this look -> this
bin"; the policy can learn the bin only from the command. Pockets, neighbours and pick order are shuffled by seed.
"""

import numpy as np

from ..task import POCKETS


def build_plan(catalog, order="W", seed=0, heldout=False):
    """Returns [{tray, pockets: [id|None]*6, picks: [{part, pocket, command, probe}]}]."""
    rng = np.random.default_rng(seed)
    parts = [c for c in catalog if c.heldout == heldout]
    first_probe = {c.id: bool(rng.integers(2)) for c in parts}  # which pass gets the opposite command
    last_pocket, trays = {}, []
    for rnd in range(2):
        ids = [c.id for c in parts]
        rng.shuffle(ids)
        for start in range(0, len(ids), POCKETS):
            chunk = ids[start : start + POCKETS]
            for _ in range(50):  # part lands in a different pocket on the second pass
                slots = rng.permutation(POCKETS)[: len(chunk)]
                if all(last_pocket.get(pid) != int(slot) for pid, slot in zip(chunk, slots)):
                    break
            pockets = [None] * POCKETS
            for pid, slot in zip(chunk, slots):
                pockets[int(slot)] = pid
                last_pocket[pid] = int(slot)
            by_id = {c.id: c for c in parts}
            picks = []
            for pid in rng.permutation(chunk):
                pid = str(pid)
                probe = first_probe[pid] == (rnd == 0)
                expected = by_id[pid].expected_bin(order)
                command = ("reject" if expected == "ok" else "ok") if probe else expected
                picks.append({"part": pid, "pocket": pockets.index(pid) + 1, "command": command, "probe": probe})
            trays.append({"tray": len(trays) + 1, "pockets": pockets, "picks": picks})
    return trays
