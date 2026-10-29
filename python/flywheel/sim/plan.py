"""Balanced recording / eval plan.

To break shortcut memorization each part is picked twice: once into the correct bin per the work order, once
deliberately into the opposite bin (command-following probe, `probe`). So the demos can't tie "this look -> this
bin"; the policy can learn the bin only from the command. Pockets, neighbours and pick order are shuffled by seed.
"""

import numpy as np

from .parts import TYPE_NAMES, VARIANTS

TASK = {"ok": "put the part in the OK bin", "reject": "put the part in the REJECT bin"}
POCKETS = 6


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


def to_markdown(trays, catalog, order, title):
    by_id = {c.id: c for c in catalog}
    n = sum(len(t["picks"]) for t in trays)
    lines = [
        f"# {title}",
        "",
        f"Work order: **{VARIANTS[order]}** variant. {len(trays)} trays, {n} episodes; each part goes once into the "
        "correct bin and once deliberately into the opposite bin (`probe` column).",
        "",
        "There is one tray; a tray means laying 6 parts in it and picking them in order. Episode numbers run across the "
        "whole plan. Pocket 1 is at the left of the row nearest the arm (seen from behind the arm), 4 at the left of the far row.",
        "",
    ]
    no = 0
    for t in trays:
        lines += [f"## Tray {t['tray']}", "", "| Pocket | Part | Type |", "| --- | --- | --- |"]
        for k, pid in enumerate(t["pockets"], start=1):
            if pid:
                lines.append(f"| {k} | `{pid}` | {TYPE_NAMES[by_id[pid].type]} |")
        lines += ["", "| Episode | Pocket | Part | Command | Probe |", "| --- | --- | --- | --- | --- |"]
        for pick in t["picks"]:
            no += 1
            lines.append(f"| {no} | {pick['pocket']} | `{pick['part']}` | {TASK[pick['command']]} | {'yes' if pick['probe'] else ''} |")
        lines.append("")
    return "\n".join(lines)
