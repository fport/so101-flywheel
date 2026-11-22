"""Slices a print file with Bambu Studio's own slicer (command line) and summarizes it.

Instead of guessing from the on-screen preview: warnings, time, filament, and per-layer bridge / top surface
areas. If the bridge area jumps on some layer (e.g. a floor without feet) it shows up before printing.
"""

import json
import math
import re
import subprocess
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

BAMBU = Path("/Applications/BambuStudio.app/Contents/MacOS/BambuStudio")
PROFILES = Path("/Applications/BambuStudio.app/Contents/Resources/profiles/BBL")
MACHINE = "machine/Bambu Lab X2D 0.4 nozzle.json"
PROCESS = "process/0.20mm Standard @BBL X2D.json"  # Bambu Studio's default for the X2D
FILAMENT = "filament/Bambu PLA Basic @BBL X2D 0.4 nozzle.json"
LINE = 0.42  # line width for area estimates (mm)


def filament_count(path):
    """Two-color jobs (filament per part) carry the filament list in the Bambu project config."""
    with zipfile.ZipFile(path) as z:
        if "Metadata/project_settings.config" not in z.namelist():
            return 1
        return len(json.loads(z.read("Metadata/project_settings.config"))["filament_settings_id"])


def resolved_profile(rel, out_dir):
    """Merges a profile along its `inherits` chain and writes it to out_dir. The command line doesn't resolve the chain;
    the raw profile slices with values that differ from Bambu Studio (e.g. 4 top shell layers, no elephant foot fix)."""
    path = PROFILES / rel
    chain = []
    while path.exists():
        chain.append(json.loads(path.read_text()))
        path = path.parent / f"{chain[-1].get('inherits', '')}.json"
    merged = {}
    for prof in reversed(chain):
        merged.update(prof)
    merged.pop("inherits", None)
    out = Path(out_dir) / Path(rel).name
    out.write_text(json.dumps(merged))
    return out


def slice_3mf(path, out_dir):
    prof_dir = Path(out_dir) / "profiles"
    prof_dir.mkdir()
    machine, process, filament = (resolved_profile(r, prof_dir) for r in (MACHINE, PROCESS, FILAMENT))
    cmd = [str(BAMBU), "--slice", "0", "--outputdir", str(out_dir),
           "--load-settings", f"{machine};{process}",
           "--load-filaments", ";".join([str(filament)] * filament_count(path)), str(path)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=900)
    result = json.loads((Path(out_dir) / "result.json").read_text())
    return result, sorted(Path(out_dir).glob("plate_*.gcode"))


def layer_features(gcode):
    """{z: {feature: area_mm2}} — extrusion length x line width."""
    out = defaultdict(lambda: defaultdict(float))
    z, feat, x, y = 0.0, None, None, None
    for line in open(gcode):
        if line.startswith("; Z_HEIGHT:"):
            z = round(float(line.split(":")[1]), 2)
        elif line.startswith("; FEATURE:"):
            feat = line.split(":", 1)[1].strip()
        elif line[:3] in ("G1 ", "G0 "):
            m = dict(re.findall(r"([XYE])([-\d.]+)", line))
            nx, ny = float(m.get("X", x or 0)), float(m.get("Y", y or 0))
            if line[:2] == "G1" and float(m.get("E", 0)) > 0 and x is not None:
                out[z][feat] += math.hypot(nx - x, ny - y) * LINE
            x, y = nx, ny
    return out


def report(path):
    with tempfile.TemporaryDirectory() as tmp:
        result, gcodes = slice_3mf(path, tmp)
        lines = [f"{Path(path).name}: {result.get('error_string')}"]
        for plate, gcode in zip(result["sliced_plates"], gcodes):
            hours = plate["total_predication"] / 3600
            warn = plate.get("warning_message") or "none"
            lines.append(f"  time {hours:.1f} h, warning: {warn}")
            if len(plate["filaments"]) > 1:
                used = ", ".join(f"filament {f['id']}: {f['total_used_g']:.1f} g" for f in plate["filaments"])
                lines.append(f"  {used}; filament changes {plate['filament_change_times']}")
            feats = layer_features(gcode)
            first_bridge = {z: f.get("Bridge", 0.0) for z, f in feats.items() if f.get("Bridge", 0.0) > 0}
            if first_bridge:
                z, area = max(first_bridge.items(), key=lambda kv: kv[1])
                lines.append(f"  largest bridge layer z={z} mm, {area:.0f} mm2 (includes internal bridges; normal unless on an outer surface)")
            tops = [z for z, f in feats.items() if f.get("Top surface", 0) > 1]
            lines.append(f"  {len(feats)} layers, layers with top surface: {', '.join(str(z) for z in sorted(tops))}")
        return "\n".join(lines)
