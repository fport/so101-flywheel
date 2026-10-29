# 3D printing: lessons learned and checklist

The parts of this cell are generated from code (`python/flywheel/hardware/`, dimensions in `hardware/params.yaml`). The points below come
from mistakes that actually happened in the first prints; go through this list when designing a new part or before sending a file
to the printer.

## Printer and material

- **Bambu Lab X2D, dual nozzle.** The plate is 256 × 256 × 260 mm, but the x range both nozzles can reach is 20.5–235.5 mm.
  The grey strip on the left ("Left nozzle only area") is open only to the main nozzle. All jobs are laid out inside `printer.shared_x`;
  `cli.pack` enforces this and a test checks it too.
- **PLA only.** Colours: black (ground, surfaces the camera sees), metal grey, orange, white.
  Matte filament for surfaces visible on camera; glossy filament reflects.
- **All QC parts of a variant come from the same spool.** A shade difference breaks the held-out measurement. If the spool is low, the parts
  are printed first and big jobs like the bin are left for last (`EXTRA_JOBS`). A failed part is added to the last job of the same colour
  (`REPRINTS`).
- **Orange is scarce.** It was out of stock in the store; it is only used for the orange QC parts and the reject bin.

## Design rules (each from a real mistake)

| Rule | Why |
| --- | --- |
| A Gridfinity foot must be on every cell, not only at the corners | With corner feet only, the body's floor was 4.75 mm in the air and an 84 mm bridge sagged in the middle; the tray and backdrop came out wavy |
| The male fit must be straight along the length of the female hole; a tapered male end in a straight hole only touches at the mouth | Every joint on the mast wobbled like a hinge, ~1 cm sway at the 66 cm top |
| In tight fits, let crush ribs take up print error (~0.05 mm interference per side), not a flat surface | FDM is ±0.1 mm; a 0.05 mm flat clearance either jams or wobbles |
| Don't make walls and rims thinner than 0.8 mm | The tag seat's 0.2 mm wall could not be printed; the 0.5 mm wall on the brake pad ear would break |
| In PLA, a spring tab that is 1.5 mm thick and short breaks on first insertion | The camera cradle's tab broke; either make it long and thin or use a rubber band |
| Build the fit into the body instead of using loose small parts (wedge, clip) | The user wants few parts; the dovetails were moved onto the tiles as tongues |
| Don't add parts nobody asked for; ask before adding | Cable clips were printed without being asked for |
| Defects must be visible from the top camera | The bracket's "missing hole" was in the side hole; the camera could never see it |
| Put an orientation mark on a symmetric part that can be fitted the wrong way round | The tag tile rotates 180° and the seat shifted; the tag number was engraved |
| Brim (5 mm) on tall thin parts | The 250 mm mast pieces |
| Two colours don't work with a full plate | Bambu puts the purge tower at the foot of the parts and it can't be controlled with settings; single-colour embossed text is enough |
| If two colours are needed, each colour should be in its own nozzle, with no colour changes | Tag tiles: black body + 1 mm white inlay, two nozzles on the X2D; no tower and no purging. `slicecheck` counts filament changes, it must be 0 |

Parts that move during assembly (the feet have 0.25 mm clearance per side in the grid) are fixed with double-sided tape; the tray, bins and tag tiles
must not slide, since the arm learns their positions.

## Check before sending

```bash
cd python
uv run flywheel-cad build          # generates the files, checks collisions / plate / fits
uv run pytest -q                   # geometry tests (bridge under feet, tongue-pocket lock, shared nozzle area...)
uv run flywheel-cad slicecheck ../hardware/print/06_*.3mf   # Bambu's own slicer: warnings, time, layers
```

In addition:

1. **Is the file up to date?** If a `hardware/print/*.3mf` contains `Metadata/plate_1.png` (thumbnail), Bambu Studio has saved the project
   over it and the geometry may be stale. (In two-colour jobs, the `Metadata/*.config` files we write ourselves are normal.) `flywheel-cad build` regenerates it.
2. **For a new or changed job**, have an independent agent measure the geometry (overhang/bridge, thin walls, collision and clearance in the assembled
   position, plate layout). Looking at the preview by eye was not enough.
3. **Small test before a long print:** new fit tolerances (e.g. the mast) first with the small pieces in `01_test`.
4. **When interpreting a screenshot**, check what height the layer slider is at. The wavy pattern on an intermediate layer is sparse infill;
   it is not an outer surface problem. If unsure, run `slicecheck`, don't guess.

## Common snags in Bambu Studio

- The "The 3mf file has invalid config" warning is normal: our files carry no Bambu settings. Click OK.
- Open the file with **File > New Project**, then **File > Import**. Don't open it from "Recent projects"; you get stale geometry.
  If you want to save, use **Save Project As** and save outside `hardware/print/`.
- AMS mapping: when sending, black is **A1**. The colour in the preview is the colour registered in the AMS; if it differs from the real spool
  it does not affect the print and can be fixed from the Device tab.
- When dragging a part it is easy to grab only a sub-part instead of the object; don't drag multi-part objects.
- **Two-colour job (10, tag tiles):** Import keeps the current project's filament list; add the second filament with **+** in
  Project Filaments. In the object list `tag_N_1` → 1 and `tag_N_2` → 2; then set filament 1's colour to **black**
  and filament 2's to **white** (if reversed, all tags come out negative and the camera can't read them). Correct result: large tiles black,
  number white; small tiles white up to the edge. If parts from a previous project are left on the plate, delete them.
- A part that doesn't stick on the first layer: the **Skip** icon to the left of the pause button in Device > Printing Progress.
  Find the skipped part's code in the table (the `13_*.3mf` layout) and add it to `REPRINTS`.
- Most first-layer problems come from grease on the plate: wash it with dish soap and warm water.
- When printing PLA in an enclosed printer, leave the top lid ajar.
