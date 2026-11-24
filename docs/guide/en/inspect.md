# 3 · Inspect the tray

**Goal:** from one photo of the tray, a bin for each of the six pockets. This is the part that decides; the arm
only executes. Keeping the two apart means a wrong decision and a clumsy grasp are separate, fixable problems.

```bash
uv run flywheel-inspect                       # top camera, now
uv run flywheel-inspect --parts L-T-02 B-W-07 J-W-08 J-W-06 D-W-02 D-W-06   # score it: IDs in pocket order
uv run flywheel-inspect --image outputs/inspect/<time>/frame.png          # a saved frame
```

Pocket 1 is nearest the arm on its left; 4 is behind it. The part ID is engraved under each part.

## How it works

1. **Find the tags.** Four AprilTags sit at known millimetre positions, so one homography maps tray millimetres to
   image pixels, wherever the camera is.
2. **Cut six crops**, one per pocket, all at the same scale (4 px/mm).
3. **Measure** on the silhouette of the light part against the black tray: holes (the tray shows through), rim spokes,
   gear teeth, a corner missing from the outline, colour.
4. **Ask a VLM** (Claude, if `ANTHROPIC_API_KEY` is set) for what counting cannot give: the part type, and a second
   opinion. Without a key, pass `--parts` so the type is known.
5. **Decide**: REJECT if the colour is not the one in the work order (`--order W|T`), a hole is missing, or the part is
   broken. Otherwise OK.

It ends with the sentences the policy will be given, one per pocket.

## Check

Open `outputs/inspect/<time>/overlay.jpg`: the orange squares must sit on the pockets. Then run it with `--parts` on
three or four different trays; `bin 6/6` each time is the bar before you trust it.

## Try next

The measuring step is hand-written for these four parts. A *grounding* model finds things from a text prompt instead:
see [grounding](learn-grounding.md).
