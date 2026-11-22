# Concepts in one page

| Term | In this project |
| --- | --- |
| **Teleoperation** | You move the leader arm, the follower copies it. This is how demonstrations are made |
| **Episode** | One recorded attempt: one part, from the start pose back to the start pose |
| **Observation** | What the policy sees each step: two images + six joint angles |
| **Action** | What it outputs: six target joint angles (here: where the leader was) |
| **Policy** | The function observation + sentence → actions. The thing you train |
| **Imitation learning / behaviour cloning** | Train the policy to output what you did in the same situation. No reward |
| **Distribution shift** | The policy drifts into states you never showed; its errors then compound. Why variety and corrections matter |
| **Action chunking** | Predict the next 50 actions at once instead of one: smoother, and small errors do not accumulate per step |
| **VLM** | Vision-language model: image encoder + projector + language model. Reads images, writes text |
| **VLA** | A VLM with an action head: reads images and a sentence, writes robot actions. SmolVLA, π0, GR00T |
| **Flow matching** | How SmolVLA's action head works: start from noise, refine into an action chunk in a few steps |
| **Fine-tuning** | Start from pretrained weights and train on your small dataset. Here the VLM stays frozen and the action expert is trained |
| **Grounding** | Linking words to places in an image: boxes or points for "the gear" |
| **Rollout** | Running the policy on the real robot |
| **DAgger** | Let the policy run, correct it when it fails, add the corrections to the data, retrain |
| **Data flywheel** | Record → train → evaluate → collect what failed → retrain. Each turn makes the next one cheaper |
| **World model** | A model that predicts what the camera will see next. Cosmos 3 is one; it can also emit actions |
| **Sim-to-real** | Train on simulated data, run on the real robot. The gap is mostly how things look and how they slip |

## Reading: *Vision Language Models* (Noyan, Farré, Marafioti)

Read in this order for this project; page numbers are PDF pages.

| Read | Pages | For |
| --- | --- | --- |
| Ch. 11, VLAs | 499–516 | How SmolVLA is built, action chunks, the asynchronous control loop |
| Ch. 4, VLA data | 146–149, 160, 175–185 | Checks before training: frame rate, joint limits, timing between frames and joints |
| Ch. 2, localisation | 87–94 | Open-vocabulary detection, pointing, location tokens: for the inspection step |
| Ch. 11, structured points | 484–487 | Getting validated coordinates out of a VLM |
| Ch. 5, LoRA / QLoRA | 186–204 | Adapting a small VLM as a verifier |
| Ch. 3 and 6 | 131–134, 235–251 | Image tokens, tiling, pixel shuffle; SmolVLM's architecture |

Skip for now: Ch. 8 (documents), Ch. 10 (any-to-any), the agents part of Ch. 11.

Three ideas from the book that this project uses directly:

1. A VLA is a VLM plus an action expert; the VLM runs once per chunk, the expert denoises (p. 507–511).
2. An 80 ms mismatch between a frame and its joint reading already teaches wrong behaviour: check timing before
   training (p. 506–507).
3. For a success check, judge the end state ("is the part in the bin?"), not each step (p. 474–475).
