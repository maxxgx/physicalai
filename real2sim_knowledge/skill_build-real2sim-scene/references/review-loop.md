# Review loop

The builder (you) and the reviewer (a fresh subagent each round) never share notes. The reviewer reads only the rubric,
the delivery, its own previous reports and its `NOTES.md`. This is what makes the grade worth reporting.

## Setup (once, before building)

1. **Held-out split.** Choose windows spread over the video, away from the first and last seconds, from the timestamps
   alone: about 15% of the duration in three to five windows (five 2 s windows for a 75 s video, three 1 s windows for
   20 s), each with a 0.5 s guard. Write `review/HELDOUT_PHONE.json`:
   `{"windows_s": [[9.5, 11.5], [27.5, 29.5], ...], "guard_s": 0.5}`. Frames are named by video index
   (`fNNNNN.jpg`, t = NNNNN / fps), so `common.held()` filters by name without opening anything.
2. **Dev split for yourself.** Five other 2 s blocks of allowed frames in `notes/DEV_SPLIT.json`, left out of a second
   "fair" splat, so `dev_eval.py` can score changes before a review is spent on them.
3. **Rubric.** Adapt `assets/rubric-example.md` to the scene: its dimensions (phone novel views, phone video with the sim
   running, layout, task, overview camera, wrist camera, readiness), weights, hard rules, delivery format and section 7
   procedure carry over; replace object names, counts and thresholds. Copy `assets/phone_eval.py` to `review/tools/`.
   Freeze both: after the first review they do not change. Then stop writing under `review/`.

## Each round

1. Score the change on the dev split. Spend a review only when something improved or a blocker was addressed.
2. `deliver.py rNN` freezes a delivery: plugin snapshot, assets, kit, versions, the MANIFEST with every command the
   reviewer runs, and the notes on declarations and changes (`notes/MANIFEST_NOTES.md`). Run its commands once under
   the read-deny sandbox from the rubric to prove they never touch the held-out frames.
3. Spawn a fresh `general-purpose` reviewer in the background with this brief (fill in the paths and NN):

   > You are the independent reviewer for a real2sim project. Grade delivery rNN strictly against the frozen rubric
   > `<REV>/RUBRIC.md` (section 7 is your procedure). Do not fix or edit anything, do not soften grades. Read only the
   > rubric, the MANIFEST and delivery files, `<REV>/NOTES.md`, previous reports and your own evidence; nothing under
   > `<ROOT>/notes/`. Run every command yourself, draw your own seeds and video segments, and record failures as the
   > rubric says. Work in `<REV>/work/rNN/`, write `<REV>/reports/rNN.md` and add your readings to `NOTES.md`; write
   > nothing else. Never describe held-out frames beyond what the report needs. Run SPEED with nothing else running.
   > Reply in under 300 words: levels D1 to Dn with the previous level, the overall grade, hard-rule findings, key
   > numbers, and the top three conditions blocking the next level.

4. Fix the top blockers it names. Add the changes to `MANIFEST_NOTES.md` for the next delivery.

## Stop rule

Stop and report when two rounds in a row change no level, when the remaining blockers need the user (a camera pose,
real recordings, a decision), or at the user's budget. Say what the next level needs and who must provide it.
