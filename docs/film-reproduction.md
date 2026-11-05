# Film editors and authored robot previews

The film tools are local editors and browser capture helpers. They do not run inference, upload footage, publish a video, or execute a physical robot. The robot motion is an authored kinematic rehearsal using saved object assets and explicit display fixtures. It does not verify grasp forces, contact, insertion or charging.

The implementation is inspectable in these source files:

- [`industrial-arm-presentation-20261105-v1.js`](../app/showcase/industrial-arm-presentation-20261105-v1.js) builds the original industrial-style arm display. It changes presentation geometry, rather than supplying a simulator or robot driver.
- [`case-grasp-adapter-20261105-v2.js`](../app/showcase/case-grasp-adapter-20261105-v2.js) applies explicit surface-grasp offsets, world jaw direction, aperture and lid-cue choices to the authored case task.
- [`case-B-display-adapter-20261105-v3.js`](../app/showcase/case-B-display-adapter-20261105-v3.js) declares the trailer's A-based display proxy for the second earbud and keeps the original B asset separate. The bound capture states the exact copy/pose method. This is an authored presentation derivative, not a repaired reconstruction of B.
- [`charging-rehearsal-plan-20261105-v1.js`](../app/showcase/charging-rehearsal-plan-20261105-v1.js) defines the ordered two-end cable task, release, retreat and final hold. The [view module](../app/showcase/charging-rehearsal-view-20261105-v1.js) defines its editorial camera.
- [`build_story_narration_20261105_v3.py`](../app/scripts/build_story_narration_20261105_v3.py) times complete narration paragraphs at a natural local voice rate, rejects overflowing takes, and exports a mix, captions and transcript.
- [`build_story_robot_films_20261105_v1.py`](../app/scripts/build_story_robot_films_20261105_v1.py) assembles the separately validated case and charging captures with the visual pipeline and application walkthrough.
- [`replace_case_film_20261105_v3.py`](../app/scripts/replace_case_film_20261105_v3.py) replaces the case shot once for both film lengths, copies the other video ranges and audio packets, and checks the copied decoded pictures. A small keyframe-boundary interval is explicitly recorded as re-encoded.

The browser captures have exact-byte receipts and independent intake validators. A recorder trace is checked against the pinned authored plan and source references; the downloaded video is separately decoded to check its dimensions and continuous clock. Those checks are evidence about the displayed rehearsal and recording, not a physical validation.

The latest short film's [narration transcript](media/transcript.txt) and [English captions](media/captions.vtt) accompany its downloadable MP4. The current narration uses the normal local voice; the editor preserves that audio when replacing the case shot.

The source checkout deliberately excludes the recordings, saved approved assets, private capture bindings, review receipts and rendered working files. Exact film reproduction requires those separately approved inputs at the paths checked by the helpers. Generated capture pages contain references to that omitted saved context and are local developer tools; they are not additional public demo routes. Do not substitute another asset or rebind an old receipt to changed bytes.

The optional film environment needs FFmpeg, Pillow and NumPy. The narration tool additionally uses macOS `say`; geometry preparation uses SciPy and trimesh. These are separate from the lightweight application review environment. No model weights are needed for the synthetic checks below.

From `app/`, the decoded-clock fixtures need only Python's standard library:

```sh
python3 -I -S -B -m unittest discover -s tests -p test_charging_capture_clock_v1.py -v
node --test tests/test_charging_rehearsal_v1.cjs
node --test tests/test_case_B_display_proxy_v3.cjs
```

The Node test exercises ordered endpoint motion and camera continuity with synthetic coordinates. Saved-asset checks explicitly skip when their separate inputs are absent. Full captures and fresh inference are not reproduced by these fixtures.

In the optional film environment, `python -B -m unittest discover -s tests -p test_case_film_replacement_v3.py -v` checks the editor's segment/copy behavior using a tiny generated clip. It uses no owner footage.
