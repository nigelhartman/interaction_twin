# SAM 3.1 local checkpoint adapter: prepared, not yet run

`scripts/sam31_segment.py` provides a non-inferencing preflight and a bounded
future text-segmentation entry point for one timestamped source image. No gated
weights, account credentials, model runtime installation or learned SAM 3.1
inference were accessed during this preparation.

## Verified official API distinction

The public official source is cloned, without modification, at
`models/sam3`, commit
`f1647e8d0c1d280991f4d1b7d974c2704537fe66`.
The Python package reports `0.1.0`; the model generation is selected separately
through `version="sam3.1"` and its checkpoint. The SAM 3.1 release introduces
Object Multiplex for joint video tracking. This preparation does not claim that
tracking already ran. [Official release notes](https://github.com/facebookresearch/sam3/blob/f1647e8d0c1d280991f4d1b7d974c2704537fe66/RELEASE_SAM3p1.md)

The familiar `build_sam3_image_model()` still constructs the SAM 3 detector and
defaults to SAM 3 weights. It should not simply be relabelled SAM 3.1. The reviewed
SAM 3.1 entry point instead builds the multiplex predictor and loads
`sam3.1_multiplex.pt`. [Official builder](https://github.com/facebookresearch/sam3/blob/f1647e8d0c1d280991f4d1b7d974c2704537fe66/sam3/model_builder.py)

For the small initial scope, that predictor explicitly accepts a PNG/JPEG image
as a single-frame video resource. The adapter uses `start_session(image_path)`,
then `add_prompt(frame_index=0, text=...)`, then closes the session. The original
video PTS remains in our metadata; session index zero is not substituted for it.
[Official image loader](https://github.com/facebookresearch/sam3/blob/f1647e8d0c1d280991f4d1b7d974c2704537fe66/sam3/model/io_utils.py),
[request API](https://github.com/facebookresearch/sam3/blob/f1647e8d0c1d280991f4d1b7d974c2704537fe66/sam3/model/sam3_base_predictor.py)

## Runtime and checkpoint prerequisites

The official README requires Python ≥3.12, PyTorch ≥2.7 and CUDA ≥12.6. Its example
installs PyTorch 2.10.0 with CUDA 12.8. The package additionally declares
`numpy>=1.26,<2`. Use a separate compatible environment; our existing local
NumPy 2.x/MPS environment is not treated as compatible.
[Official installation](https://github.com/facebookresearch/sam3/blob/f1647e8d0c1d280991f4d1b7d974c2704537fe66/README.md#installation),
[dependency declaration](https://github.com/facebookresearch/sam3/blob/f1647e8d0c1d280991f4d1b7d974c2704537fe66/pyproject.toml)

This adapter targets the Linux CUDA cluster, verifies CUDA bfloat16 support and
checks the runtime's required modules, including the source's `pkg_resources`
import. No undocumented minimum VRAM number is invented. Optional FlashAttention
3 and compilation are disabled explicitly using the reviewed official flags;
there is no CPU/MPS inference fallback. The actual environment, checkpoint
compatibility and memory use still need a first GPU smoke test.

Once approved weights have been downloaded through normal user authentication,
place a provenance JSON next to the local checkpoint:

```json
{
  "model_repository": "facebook/sam3.1",
  "model_revision": "FULL_RESOLVED_40_CHARACTER_HUGGING_FACE_COMMIT",
  "checkpoint_file": "sam3.1_multiplex.pt",
  "checkpoint_sha256": "ACTUAL_64_CHARACTER_DOWNLOADED_FILE_SHA256"
}
```

The revision and file hash must come from the download operation, not guessed
values. Preflight checks the local hash and version against this manifest. It
does not query the account or independently authenticate the manifest's origin.
No token is accepted as an adapter argument. The run path disables Hub implicit
tokens and automatic checkpoint downloads and passes the local checkpoint and
the repository's tokenizer explicitly.

## Input and output contracts

Existing portable `sam3d-inputs-v1/OBJECT/input.json` bundles can supply the
full-frame RGB, image hash, object/concept ID, source-video hash, original PTS,
sample index and time domain. An existing binary mask is optional and is only a
comparison reference. Its pixel coordinates, hash and manual/model prompt
provenance are validated; it is never passed as a segmentation seed. The new
text prompt and its origin are separately required. Existing discovery output
can supply a short concept such as `earbud`; this adapter does not invent labels
or silently add manual boxes.

After genuine model inference, the adapter preserves every returned instance's
binary mask, model ID, probability and normalized box, plus source-pixel RGBA
cutouts and a diagnostic overlay. Masks must have the original source dimensions.
Returned IDs are not declared physical identity across frames. Probabilities and
overlap with an earlier SAM mask are not reported as measured accuracy. Zero
instances is a legitimate model result; missing/invalid outputs fail instead of
creating a replacement mask.

The upstream builder can load state dictionaries non-strictly. Before image
inference, this adapter additionally requires full final-model key/shape coverage
from the local checkpoint. Any partial, incompatible or legacy-remapped
checkpoint stops before segmentation rather than accepting random parameters.
This extra check is intentionally unverified against the gated checkpoint until
the first smoke test; an incompatibility requires review, not a silent fallback.

## Commands and actual checks

Run non-inferencing preflight with a fresh report path:

```sh
.venv-sam/bin/python scripts/sam31_segment.py preflight \
  --input-manifest data/recording-144929-1fps-v3/sam3d-inputs-v1/small_object_a/input.json \
  --repo models/sam3 \
  --prompt earbud --prompt-origin model_generated_prompt \
  --output /private/tmp/metahack-sam31-preflight-FRESH.json
```

An actual local preflight was run successfully as a prerequisite check. It
validated the real upright portrait input and pinned code, then exited 2 with
`prerequisites_missing`: no checkpoint manifest, Darwin rather than the target
Linux environment, incompatible NumPy 2.x, missing `timm`/`ftfy`/`pkg_resources`,
no CUDA ≥12.6 build and no visible CUDA GPU. It correctly records
`learned_inference_executed: false`. The diagnostic report is
`/private/tmp/metahack-sam31-preflight-20261028-v1.json`.

In the future prepared GPU environment, use that environment's Python and add
`--checkpoint-manifest /LOCAL/APPROVED/WEIGHTS/checkpoint.json`. Replace
`preflight` with `run` and choose a fresh output directory only after preflight
passes. A passing preflight means ready for the first model smoke test, not that
the model has already run or produced useful masks.

Nine contract tests pass; Python compilation passes:

```sh
.venv-sam/bin/python -m unittest discover -s tests -p test_sam31_contract.py
```

Tests cover original pixels/PTS, optional reference masks, hash/coordinate/EXIF
rejection, exact checkpoint-version provenance, strict checkpoint coverage,
actual-output preservation, legitimate empty detections and rejection of
missing/soft/resized outputs. All model outputs in unit tests are explicitly
synthetic fixtures; no test represents SAM 3.1 quality or GPU inference.
