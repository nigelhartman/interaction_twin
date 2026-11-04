# Hosted SAM 3.1 adapter

`meta_sam_api.py` adds hosted **2D image segmentation and video tracking** without
changing the preserved local checkpoint adapter or silently replacing old
SAM2 results. The module owns no credentials, networking, automatic retries or
budget. Its caller must supply one budget-guarded `request(payload) -> dict`
callback and approve the media upload. This implementation was validated offline
with actual Meta parser software and synthetic response fixtures; that alone is
not a successful live model run or segmentation-quality measurement.

## Verified request contract

The endpoint is `POST https://api.meta.ai/v1/responses`, model `sam-3.1`, with the
normal Meta Bearer key. A single user message contains one short noun phrase and
one `input_image` or `input_video`. This version chooses documented non-streaming
mode so the caller can bound and account for one complete request. Streaming is
also officially supported and recommended upstream for longer clips, but is not
implemented by this transport-free adapter.

The prepared payload is:

```json
{
  "model": "sam-3.1",
  "stream": false,
  "input": [{"type": "message", "role": "user", "content": [
    {"type": "input_text", "text": "earbud"},
    {"type": "input_image", "image_url": "data:image/png;base64,..."}
  ]}],
  "metadata": {"mask_encoding": "one_bit", "include_confidence": "true"}
}
```

Video uses `{"type":"input_video","video_url":"data:video/mp4;base64,..."}`
or a previously uploaded `file_id`. Meta documents MP4/MOV with H.264, H.265 or
MPEG-4. Uploading through `POST /v1/files` with `purpose=user_data` is a separate
caller-owned operation; this module never silently uploads an extra copy.

Use one concept per concept call. Box prompts are supported through the separate
`segment_image_boxes()` and `segment_video_boxes()` helpers below. Point prompts
are an official capability but are not implemented here. The concept builder
continues rejecting token prompts and comma-separated concept lists.

Sources: [SAM overview](https://dev.meta.ai/docs/sam/overview),
[request and media rules](https://dev.meta.ai/docs/sam/segmenting),
[official API basics](https://dev.meta.ai/docs/cookbook/sam-api-basics).

### Automatically proposed box prompts

`segment_image_boxes(input_manifest, objects, output, request, prompt_origin=...)`
accepts observations such as `{"id": 7, "box_xyxy": [12.4, 30.1, 54.8, 77.2]}`.
The caller must supply model-generated native-pixel boxes and explicit model
provenance; this automatic route rejects manual/assistant annotation origins.
Coordinates are half-open. The helper preserves the original floats, rounds
left/top down and right/bottom up without clipping, then uses Meta's official
`build_object_prompt()` to encode inclusive wire right/bottom minus one. The
chosen nonnegative numeric object IDs return unchanged as strings, with no
one-based/zero-based ID offset.

`segment_video_boxes(video_path, objects, output, request, width=..., height=...,
frame_sources=..., prompt_origin=..., propagation_direction="both")` additionally
accepts `frame_index` per box. It is the zero-based index in the submitted clip,
mapped to original PTS through `frame_sources`. The same object ID may have boxes
at different frames; duplicate ID/frame seeds are rejected. Up to 16 distinct
objects and 128 box observations are allowed. `forward`, `backward` and `both`
are explicit tracking directions. Backward tracking requires an earliest prompt
after frame zero.

No concept phrase is mixed with box grammar. Returned unprompted IDs fail rather
than being assigned to components. Native outputs, original boxes, rounding,
request text and source timestamps remain available for inspection. A box from
another model can be wrong: successful prompt formatting does not establish
component accuracy or physical identity.

## Response parsing and source coordinates

The returned text is a special-token grammar, not a JSON array of masks. The
official `meta-sam-parser==0.1.0` package decodes both lossless and one-bit
base85 payloads. It is a small dependency-free Python package, separate from
the GPU SAM checkpoint/runtime. The adapter checks its reviewed version before
calling the supplied transport. Missing parser software stops before spending.

```text
<Nf>OBJECT_ID<|box;...|><|mask;...|>
```

The box uses source pixels and inclusive bottom/right wire coordinates. The
official parser normalizes these into half-open pixel bounds. The mask is a
box-local raster, so the adapter places it at that box in an otherwise empty
full-source-size binary mask. It validates original width/height, bounds,
decoded raster shape, finite source timestamps and unique frame/object pairs.
It does not stretch a returned mask to hide a coordinate mismatch.

Video records may skip frames or arrive out of order. `frame_sources[N]` must
explicitly map submitted frame N to the original recording's PTS, video hash and
sample index. No `N/fps` timestamp is invented. A caller preparing a reduced-fps
derivative must verify its encoded frame order, displayed dimensions and exact
original-PTS mapping before submitting it; the adapter does not itself probe,
decode, crop or resample a video. A video with no match on a frame emits no mask
for that frame. Missing masks are never filled with old SAM2 outputs.

Model IDs are retained as response-scoped visible tracks. They do not establish
physical identity across complete occlusion, entry/exit or independent requests.
Confidence is a model score, not measured accuracy. Old SAM2 mask plausibility
audits are not attached to these new masks.

Sources: [wire grammar and identity caveats](https://dev.meta.ai/docs/sam/reading-segmentation),
[official parser and raster conversions](https://dev.meta.ai/docs/sam/client-libraries),
[Meta-published parser distribution](https://pypi.org/project/meta-sam-parser/0.1.0/).

## Bounded output and failure behaviour

Each run requires a fresh output directory and calls the transport exactly once.
It preserves the raw Responses JSON before decoding, allowing a malformed or
incomplete paid response to be inspected without running it again. Failed,
refused, ambiguous, malformed, duplicate or unsupported coordinate results stop;
they do not create replacement masks. A completed empty lane is a valid
`completed_zero_matches` result.

Successful export contains `response.json`, `metadata.json` and binary
`instance-NNNNNN-mask.png` files. Metadata records source PTS, source/manifest
hashes, prompt origin, model/response ID, official parser version, reported usage,
model confidence, box coordinates and mask hashes. These files are candidate
perception evidence, not accepted geometry or robot-ready assets. Reference masks
in existing portable inputs are validated for provenance but are **not sent as
segmentation prompts**.

Local resource ceilings are 20 MiB inline media, 4096x4096 source pixels, 5000
mapped video frames and 20,000 object records. These are our prototype allowances,
not claims about Meta's service limits. Decode output is additionally bounded to
100 MiB of text. Longer inputs require an explicitly reviewed larger plan or
separate bounded clips, not automatic chunking/retry.

## Cost accounting

Official standard USD pricing checked on 30 October 2026:

| Request | Published cost |
| --- | --- |
| Image | $2.50 / 1000 images = $0.0025 per request |
| Video | $0.20 / 1000 frames = $0.0002 per frame |
| Video minimum | 50 billed frames per request = at least $0.01 |

Each separate concept is another request. A 1000-frame clip segmented for three
concepts is therefore about $0.60 at the published rate, before any extra runs.
SAM is not billed by output text tokens. The documented
`usage.video_frames_processed` field reports billed video frames. Preserve the
whole usage object, including any additional returned fields. Missing usage is
unknown, never evidence of zero cost: keep the reservation until reconciliation.

`estimate_cost()` supplies the deterministic published per-image/per-frame
estimate. The shared application ledger must reserve before network dispatch,
account for unsuccessful/uncertain attempts conservatively, and reconcile actual
usage. This module cannot enforce the user's total $100 allowance by itself or
control spending outside the app.

Source: [pricing and rate limits](https://dev.meta.ai/docs/pricing-rate-limits).

## Integration and small smoke test

Install the reviewed parser into the processing environment through ordinary
dependency management, without downloading SAM model checkpoints:

```sh
python -m pip install meta-sam-parser==0.1.0
```

The application can dispatch one existing timestamped source anchor:

```python
from pathlib import Path
from meta_sam_api import segment_image

# budget_guarded_request is supplied by the application. It owns authentication,
# reservation, network transport, timeout, response size and usage settlement.
result = segment_image(
    Path("data/recording-144929-1fps-v3/sam3d-inputs-v1/small_object_a/input.json"),
    "earbud",
    Path("data/NEW-UNUSED-SAM-API-ANCHOR-DIRECTORY"),
    budget_guarded_request,
    prompt_origin="previous_model_discovery_concept; new Meta segmentation",
)
```

Inspect the full-source-size mask on the original image. A plausible anchor does
not certify tracking or justify promoting old audits. Next use one small,
timestamp-mapped clip, inspect every returned frame and stable-ID interval, then
select new reconstruction inputs and rerun affected stages in fresh directories.

## What the key does not unlock

The official hosted Model API catalogue lists SAM 3.1 but does not document a
SAM 3D Objects reconstruction endpoint. That is an observation about the public
catalogue, not proof that no other service exists. Our SAM3D route still uses the
official open-weight repository: its setup requires Linux, an NVIDIA GPU with at
least 32 GB VRAM and authenticated access to the approved Hugging Face weights.
A working Meta API key does not satisfy those separate requirements.

Sources: [current hosted model catalogue](https://dev.meta.ai/docs/models),
[official SAM3D setup](https://github.com/facebookresearch/sam-3d-objects/blob/main/doc/setup.md).

## Offline verification performed

Nineteen tests pass using the actual official parser. They cover correct full-
image placement and source PTS, sparse/out-of-order video frames, valid zero
matches, optional confidence, reported/unknown usage, malformed payloads, wrong
dimensions, no resizing, duplicate records, failed/refused responses, exactly one
callback attempt, fresh-output protection and failure before spending when the
parser is unavailable. The masks are small published software fixtures. These
tests are not live inference or evidence that SAM3.1 improves our footage. Box
tests additionally exercise official prompt formatting, outward rounding,
unchanged IDs, multiple video seed frames, original PTS, invalid coordinates,
directions and unprompted response IDs.

```sh
python -m unittest discover -s tests -p test_meta_sam_api.py -v
```
