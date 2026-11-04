# Private Marketing Studio

Open the app, visit `/admin/`, and sign in with your owner password. The private desk has four sections: **Overview**, **Twins**, **Marketing**, and **Memory**. Uploads, reviews, campaign drafts and editing feedback are private. The public library changes only through its separate twin approval workflow.

First use: double-click **Open Private Studio.command** in the project folder. If owner access has not been configured, its Terminal asks you to choose and confirm your password with hidden input. The launcher then opens the private studio and reuses the existing app server when available. No default password has been created for you. The normal login persists for the session; the setup prompt runs only once.

## Make a campaign

1. Open **Marketing → New campaign** and give it a title.
2. Choose a model or collection, then select its available source media. Only declared, hash-verified project footage is offered. A new curated pack may show **rights needed**. Its compact panel enables the exact original recording only after you acknowledge that you have marketing rights. This does not authorize arbitrary URLs or unknown files; unavailable source recordings remain unavailable. Externally sourced material, including the camera review, is never automatically enabled or silently substituted.
3. Choose **Show a twin**, **Introduce a pack**, **Explain the pipeline**, or **Robotics workflow**. Robotics needs the actual browser-workbench recording.
4. Choose landscape **16:9**, portrait **9:16**, or square **1:1**, and **15**, **30**, or **45 seconds**. Add a brief and choose an approved soundtrack.
5. Leave **Use Muse to plan this draft** off for a local draft, or explicitly check it to let Muse interpret the brief through the guarded API. Select **Generate draft**. This saves a local draft and queues its rendering; the optional Muse task creates another revision. The saved storyboard, footage references and previous revisions remain available if a worker stops.
6. Review the film, storyboard and draft X/LinkedIn copy. The revision picker controls exactly which version you are viewing.

The initial local draft uses bounded project facts and editing templates. It does not claim that a model has interpreted your brief. Each scene retains its source reference, time range and grounded caption.

## Improve a revision

Each campaign lineage allows two refinement rounds. After they are used, start a new campaign for a new brief; the existing film can still be inspected, approved and downloaded.

Write a note against the selected revision. **Apply local edit** applies the explicit pacing, caption-density, scene-order and voice controls. The note remains saved editing guidance. Selecting **Remember as a brand preference** also makes that scoped preference available to future campaigns; it is not model training or new evidence about a twin.

**Improve with Muse** is a separate, explicit API action. Muse receives the brand guide, declared source references, allowed facts, current storyboard/copy and scoped feedback as text. It does not receive a new upload of your video. Separate proposer and critic prompts use the same model; they are not independent model verification. The existing usage guards remain in force.

A queued/running Muse task keeps the previous revision visible and polls saved status. Once the exact task produces a new source-bound revision, the desk selects it once. Later manual revision choices are respected. A failed or uncertain model attempt is preserved and is never automatically resent. Refresh the campaign to inspect its outcome. One explicit Muse action allows at most two editor/critic cycles (four text calls). Only a complete, schema-valid critic verdict of **revise** permits the single automatic correction. A transport failure, malformed response or uncertain charge stops the action; it never authorizes resending an earlier request. Open **Muse review** to read the saved critic issues and editing notes, including each completed cycle when available. These are model suggestions, not pixel review or physical acceptance.

A local presentation refresh can retain the exact Muse storyboard, copy and creative proof while changing the compositor layout. Its label is **Muse creative review · local presentation refresh** only when the source revision, record hashes and creative fields match. The original text review does not become a pixel review of the refreshed film, and a layout-only refresh does not consume a creative refinement round.

A failed **local video render** has an explicit retry button. The retry creates a separate attempt for the same revision and preserves the original failure. This is distinct from a paid model retry.

## Approve and download

Inspect the exact selected revision, then choose **Approve this revision → Prepare download → Download package**. Approval covers that revision and its completed renders only. The package includes the video, poster, captions, social copy, storyboard, brand and approval/source-provenance records.

Nothing posts to X, LinkedIn or another service. Publishing is a separate owner action. Marketing approval does not approve a private twin for the public object library, certify physical accuracy or grant rights to third-party material.

## Art direction: Cobalt Motion

Keep the existing cobalt mark, **Interaction Twin**, and **“Show it once. Keep how it works.”** Use quiet off-white surfaces, near-black text, cobalt accents, readable sentence-case headlines and one clear message per scene. Short opening/closing cards, restrained crossfades and contained footage keep the interaction visible in every aspect ratio.

The story is **real interaction → separated parts → connected description → reusable task**. Show real footage early. Robot clips must identify authored staging and unverified grasp/contact. Do not claim perfect reconstruction, physics-ready assets, successful hardware execution, trained robot policies or a new model release. The complete design reference is [marketing-art-direction.md](marketing-art-direction.md); the machine-readable guide is [marketing-brand.json](../config/marketing-brand.json).

## Music and licence records

The default is the project's original instrumental. To import another soundtrack, open **Music library & imports**. Use **Find music on Pixabay** to visit its website yourself; the app does not offer a fabricated music-search API or fetch arbitrary URLs.

Import an audio-only WAV, MP3, M4A, FLAC or OGG file up to **25 MiB**. Supply its title and contributor, choose Pixabay or your own original music, and acknowledge that you have the necessary rights. A Pixabay import also needs its original HTTPS track page. Record the available Content ID information and any licence/certificate reference in the optional note.

The private record retains the audio hash, source page, contributor, import time, acknowledgement and certificate/Content ID note. The app does not independently verify a certificate or promise that a track cannot trigger an automated claim. Preserve any original licence document yourself. If audio upload succeeds but metadata registration fails, the page retains the upload ID so retrying registration does not upload the file again. Only registered music IDs reach the renderer.

## Understand Overview

**Guard remaining**, **Estimated usage**, request counts and reported tokens come from local request accounting. They are not the live Meta credit balance. Uncertain requests retain their bounded holds; missing reported usage remains **Unavailable**, not zero. Check the Meta billing page yourself when reconciling credits.

Active processing and Current jobs include both twin processing and local marketing renders. The total remains unavailable if either source is missing, while each known subtotal stays visible. Render state and heartbeat are saved observations, not independent proof that an external worker is alive.

GPU allocation details are a cached, time-stamped read-only observation. **Check GPU now** explicitly refreshes it. Opening Overview or refreshing its local snapshot does not start SSH, inference or rendering. GPU utilization and memory remain **Not measured** when no device measurement was taken.

The **Twins** tab retains the existing upload, private inspection, bounded repair and owner approval workflow. Its ordinary capture profile accepts up to **40 seconds** and reconstructs up to **four components**. Marketing reuses approved project media; it does not run that reconstruction pipeline again.
