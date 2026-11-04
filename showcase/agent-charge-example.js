/* Exact saved public evidence only. No MCP, model or robot execution. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const files = [
    ['task.json', '5d88dd1cab5b0b4a8d1757cc094bf30a98cc106c60d8d94dbad55c3c19c4200f', 30993],
    ['trace.json', '3f97b9aa92d04184be5582251137778cb21274dc8f5e3e7393d116f938b0b1ff', 27726],
    ['summary.json', '39b96fba15948203608fcc9fbb8229b94711092273dac4957ef62ca3e4500ce3', 565],
    ['muse-review.json', '8d2724ba8a081d1f2507c09858ac32e57ee79daa0dd219a150319e2cecf92a5b', 7370]
  ];
  const objectURLs = [];
  const element = (tag, text, className) => {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (className) node.className = className;
    return node;
  };
  function sourceURL(value) {
    if (typeof value !== 'string' || !/^\/files\/data\/[A-Za-z0-9_./-]+$/.test(value) || value.split('/').some(p => p === '..' || p === '.' || p.startsWith('.'))) throw Error('Invalid source reference');
    return "/interaction_twin/"+value.slice(1);
  }
  function libraryURL(source) {
    const expected = '/showcase/library.html?case=' + source.twin_id;
    if (!/^[A-Za-z0-9_-]+$/.test(source.twin_id) || source.library_url !== expected) throw Error('Invalid twin reference');
    return "/interaction_twin/"+expected.slice(1);
  }
  async function verifiedBytes(url, hash, exactBytes, maximum = 1024 * 1024) {
    const response = await fetch(url, {credentials: 'omit', cache: 'no-cache'});
    if (!response.ok) throw Error('Saved example unavailable');
    const raw = await response.arrayBuffer();
    if (!raw.byteLength || raw.byteLength > maximum || exactBytes && raw.byteLength !== exactBytes) throw Error('Saved evidence size changed');
    const digest = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', raw)), v => v.toString(16).padStart(2, '0')).join('');
    if (digest !== hash) throw Error('Saved evidence changed');
    return raw;
  }
  async function load() {
    const [task, trace, summary, review] = await Promise.all(files.map(async ([name, hash, size]) => JSON.parse(new TextDecoder().decode(await verifiedBytes('demos/charge-phone-v1/' + name, hash, size)))));
    if (task.format !== 'interaction-twin-cross-recording-task' || task.instruction !== 'Prepare a task to charge this phone.' || task.sources.length !== 2 || task.claims.robot_execution_available !== false || task.claims.charging_success_verified !== false || task.claims.inference_performed !== false || task.task_draft.sequence_kind !== 'evidence_linked_templates_not_robot_commands') throw Error('Unexpected task scope');
    if (trace.transport !== 'actual_official_sdk_subprocess_stdio' || trace.workflow !== 'deterministic_scripted_client_not_live_llm' || trace.calls.length !== 42 || summary.mcp_calls !== 42 || summary.task_sha256 !== files[0][1] || summary.trace_sha256 !== files[1][1] || summary.live_llm_agent !== false || summary.physical_success_verified !== false || trace.native_asset_bytes_verified.length !== summary.verified_native_models || trace.source_evidence_bytes_verified.length !== summary.verified_evidence_frames) throw Error('Unexpected execution trace');
    if (review.status !== 'validated_saved_muse_assessment' || review.review_scope !== 'text_metadata_only_no_images_or_meshes' || review.task_file_sha256 !== files[0][1] || review.trace_file_sha256 !== files[1][1] || review.result.overall_assessment !== 'conditional_task_draft_only' || review.robot_execution_available !== false) throw Error('Unexpected review scope');
    const phone = task.sources.find(s => s.role === 'phone');
    const charger = task.sources.find(s => s.role === 'charger');
    if (!phone || !charger) throw Error('Missing source twins');
    const cards = [
      {title: 'Phone', source: phone, interface: 'phone_port', part: 'phone_blue', note: 'Recording B · phone port', alt: 'Actual source frame showing the phone and its cable connector'},
      {title: 'USB-C cable', source: charger, interface: 'cable_free_end', part: 'power_cable', note: 'Recording A · free cable end', alt: 'Actual source frame showing the free end of the cable'},
      {title: 'Charger', source: charger, interface: 'adapter_port', part: 'adapter_late', note: 'Recording A · adapter port', alt: 'Actual source frame showing the charger and cable'}
    ];
    const prepared = await Promise.all(cards.map(async card => {
      const observation = card.source.interfaces.find(i => i.id === card.interface)?.evidence[0];
      const asset = card.source.assets.find(a => a.track_id === card.part);
      if (!observation || !asset || !trace.native_asset_bytes_verified.some(a => a.twin_id === card.source.twin_id && a.component_id === card.part && a.sha256 === asset.sha256)) throw Error('Missing exact asset evidence');
      const verifiedFrame = trace.source_evidence_bytes_verified.find(e => e.twin_id === card.source.twin_id && e.evidence_id === observation.id && e.sha256 === observation.frame_sha256);
      if (!verifiedFrame) throw Error('Frame not retrieved in saved workflow');
      return {...card, url: libraryURL(card.source), raw: await verifiedBytes(sourceURL(observation.frame_url), observation.frame_sha256, verifiedFrame.bytes), timestamp: observation.timestamp_s};
    }));
    const fragment = document.createDocumentFragment();
    for (const card of prepared) {
      const article = element('article', undefined, 'agent-example-asset');
      const anchor = element('a'); anchor.href = card.url; anchor.setAttribute('aria-label', 'Inspect ' + card.title + ' twin');
      const image = element('img');
      const url = URL.createObjectURL(new Blob([card.raw], {type: 'image/jpeg'})); objectURLs.push(url);
      image.src = url; image.alt = card.alt; image.width = 360; image.height = 240; image.decoding = 'async';
      const body = element('div', undefined, 'agent-example-asset-body');
      body.append(element('span', card.note, 'agent-example-source'), element('h3', card.title), element('span', 'Inspect twin ↗', 'agent-example-inspect'));
      anchor.append(image, body); article.append(anchor); fragment.append(article);
    }
    $('charge-example-assets').append(fragment);
    $('charge-task-summary').textContent = task.task_draft.steps.length + ' proposed steps · ' + task.connection_candidates.length + ' connection candidates · evidence from two recordings.';
    $('charge-trace-summary').textContent = 'An actual saved MCP client made ' + trace.calls.length + ' calls, retrieved ' + summary.verified_native_models + ' original model files and ' + summary.verified_evidence_frames + ' source frames, and checked their hashes. A deterministic compiler assembled the task draft.';
    $('charge-review-summary').textContent = 'A separate Meta Muse metadata critique kept the result as a conditional draft and flagged connector fit, cable identity and electrical checks.';
    $('charge-example-content').hidden = false;
    $('charge-example-status').textContent = '';
    $('charge-example-status').hidden = true;
  }
  const ready = load().catch(() => {
    $('charge-example-content').hidden = true;
    $('charge-example-status').textContent = 'The saved example could not be verified. You can still connect your agent using the line above.';
  });
  addEventListener('pagehide', () => objectURLs.forEach(url => URL.revokeObjectURL(url)));
  // A read-only completion handle lets the local QA harness verify failure
  // isolation without invoking a model or exposing hidden runtime state.
  globalThis.InteractionTwinChargeExample = {ready};
})();
