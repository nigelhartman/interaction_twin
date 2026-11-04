/** Read-only camera comparison from ordinary video. No model or job endpoint is called. */
export const AUTOMATIC_CAMERA_RECORDS = Object.freeze([Object.freeze({"case_id":"observed-geometry-20261031-dense-camera-input-v2","track_id":"camera_body","source_sha256":"15531c5b696b8f13e6e2bcf524c0e9f73048caa2a33aeb60536a5852dbc566ac","report":{"path":"data/automatic-camera-20261101/camera-body-display-v1/ui-report.json","sha256":"a04a2d7134c9831885cd26b782dbf77beb38be7ecbd3d85f6a34788376e9271b"}})]);
const HASH = /^[0-9a-f]{64}$/;
const CASE = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,220}$/;
const STATUSES = new Set(['ambiguous', 'unsupported', 'candidate_only']);
const SPLITS = new Set(['train', 'reserved']);
const decoder = new TextDecoder();

function requireThat(test, message) { if (!test) throw new Error(message); }
function textValue(value, label, limit = 1500) {
  requireThat(typeof value === 'string' && value.length > 0 && value.length <= limit, `Invalid ${label}`);
  return value;
}
function nonnegative(value, label) { requireThat(value === null || (typeof value === 'number' && Number.isFinite(value) && value >= 0), `Invalid ${label}`); }
function count(value, label) { requireThat(Number.isInteger(value) && value >= 0 && value <= 10000000, `Invalid ${label}`); }
export function evidenceURL(path) {
  requireThat(typeof path === 'string' && /^data\/[A-Za-z0-9_.-]+(?:\/[A-Za-z0-9_.-]+)*$/.test(path) && !path.split('/').some(p => p === '.' || p === '..' || p.startsWith('.')), 'Unexpected saved camera evidence path');
  return '/files/' + path;
}
function identity(value, label) {
  requireThat(value && typeof value === 'object' && !Array.isArray(value), `Missing ${label}`);
  evidenceURL(value.path); requireThat(HASH.test(value.sha256 || ''), `Missing ${label} hash`);
}
function validateBinding(value) {
  requireThat(value && CASE.test(value.caseId || '') && HASH.test(value.sourceSha256 || '') && CASE.test(value.trackId || ''), 'Choose a source-bound component to inspect its camera comparison');
  return {caseId:value.caseId, sourceSha256:value.sourceSha256, trackId:value.trackId};
}
function sameBinding(record, binding) { return record.case_id === binding.caseId && record.track_id === binding.trackId && record.source_sha256 === binding.sourceSha256; }
export function validateCameraReport(report, binding) {
  validateBinding(binding);
  requireThat(report?.format === 'automatic-camera-comparison-ui' && report.version === 1, 'Unsupported camera comparison report');
  requireThat(report.case_id === binding.caseId && report.track_id === binding.trackId && report.source?.sha256 === binding.sourceSha256, 'Camera comparison belongs to another source or component');
  identity(report.source, 'source identity');
  requireThat(report.physical_accuracy_verified === false && report.object_metric_scale_established === false && report.applied_to_geometry === false && report.new_api_calls === 0, 'Camera comparison exceeds its diagnostic scope');
  requireThat(Array.isArray(report.image_size) && report.image_size.length === 2 && report.image_size.every(v => Number.isInteger(v) && v > 0 && v <= 32768), 'Missing native image dimensions');
  requireThat(report.outcome && STATUSES.has(report.outcome.status), 'Unsupported camera comparison outcome');
  textValue(report.outcome.title, 'outcome title', 300); textValue(report.outcome.summary, 'outcome summary', 3000);
  requireThat(Array.isArray(report.hypotheses) && report.hypotheses.length > 0 && report.hypotheses.length <= 8, 'Missing saved camera hypotheses');
  const ids = new Set();
  for (const h of report.hypotheses) {
    requireThat(CASE.test(h.id || '') && !ids.has(h.id), 'Duplicate or invalid camera hypothesis'); ids.add(h.id);
    for (const field of ['label', 'method', 'status', 'reason']) textValue(h[field], `hypothesis ${field}`, 3000);
    if (h.viewer !== undefined && h.viewer !== null) identity(h.viewer, 'hypothesis viewer');
    requireThat(h.focal_px === null || (Array.isArray(h.focal_px) && h.focal_px.length === 2 && h.focal_px.every(v => typeof v === 'number' && Number.isFinite(v) && v > 0 && v < 1e7)), 'Invalid estimated focal length');
    for (const split of ['training', 'reserved']) {
      const m=h[split]; requireThat(m && typeof m === 'object', 'Missing saved camera check metrics');
      nonnegative(m.median_px, `${split} median`); nonnegative(m.p90_px, `${split} p90`); count(m.views, `${split} view count`); count(m.observations, `${split} observation count`);
      requireThat(m.observations > 0 || (m.median_px === null && m.p90_px === null), 'Empty camera check has a residual');
      if (m.passing_views !== undefined && m.passing_views !== null) { count(m.passing_views, 'passing reserved view count'); requireThat(m.passing_views <= m.views, 'Passing view count exceeds scored views'); }
    }
  }
  requireThat(Array.isArray(report.frames) && report.frames.length <= 240, 'Invalid camera evidence frame list');
  const frameIds = new Set();
  for (const frame of report.frames) {
    requireThat(CASE.test(String(frame.id)) && !frameIds.has(String(frame.id)), 'Duplicate or invalid camera evidence frame'); frameIds.add(String(frame.id));
    requireThat(typeof frame.pts_s === 'number' && Number.isFinite(frame.pts_s) && frame.pts_s >= 0, 'Invalid source frame time');
    identity(frame, 'source frame'); requireThat(/\.(jpg|jpeg|png)$/.test(frame.path), 'Unsupported source frame image');
    requireThat(SPLITS.has(frame.split), 'Missing frozen frame split'); textValue(frame.role, 'frame role', 1000);
  }
  requireThat(Array.isArray(report.limitations) && report.limitations.length > 0 && report.limitations.length <= 30, 'Missing camera uncertainty');
  report.limitations.forEach(v => textValue(v, 'camera limitation', 2000));
  identity(report.provenance?.report, 'original comparison report');
  return report;
}
export function validateOriginalComparison(original, report, binding) {
  requireThat(original?.format === 'automatic-camera-comparison' && original.version === 1, 'Unsupported original camera comparison');
  requireThat(original.case_id === binding.caseId && original.track_id === binding.trackId && original.source?.sha256 === binding.sourceSha256 && original.source?.path === report.source.path, 'Original comparison belongs to another source or component');
  for (const field of ['camera_calibrated','physical_shape_verified','physical_scale_known','production_camera_changed','manual_geometry_or_poses_used']) requireThat(original[field] === false, 'Original camera comparison exceeds its diagnostic scope');
  requireThat(original.new_api_calls === 0 && HASH.test(original.input_protocol_sha256 || ''), 'Original comparison has no frozen local protocol');
  requireThat(Array.isArray(original.source_frames) && original.source_frames.length === report.frames.length, 'Original source frame list differs from the viewer evidence');
  const frames = new Map(original.source_frames.map(f=>[String(f.id), f]));
  requireThat(frames.size === report.frames.length, 'Original source frame identities are duplicated');
  for (const f of report.frames) {
    const saved=frames.get(String(f.id));
    requireThat(saved && saved.image_url === evidenceURL(f.path) && saved.image_sha256 === f.sha256 && Math.abs(saved.timestamp_s-f.pts_s) <= 1e-7 && (saved.split === 'heldout' ? 'reserved' : saved.split) === f.split, 'Original source frame hash, timestamp or path differs');
  }
  for (const h of report.hypotheses) {
    const saved = original.hypotheses?.find(x=>x.id===h.id);
    requireThat(saved && (h.viewer == null ? saved.viewer == null : saved.viewer?.path===h.viewer.path && saved.viewer?.sha256===h.viewer.sha256), 'Original hypothesis viewer identity differs');
    if (h.viewer) {
      requireThat(Array.isArray(saved.K) && saved.K.length===3 && saved.K.every(row=>Array.isArray(row)&&row.length===3&&row.every(v=>typeof v==='number'&&Number.isFinite(v))), 'Original camera matrix is missing');
      requireThat(h.focal_px && Math.abs(saved.K[0][0]-h.focal_px[0])<=1e-8 && Math.abs(saved.K[1][1]-h.focal_px[1])<=1e-8, 'Viewer focal length differs from original camera');
    }
  }
  return original;
}
export function validateHypothesisViewer(viewer, hypothesis, original, report) {
  requireThat(viewer?.format === 'observed-geometry-sparse-experiment' && viewer.kind === 'automatic_camera_hypothesis' && viewer.status === 'automatic_camera_diagnostic_only', 'Unsupported camera hypothesis geometry');
  requireThat(viewer.input_protocol_sha256 === original.input_protocol_sha256 && viewer.new_api_calls === 0 && viewer.prior_geometry_or_depth_used === false, 'Hypothesis geometry has another input protocol');
  for (const field of ['camera_calibrated','physical_shape_verified','physical_scale_known','production_camera_changed']) requireThat(viewer.calibration?.[field] === false, 'Hypothesis geometry exceeds its diagnostic scope');
  requireThat(viewer.calibration?.selected_production_camera_id === null && viewer.calibration?.reserved_used_for_selection === false, 'Hypothesis camera was selected outside the frozen training checks');
  const originalHypothesis = original.hypotheses.find(h=>h.id===hypothesis.id);
  requireThat(originalHypothesis?.K && Array.isArray(viewer.cameras) && viewer.cameras.length>0 && viewer.cameras.length<=report.frames.length, 'Hypothesis camera matrix is missing');
  for (const camera of viewer.cameras) requireThat(Array.isArray(camera.K) && camera.K.length===3 && camera.K.every((row,i)=>Array.isArray(row)&&row.length===3&&row.every((v,j)=>typeof v==='number'&&Number.isFinite(v)&&Math.abs(v-originalHypothesis.K[i][j])<=1e-8)), 'Hypothesis geometry uses another camera matrix');
  requireThat(Array.isArray(viewer.frames) && viewer.frames.length === report.frames.length, 'Hypothesis geometry has another frame set');
  const frames = new Map(viewer.frames.map(f=>[String(f.id),f]));
  requireThat(frames.size === report.frames.length, 'Hypothesis geometry duplicates source frames');
  for (const f of report.frames) {
    const saved=frames.get(String(f.id));
    requireThat(saved && saved.image_url===evidenceURL(f.path) && Math.abs(saved.timestamp_s-f.pts_s)<=1e-7 && saved.width===report.image_size[0] && saved.height===report.image_size[1], 'Hypothesis geometry source timestamp, path or pixels differ');
  }
  return viewer;
}
export async function sha256(bytes) {
  return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), x => x.toString(16).padStart(2, '0')).join('');
}
async function readVerified(record, fetchImpl, cap = 24 * 1024 * 1024) {
  identity(record, 'saved evidence');
  const response = await fetchImpl(evidenceURL(record.path), {cache:'no-store', method:'GET'});
  requireThat(response.ok, 'Saved camera evidence is unavailable');
  const bytes = await response.arrayBuffer();
  requireThat(bytes.byteLength > 0 && bytes.byteLength <= cap, 'Saved camera evidence is empty or exceeds the viewer limit');
  requireThat(await sha256(bytes) === record.sha256, 'Saved camera evidence changed; comparison remains unavailable');
  return bytes;
}

/** Controller is separately testable with hostile response ordering; commits are atomic. */
export function createAutomaticCameraController({records = AUTOMATIC_CAMERA_RECORDS, fetchImpl = (...args) => fetch(...args), commit = () => {}} = {}) {
  let epoch = 0, frameEpoch = 0, inspectionEpoch = 0, current = null, disposed = false;
  const emit = state => { if (!disposed) commit(state); };
  async function setBinding(value) {
    const mine = ++epoch; ++frameEpoch; ++inspectionEpoch; current = null;
    emit({status:'loading', report:null, frame:null, message:'Reading this component’s own camera evidence…'});
    let binding;
    try { binding = validateBinding(value); }
    catch (error) { emit({status:'unsupported', report:null, frame:null, message:error.message}); return; }
    const rows = records.filter(r => sameBinding(r, binding));
    if (rows.length !== 1) { emit({status:'unsupported', report:null, frame:null, binding, message: rows.length ? 'Conflicting camera records; no comparison is substituted.' : 'No saved automatic-camera comparison for this exact source and component yet.'}); return; }
    try {
      const record = rows[0]; identity(record.report, 'comparison sidecar');
      const bytes = await readVerified(record.report, fetchImpl);
      if (disposed || mine !== epoch) return;
      const report = validateCameraReport(JSON.parse(decoder.decode(bytes)), binding);
      const originalBytes = await readVerified(report.provenance.report, fetchImpl);
      if (disposed || mine !== epoch) return;
      const original = validateOriginalComparison(JSON.parse(decoder.decode(originalBytes)), report, binding);
      current = {binding, report, bytes, originalBytes, original, reportIdentity:record.report};
      emit({status:'ready', ...current, frame:null});
    } catch (error) { if (!disposed && mine === epoch) { current = null; emit({status:'error', report:null, frame:null, binding, message:error.message}); } }
  }
  async function selectFrame(id) {
    const mine = ++frameEpoch, selectionEpoch = epoch, selected = current;
    if (!selected || disposed) return;
    const frame = selected.report.frames.find(f => String(f.id) === String(id));
    emit({status:'ready', ...selected, frame:null, frameLoading:true, frameId:String(id)});
    if (!frame) { emit({status:'ready', ...selected, frame:null, frameError:'This source frame is absent from the selected comparison.'}); return; }
    try {
      const bytes = await readVerified(frame, fetchImpl, 12 * 1024 * 1024);
      if (disposed || mine !== frameEpoch || selectionEpoch !== epoch || current !== selected) return;
      emit({status:'ready', ...selected, frame:{...frame, bytes}});
    } catch (error) { if (!disposed && mine === frameEpoch && selectionEpoch === epoch && current === selected) emit({status:'ready', ...selected, frame:null, frameError:error.message}); }
  }
  async function inspectHypothesis(id) {
    const mine=++inspectionEpoch, selectionEpoch=epoch, selected=current;
    if (!selected || disposed) return null;
    const hypothesis=selected.report.hypotheses.find(h=>h.id===id);
    requireThat(hypothesis?.viewer, 'No saved geometry for this camera hypothesis');
    const bytes=await readVerified(hypothesis.viewer,fetchImpl);
    if (disposed || mine!==inspectionEpoch || selectionEpoch!==epoch || current!==selected) return null;
    const data=validateHypothesisViewer(JSON.parse(decoder.decode(bytes)),hypothesis,selected.original,selected.report);
    // Validate the actual source pixels before handing a viewer to another component.
    for (let i=0;i<selected.report.frames.length;i+=4) {
      await Promise.all(selected.report.frames.slice(i,i+4).map(frame=>readVerified(frame,fetchImpl,12*1024*1024)));
      if (disposed || mine!==inspectionEpoch || selectionEpoch!==epoch || current!==selected) return null;
    }
    return {id:hypothesis.id,label:hypothesis.label,diagnosticOnly:true,viewer:{...hypothesis.viewer,url:evidenceURL(hypothesis.viewer.path),bytes,data},sourceBinding:{...selected.binding}};
  }
  return {setBinding, selectFrame, inspectHypothesis, dispose() { disposed = true; ++epoch; ++frameEpoch; ++inspectionEpoch; current = null; }};
}
/** Resolve a reloadable diagnostic link through the same verified, read-only boundary. */
export async function resolveAutomaticCameraHypothesis({caseId,trackId,sourceSha256,hypothesisId} = {}, options = {}) {
  const binding=validateBinding({caseId,trackId,sourceSha256});
  requireThat(CASE.test(hypothesisId || ''), 'A saved camera hypothesis identifier is required');
  let latest=null;
  const controller=createAutomaticCameraController({records:options.records || AUTOMATIC_CAMERA_RECORDS,fetchImpl:options.fetchImpl || ((...args)=>fetch(...args)),commit:state=>{latest=state;}});
  try {
    await controller.setBinding(binding);
    requireThat(latest?.status==='ready', latest?.message || 'Camera comparison is unavailable for this exact source');
    const result=await controller.inspectHypothesis(hypothesisId);
    requireThat(result && result.diagnosticOnly===true, 'Verified camera hypothesis is unavailable');
    return result;
  } finally { controller.dispose(); }
}
function element(tag, text, className) { const e=document.createElement(tag); if (text!==undefined) e.textContent=String(text); if(className)e.className=className; return e; }
const fixed = (value, precision=2) => typeof value==='number' && Number.isFinite(value) ? value.toFixed(precision) : '—';
const metricText = value => value === null ? '—' : `${fixed(value)} px`;
const statusLabel = value => ({converged_TRAIN_fit:'Converged training fit · lens unverified',bounded_fit_not_converged:'Evaluation limit reached · not converged',ambiguous_not_selected:'Ambiguous · preserved diagnostic'})[value] || value.replaceAll('_',' ');
const methodLabel = value => ({assumed:'Initial camera from image dimensions',learned_intrinsics_prior:'Camera prior predicted from existing footage',previous_ambiguous_diagnostic:'Earlier mathematical focal estimate'})[value] || value.replaceAll('_',' ');

export function createAutomaticCameraPanel(container, options = {}) {
  if (!container) throw new Error('A camera comparison mount is required');
  container.classList.add('automatic-camera');
  container.setAttribute('aria-label','Automatic camera comparison from ordinary video');
  const heading=element('div',undefined,'ac-heading'), intro=element('div');
  intro.append(element('div','CAMERA FROM THE FOOTAGE','ac-eyebrow'),element('h2','One clip. Several possible cameras.'),element('p','Estimate the lens from the video you already have. Compare an assumed camera, a learned prior and mathematical refinement against the same saved image evidence.'));
  heading.append(intro,element('span','SAVED RESULTS · NO CREDIT USE','ac-local'));
  const body=element('div',undefined,'ac-body'), footer=element('div','These are camera hypotheses, not measured calibration. They do not establish metric scale, hidden surfaces, assembly or physical simulation. Opening this panel never starts inference.','ac-foot');
  container.replaceChildren(heading,body,footer);
  let urls=[], live=true;
  const revoke = () => { for (const url of urls) URL.revokeObjectURL(url); urls=[]; };
  const blobURL=(bytes,type)=>{const u=URL.createObjectURL(new Blob([bytes],{type}));urls.push(u);return u;};
  const link=(label,url,downloadName)=>{const a=element('a',label,'ac-button');a.href=url;if(downloadName)a.download=downloadName;return a;};
  const controller=createAutomaticCameraController({...options,commit:state=>{
    if (!live) return;
    const restoreFrameFocus = container.contains(document.activeElement) && document.activeElement?.classList.contains('ac-frame-select');
    revoke(); body.replaceChildren();
    const banner=element('div',undefined,'ac-status'+(['ready','error'].includes(state.status)?state.status==='error'?' ac-error':' ac-caution':''));
    if(state.status!=='ready'){
      banner.setAttribute('role',state.status==='error'?'alert':'status');
      banner.append(element('strong',state.status==='loading'?'Checking the selected source…':state.status==='error'?'Comparison unavailable':'Camera evidence stays with its source'),element('p',state.message||'Select a source-bound component to inspect the saved comparison.'));
      body.append(banner); return;
    }
    const r=state.report;
    const times=r.frames.map(frame=>frame.pts_s),windowText=times.length?`${fixed(Math.min(...times),3)}–${fixed(Math.max(...times),3)} s`:'no source frames';
    banner.append(element('div',`SEPARATE FROZEN STUDY · ${r.frames.length} source ${r.frames.length===1?'frame':'frames'} · ${windowText}`,'ac-study-scope'),element('strong',r.outcome.title),element('p',r.outcome.summary),element('p','These measurements belong to this frozen camera study. A selected reconstruction may use a different frame subset and camera; its own measurements remain in the reconstruction viewer.','ac-study-note'));body.append(banner);
    const flow=element('div',undefined,'ac-flow');for(const [i,t] of ['Original video pixels','Frozen image matches','Camera hypotheses','Reserved image checks'].entries()){if(i)flow.append(element('i','→'));flow.append(element('span',t));}body.append(flow);
    const grid=element('div',undefined,'ac-grid');
    for(const [i,h]of r.hypotheses.entries()){
      const card=element('article',undefined,'ac-card');card.append(element('span',String(i+1).padStart(2,'0'),'ac-number'),element('h3',h.label),element('div',methodLabel(h.method),'ac-method'));
      card.append(element('div',statusLabel(h.status),'ac-fit-status'));
      const focal=element('div',undefined,'ac-focal');focal.append(element('span',h.focal_px?fixed(h.focal_px[0],1):'—'),element('small',h.focal_px?'px · horizontal focal':'no focal estimate'));card.append(focal);
      card.append(element('div',h.focal_px?`Vertical focal ${fixed(h.focal_px[1],1)} px · ${r.image_size.join(' × ')} native image`:'No camera values are substituted.','ac-label'),element('span','INFERRED CAMERA PARAMETERS','ac-inferred'));
      const measures=element('div',undefined,'ac-measure');for(const [label,m]of [['Training image fit',h.training],['Reserved image score',h.reserved]]){const cell=element('div');cell.append(element('strong',metricText(m.median_px)),element('span',label),element('span',`${m.observations} scored observations · ${m.views} views`));measures.append(cell);}card.append(measures);
      if(Number.isInteger(h.reserved.passing_views))card.append(element('div',`${h.reserved.passing_views} / ${h.reserved.views} reserved views pass the frozen image checks`,'ac-pass-count'));
      card.append(element('p',h.reason));
      if(h.viewer && options.onInspectHypothesis){
        const inspect=element('button','Inspect this hypothesis ↗','ac-inspect-hypothesis');
        inspect.onclick=async()=>{
          const buttons=[...container.querySelectorAll('.ac-inspect-hypothesis')];buttons.forEach(b=>b.disabled=true);inspect.textContent='Verifying geometry + source…';
          try{const result=await controller.inspectHypothesis(h.id);if(result)await options.onInspectHypothesis(result);}
          catch(error){if(inspect.isConnected){card.append(element('p',error.message,'ac-inspection-error'));}}
          finally{if(inspect.isConnected){buttons.filter(b=>b.isConnected).forEach(b=>b.disabled=false);inspect.textContent='Inspect this hypothesis ↗';}}
        };card.append(inspect);
      }
      grid.append(card);
    }body.append(grid,element('p','Lower pixel error means closer image agreement under that hypothesis. It does not prove the lens or 3D shape is physically correct. Reserved observations do not tune camera parameters.','ac-score-note'));
    const evidence=element('section',undefined,'ac-source-grid'), stage=element('div',undefined,'ac-frame-stage'), panel=element('div',undefined,'ac-frame-panel');
    if(state.frame){const img=element('img');img.src=blobURL(state.frame.bytes,/\.png$/.test(state.frame.path)?'image/png':'image/jpeg');img.alt=`Original source frame ${state.frame.id} at ${fixed(state.frame.pts_s,3)} seconds`;stage.append(img);}else stage.append(element('div',state.frameLoading?'Verifying the exact source frame…':state.frameError||'Choose a saved frame to inspect the original video pixels.','ac-frame-empty'));
    panel.append(element('h3','Trace every check to the video'),element('p','Original frame pixels are observed evidence. The camera parameters above are inferred explanations of their matches.'));
    const select=element('select',undefined,'ac-frame-select');select.setAttribute('aria-label','Camera comparison source frame');select.append(new Option('Choose an exact saved frame',''));
    for(const f of r.frames){select.append(new Option(`${fixed(f.pts_s,3)} s · ${f.split==='reserved'?'Reserved':'Training'} · ${f.id}`,String(f.id)));}select.value=state.frame?.id||state.frameId||'';select.disabled=!r.frames.length;select.onchange=()=>{if(select.value)controller.selectFrame(select.value);};panel.append(select);
    if(state.frame){const f=state.frame;panel.append(element('span',f.split==='reserved'?'RESERVED FRAME':'TRAINING FRAME','ac-frame-split'+(f.split==='reserved'?' ac-reserved':'')),element('p',f.role),element('div',`SHA-256 ${f.sha256}`,'ac-hash'));
      const links=element('div',undefined,'ac-links');links.append(link('Original frame ↓',stage.querySelector('img').src,`camera-evidence-${f.id}.jpg`));
      if(options.onInspectFrame){
        const b=element('button','Inspect in reconstruction ↗'),callbackError=element('p','','ac-frame-callback-error');
        callbackError.hidden=true;callbackError.setAttribute('role','status');callbackError.setAttribute('aria-live','polite');callbackError.setAttribute('aria-atomic','true');
        b.onclick=async()=>{
          b.disabled=true;callbackError.hidden=true;callbackError.textContent='';
          try{await options.onInspectFrame({caseId:r.case_id,trackId:r.track_id,sourceSha256:r.source.sha256,frameId:f.id,ptsS:f.pts_s});}
          catch(error){if(live&&b.isConnected){callbackError.textContent=error?.message||'This exact source frame is unavailable in the selected reconstruction.';callbackError.hidden=false;}}
          finally{if(live&&b.isConnected)b.disabled=false;}
        };
        links.append(b);panel.append(callbackError);
      }panel.append(links);
    }evidence.append(stage,panel);body.append(evidence);
    const explanation=element('div',undefined,'ac-explanation');const observed=element('article');observed.append(element('h3','What is observed'),element('p','Original frame pixels and timestamps. SAM masks and tracked matches are automatically estimated from those pixels; their frozen training or reserved roles stay explicit.'));const uncertainty=element('article');uncertainty.append(element('h3','What remains uncertain'),element('p',r.limitations.join(' ')));explanation.append(observed,uncertainty);body.append(explanation);
    const details=element('details',undefined,'ac-source');details.append(element('summary','Source identity, full metrics and provenance'));const dl=element('dl');for(const[label,value]of [['Component',r.track_id],['Project',r.case_id],['Source video',r.source.path],['Source SHA-256',r.source.sha256],['Comparison SHA-256',state.reportIdentity.sha256]])dl.append(element('dt',label),element('dd',value));details.append(dl);
    const all=element('dl');for(const h of r.hypotheses)for(const [label,m]of [['Training',h.training],['Reserved',h.reserved]]){all.append(element('dt',h.label+' · '+label),element('dd',`Median ${metricText(m.median_px)}; 90th percentile ${metricText(m.p90_px)}; ${m.observations} observations in ${m.views} views.`));}details.append(all);
    const downloads=element('div',undefined,'ac-links');downloads.append(link('Full comparison JSON ↓',blobURL(state.originalBytes,'application/json'),'automatic-camera-comparison.json'),link('Viewer evidence JSON ↓',blobURL(state.bytes,'application/json'),'automatic-camera-viewer-evidence.json'));details.append(downloads);body.append(details);
    if (restoreFrameFocus) select.focus({preventScroll:true});
    if (!state.frame && !state.frameLoading && !state.frameError && !state.frameId && r.frames.length) controller.selectFrame((r.frames.find(f=>f.split==='reserved') || r.frames[0]).id);
  }});
  controller.setBinding(null);
  return {setBinding:controller.setBinding,dispose(){live=false;controller.dispose();revoke();container.replaceChildren();}};
}
