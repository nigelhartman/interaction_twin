// Saved, pinned analysis only. No POST, inference, simulator or autoplay path.
import {mountAssemblyEvidence} from './physics-assembly.js';
const INDEX_PATH = 'data/physics-validation-20261101/v2/index.json';
const INDEX_SHA256 = 'a11c13f610bd095e899eaec7b78132517078c4dba1d578b1c71d7b92e186096c';
const $ = (id) => document.getElementById(id);
const number = (v) => Number(v).toLocaleString('en-US');
const node = (tag, text, className) => {const item = document.createElement(tag); if (text !== undefined) item.textContent = text; if (className) item.className = className; return item;};
const HASH = /^[a-f0-9]{64}$/;
function fileUrl(path) {
  if (typeof path !== 'string' || !/^data\/[A-Za-z0-9_./-]+$/.test(path) || path.split('/').some((p) => !p || p.startsWith('.'))) throw new Error('Unsupported saved-artifact path');
  return `/files/${path}`;
}
async function verifiedJson(path, hash) {
  if (!HASH.test(hash)) throw new Error('Missing saved-artifact hash');
  const response = await fetch(fileUrl(path), {cache:'no-store'});
  if (!response.ok) throw new Error(`Saved artifact unavailable (${response.status})`);
  const bytes = await response.arrayBuffer();
  if (bytes.byteLength > 2 * 1024 * 1024) throw new Error('Saved JSON exceeds allowance');
  const actual = [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))].map((v) => v.toString(16).padStart(2,'0')).join('');
  if (actual !== hash) throw new Error('Saved-artifact identity changed; report not displayed');
  return {document:JSON.parse(new TextDecoder().decode(bytes)), bytes, hash, path};
}
function download(saved, filename) {
  const url = URL.createObjectURL(new Blob([saved.bytes], {type:'application/json'}));
  const link = node('a'); link.href = url; link.download = filename; document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function dl(container, pairs) {const list = node('dl'); for (const [key,value] of pairs) {const row=node('div'); row.append(node('dt',key),node('dd',value)); list.append(row);} container.append(list);}
function validateReport(report, fixture) {
  if (report.format !== 'interaction-twin-physics-validation-report' || report.version !== 1 || report.status !== 'physical_inputs_missing') throw new Error('Unsupported physical report');
  for (const key of ['simulation_ready','physical_accuracy_verified','actual_reconstructed_object_dynamics_run','contact_or_insertion_validated','robot_simulation_run','original_assets_modified']) if (report.claim_boundaries[key] !== false) throw new Error('Unsupported physical claim');
  if (report.claim_boundaries.new_api_calls !== 0 || fixture.fixture_only !== true || fixture.real_reconstructed_assets_used !== false) throw new Error('Fixture must remain separate from real assets');
  if (report.source_case.preflight.status !== 'blocked' || report.source_case.preflight.simulation_ready !== false || !Array.isArray(report.source_case.preflight.blocking_inputs)) throw new Error('Physical inputs must retain their blocked status');
  if (!Array.isArray(fixture.samples) || fixture.samples.length < 2 || fixture.samples.length > 2001) throw new Error('Invalid saved fixture sample count');
  for (const sample of fixture.samples) if (!Number.isFinite(sample.t_s) || !Array.isArray(sample.position_m) || sample.position_m.length !== 3 || !sample.position_m.every(Number.isFinite)) throw new Error('Non-finite saved fixture sample');
  for (const record of report.reconstruction_versions) if (record.simulation_ready !== false || record.collision_geometry_accepted !== false || record.physical_accuracy_verified !== false) throw new Error('Diagnostic geometry cannot become collision-ready');
}
function assetCard(id, label, mesh) {
  const card = node('article',undefined,`card${mesh?'':' no-asset'}`); card.append(node('h3',label),node('span',id,'component-id'));
  if (!mesh) {card.append(node('p','No generated asset'),node('p','The source tracklet exists. Geometry and physical properties remain absent.','small')); return card;}
  card.append(node('p',number(mesh.triangles),'asset-number'),node('span','native triangles','small'));
  dl(card,[['Vertices',number(mesh.vertices)],['Indexed shells',number(mesh.raw_index_connected_components)],['Boundary edges',number(mesh.raw_index_boundary_edge_count)],['Degenerate faces',number(mesh.degenerate_triangle_count)]]);
  card.append(node('p',mesh.raw_index_watertight?'Closed indexed surface · physical shape unverified':'Open indexed surface · diagnostic patches','small'));
  if (mesh.raw_index_connected_components>1) card.append(node('p','Several disconnected shells need inspection before collision preparation.','note'));
  const a=node('a','Native GLB ↗','asset-download'); a.href=fileUrl(mesh.artifact.path); a.setAttribute('download',''); a.style.display='inline-flex'; a.style.alignItems='center'; a.style.fontSize='12px'; card.append(a); return card;
}
function renderAssets(report) {
  const diagnostic = new Map(report.source_case.native_mesh_diagnostics.map((m) => [m.component_id,m]));
  for (const component of report.source_case.contract.components) $('part-cards').append(assetCard(component.id,component.label,diagnostic.get(component.id)));
  report.reconstruction_versions.forEach((version,i) => {const card=assetCard(version.component_id,version.kind==='observed'?'Observed camera patch':`Inferred camera layer ${i}`,version.native_mesh_diagnostics); card.append(node('p','Separate diagnostic version · not an accepted collider','note')); $('version-cards').append(card);});
}
function renderRequirements(report) {
  const components=report.source_case.contract.components;
  const groups=[['Measured scale',['meters_per_native_unit'],'A physical dimension and its uncertainty for each part.'],['Shared placement',['world_from_body_m','body_from_native_mesh'],'Common metric body frames and source calibration.'],['Collision geometry',['collision_geometry'],'Reviewed cavity-preserving colliders, with clearance evidence.'],['Mass and inertia',['mass_kg','center_of_mass_m','inertia_kg_m2'],'Mass distribution in a documented body frame.'],['Contact properties',['contact_material'],'Friction, restitution and the limits of the chosen model.'],['Joints and task validation',[],'Measured mechanisms, initial state and a frozen trial protocol.']];
  for (const [title,keys,description] of groups) {const card=node('div',undefined,'requirement'); card.append(node('strong',title),node('p',description)); const unknown=keys.length?components.filter((c)=>keys.some((k)=>c[k]===null)).length:components.length; card.append(node('span',keys.length?`${unknown} / ${components.length} components missing inputs`:'Assembly and trial inputs remain unknown')); if(title==='Shared placement'){const a=node('a','Review assembly evidence ↓','requirement-action');a.href='#assembly';card.append(a);} $('requirements').append(card);}
  const gaps=report.source_case.preflight.blocking_inputs; $('all-gaps-summary').textContent=`All ${gaps.length} missing-input records`;
  for (const gap of gaps) {const row=node('div',undefined,'gap-row'); row.append(node('strong',gap.component_id||'Shared trial'),node('code',gap.field),node('p',gap.reason)); $('all-gaps').append(row);}
}
const NS='http://www.w3.org/2000/svg';
function svgItem(parent,tag,attrs,text) {const item=document.createElementNS(NS,tag);for(const [key,value] of Object.entries(attrs))item.setAttribute(key,String(value));if(text!==undefined)item.textContent=text;parent.append(item);return item;}
function chart(svg, series, {maxY,unit,reference,limit}) {
  const box={left:75,right:776,top:40,bottom:265}; const maxX=2;
  const x=(v)=>box.left+v/maxX*(box.right-box.left),y=(v)=>box.bottom-v/maxY*(box.bottom-box.top);
  for(let i=0;i<=4;i++){const v=maxY*i/4;svgItem(svg,'line',{x1:box.left,x2:box.right,y1:y(v),y2:y(v),stroke:'#e5e8df','stroke-width':1});svgItem(svg,'text',{x:box.left-10,y:y(v)+5,'text-anchor':'end',fill:'#687166','font-size':16},Number(v.toFixed(2)).toString());}
  for(let i=0;i<=4;i++){const v=i/2;svgItem(svg,'text',{x:x(v),y:box.bottom+27,'text-anchor':'middle',fill:'#687166','font-size':16},`${v}s`);}
  svgItem(svg,'text',{x:box.left,y:26,fill:'#687166','font-size':14},unit);
  function path(points,color,dash){svgItem(svg,'path',{d:points.map(([a,b],i)=>`${i?'L':'M'}${x(a).toFixed(3)},${y(b).toFixed(3)}`).join(' '),fill:'none',stroke:color,'stroke-width':3,...(dash?{'stroke-dasharray':dash}:{})});}
  if(reference)path(reference,'#7d8fbd','7 5'); if(limit!==undefined)path([[0,limit],[maxX,limit]],'#973e2a','8 5'); path(series,'#496549');
}
function renderFixture(report,fixture) {
  $('engine-description').textContent=`Installed Blender ${fixture.engine.blender_version} executed ${number(fixture.samples.length)} saved frames in Bullet on the CPU. The synthetic 20 cm box, 1 kg mass and material settings are test definitions; none was assigned to a reconstructed component.`;
  const alert=$('solver-outcome'); alert.append(node('strong','Fixture failed one frozen check: lateral drift'),node('p',`${(fixture.horizontal_drift_m*1000).toFixed(1)} mm horizontal drift exceeds the unchanged 5 mm limit. Free fall, resting height, finite values and the rotation-aware floor check passed. The failed outcome remains saved; the cause of drift has not been established.`));
  const height=fixture.samples.map((s)=>[s.t_s,s.position_m[2]]),drift=fixture.samples.map((s)=>[s.t_s,Math.hypot(s.position_m[0],s.position_m[1])*1000]);
  const reference=fixture.samples.filter((s)=>s.t_s<=.6).map((s)=>[s.t_s,2-.5*9.81*s.t_s**2]);
  chart($('height-chart'),height,{maxY:2,unit:'height · m',reference}); chart($('drift-chart'),drift,{maxY:30,unit:'distance from initial XY · mm',limit:5});
  const titles={freefall_error_below_2cm:'Free-fall error < 20 mm',final_rest_height_within_5mm:'Final height within 5 mm',floor_penetration_below_5mm:'Rotated-corner penetration < 5 mm',horizontal_drift_below_5mm:'Horizontal drift < 5 mm',all_finite:'All positions finite'};
  for(const [key,passed] of Object.entries(report.backend.fixture_checks)){const item=node('div',undefined,`check${passed?'':' fail'}`);item.append(node('span',passed?'PASS':'FAIL'),node('div',titles[key]||key));$('fixture-checks').append(item);}
  const recheck=report.backend.rotation_aware_box_floor_recheck;
  const notes=node('div',undefined,'fixture-notes');
  const first=node('p'); first.append(node('strong','Fixed synthetic trial'),node('span',`2 seconds · 240 fps · 10 substeps/frame · 20 solver iterations. Gravity −9.81 m/s². Free-fall errors: ${fixture.freefall_checks.map((row)=>`${(row.absolute_error_m*1000).toFixed(2)} mm at ${row.t_s.toFixed(2)} s`).join('; ')}. No robot, renderer or real component dynamics ran.`));
  const second=node('p');second.append(node('strong','Preserved correction'),node('span',`All ${recheck.samples_checked} saved quaternions were used to recheck actual lowest box corners. Maximum penetration: ${(recheck.maximum_penetration_m*1000).toFixed(3)} mm. This supersedes the original center-only floor diagnostic without another solver run or altered tolerances.`));notes.append(first,second);$('fixture-detail').append(notes);
}
function renderNext(report) {
  $('next-title').textContent='Improve reconstruction from the video you already have';
  $('next-benefit').textContent='Ordinary recorded video is the starting point. Inspect the existing part tracking and reconstruction now; next, compare automatic camera estimates against source pixels. That camera comparison is planned work. No checkerboard or special capture is required to use the app.';
  const priorities=[
    ['Compare automatic camera hypotheses','Test saved learned camera estimates and mathematical focal search before choosing a camera. Keep ambiguous estimates visible instead of treating them as measured calibration.','Planned improvement · existing footage and saved estimates first'],
    ['Check each moving part separately','Use the existing component masks, tracking and reconstruction checks. Inspect missing views and source alignment without bridging gaps or placing unsupported parts together.','Existing pipeline · results remain conditional'],
    ['Keep physical unknowns explicit','Better video reconstruction can improve the visual asset. Metric scale, common placement, collision shape, mass, joints and contact still need evidence before a real-object physics trial.','Physical trial remains blocked · no invented simulator inputs']
  ];
  for(const [title,benefit,scope]of priorities){const item=node('div',undefined,'priority');item.append(node('strong',title),node('p',benefit),node('span',scope));$('priorities').append(item);}
  const history=node('details',undefined,'evidence-detail');
  history.append(node('summary','Earlier report recommendations · historical, optional'));
  history.append(node('p','The unchanged downloadable report predates the ordinary-video product focus. Its capture recommendations remain recorded below as optional diagnostic experiments, not upload requirements.'));
  history.append(node('p',`${report.recommended_next_action.title} — ${report.recommended_next_action.benefit}`));
  for(const action of report.priorities)history.append(node('p',`${action.title} — ${action.benefit}`));
  $('priorities').parentElement.append(history);
}
function renderIntegrity(report,saved,environment) {
  const out=$('integrity'); out.append(node('p',`Pinned index and report hashes matched. The source inventory verified ${report.source_case.preflight.declared_artifact_hashes_verified} declared artifact identities. Nested model-output semantics were not recursively certified. Measurements and physical truth remain unverified.`));
  dl(out,[['Report SHA-256',saved.hash],['Program SHA-256',report.source_case.contract.interaction.sha256],['Video SHA-256',report.source_case.contract.source_video.sha256],['Code SHA-256',report.code.physics_validation_sha256],['Availability scope',environment.scope],['Physical execution','No reconstructed-object dynamics, assembly/contact validation, robot control or training ran.']]);
  for(const limitation of report.limitations)out.append(node('p',limitation));
}
async function start() {
  const indexSaved=await verifiedJson(INDEX_PATH,INDEX_SHA256),index=indexSaved.document;
  if(index.format!=='interaction-twin-physics-validation-index'||index.version!==1||!Array.isArray(index.reports)||index.reports.length!==1||index.simulation_ready!==false)throw new Error('Unsupported physics report index');
  const record=index.reports[0],saved=await verifiedJson(record.path,record.sha256),report=saved.document;
  const [fixtureSaved,environmentSaved]=await Promise.all([verifiedJson(report.backend.software_fixture.path,report.backend.software_fixture.sha256),verifiedJson(report.backend.environment.path,report.backend.environment.sha256)]);
  const fixture=fixtureSaved.document;validateReport(report,fixture);
  if(environmentSaved.document.blender_cpu_execution.receipt_sha256!==fixtureSaved.hash)throw new Error('Engine availability and original fixture identities differ');
  const inventory=[[report.source_case.preflight.declared_artifact_hashes_verified,'verified artifact identities'],[report.source_case.native_mesh_diagnostics.length,'native visual assets'],[report.source_case.contract.components.length,'source component tracklets'],[0,'real-object physics runs']];
  for(const [value,label]of inventory){const stat=node('div',undefined,'stat');stat.append(node('strong',number(value)),node('span',label));$('stats').append(stat);}
  renderAssets(report);renderRequirements(report);renderFixture(report,fixture);renderNext(report);renderIntegrity(report,saved,environmentSaved.document);
  await mountAssemblyEvidence(report.source_case.contract);
  $('download-report').disabled=false;$('download-report').addEventListener('click',()=>download(saved,'physics-validation-report-v2.json'));$('download-fixture').addEventListener('click',()=>download(fixtureSaved,'original-blender-synthetic-fixture.json'));
  $('load-status').textContent='Verified saved report · physical object trial blocked';$('report').hidden=false;
  // A direct hash can target a section that was hidden during initial loading.
  // Honor that navigation only after its verified report has been displayed.
  if(['#parts','#missing','#assembly','#solver'].includes(location.hash)) document.querySelector(location.hash).scrollIntoView({block:'start'});
}
start().catch((error)=>{$('load-status').textContent='Saved report unavailable or identity check failed';$('load-status').classList.add('failed');const detail=node('p',error.message,'error-detail');$('load-status').after(detail);console.error('Physics Lab:',error.message);});
