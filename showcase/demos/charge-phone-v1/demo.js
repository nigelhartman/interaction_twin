/* Saved public metadata only. This viewer cannot invoke models, MCP or robots. */
(() => {
  'use strict';
  const $=id=>document.getElementById(id);
  const node=(tag,text,className)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(className)e.className=className;return e;};
  const link=(title,url)=>{if(typeof url!=='string'||!url.startsWith('/interaction_twin/files/data/')&&!url.startsWith('/interaction_twin/showcase/library.html?case='))throw Error('Unexpected saved source URL');const a=node('a',title);a.href=url;return a;};
  function title(role){return role==='phone'?'Phone + USB-C connector':'USB-C charger + cable';}
  function recording(role){return role==='phone'?'Recording B':'Recording A';}
  const short=s=>s.replace(/_/g,' ');
  Promise.all(['task.json','trace.json','summary.json','muse-review.json'].map(async name=>{const r=await fetch(name,{credentials:'omit',cache:'no-cache'});if(!r.ok)throw Error('Saved example unavailable');return r.json();})).then(([task,trace,summary,review])=>{
    if(task.format!=='interaction-twin-cross-recording-task'||task.sources.length!==2||task.claims.robot_execution_available!==false||trace.transport!=='actual_official_sdk_subprocess_stdio')throw Error('Unexpected saved example contract');
    const sources=new Map(task.sources.map(s=>[s.twin_id,s]));
    for(const s of [...task.sources].reverse()){
      const card=node('article',undefined,'source-card');const sourceInterface=s.interfaces.find(i=>i.id===(s.role==='phone'?'phone_port':'cable_free_end'));const e=sourceInterface.evidence[0];const image=node('img');image.src=e.frame_url;image.alt=s.role==='phone'?'Source frame showing the phone and cable at its edge':'Source frame showing the charger recording’s free cable end';image.loading='lazy';card.append(image);
      const body=node('div',undefined,'source-card-content');body.append(node('span',recording(s.role),'eyebrow'),node('h2',title(s.role)),node('p',s.role==='phone'?'Phone and connector retrieved as separate source-linked components.':'Adapter, cable and socket retrieved from another demonstration.'));
      const links=node('div',undefined,'source-links');links.append(link('Inspect twin →',s.library_url),link(`See evidence · ${e.timestamp_s.toFixed(1)}s ↗`,e.video_url));body.append(links);card.append(body);$('source-recordings').append(card);
    }
    for(const c of task.connection_candidates){const el=node('article',undefined,'connection-item');el.dataset.cross=String(c.cross_recording);el.append(node('span',c.cross_recording?'Across recordings · candidate':'Recorded contact · candidate','candidate-label'),node('h3',c.title),node('p',c.connector_family==='usb_c'?'USB-C · owner reported · fit and power unverified':'European socket context · exact standard unverified'));$('connection-candidates').append(el);}
    for(const step of task.task_draft.steps){const li=node('li');li.append(node('h3',step.title),node('p',short(step.completion_status)));const anchors=node('div',undefined,'step-evidence');for(const s of step.source_steps){const source=sources.get(s.twin_id);if(!source)throw Error('Undeclared task source');const ref=link(`${recording(source.role)} · ${s.start_s.toFixed(1)}–${s.end_s.toFixed(1)}s`,`/interaction_twin/files/${source.source.path}#t=${s.start_s}`);anchors.append(ref);}li.append(anchors);$('task-steps').append(li);}
    for(const requirement of task.missing_requirements){const p=node('p');p.append(node('strong',short(requirement.id)),document.createTextNode(requirement.description));$('missing-requirements').append(p);}
    for(const source of task.sources){const quote=node('blockquote');quote.append(node('p',source.owner_context.statement),node('small',`${recording(source.role)} · ${source.owner_context.attribution} Owner reported; not independently measured.`));$('context-attribution').append(quote);}
    $('muse-review-summary').textContent=review.status==='validated_saved_muse_assessment'?review.result.summary:'The draft is inspectable. Physical execution remains blocked by the requirements below.';
    $('trace-count').textContent=`${summary.mcp_calls} calls · ${summary.verified_native_models} native models · ${summary.verified_evidence_frames} evidence frames`;
    $('trace-description').textContent='A local client searched both twins, retrieved their programs and typed interfaces, requested this task draft, then downloaded and hash-checked every native GLB plus the cited source JPEGs. The original files remain unchanged.';
    for(const c of trace.calls){let text=`${c.sequence}. ${c.method}`;if(c.name)text+=` · ${c.name}`;if(c.arguments&&c.arguments.twin_id)text+=` · ${c.arguments.twin_id}`;if(c.uri)text+=` · ${c.uri}`;if(c.decoded_bytes)text+=` · ${c.decoded_bytes.toLocaleString()} bytes verified`;$('trace-calls').append(node('li',text));}
    $('demo-status').textContent='';
  }).catch(()=>{$('demo-status').textContent='The saved example is unavailable. Return to the library and inspect the source twins.';});
})();
