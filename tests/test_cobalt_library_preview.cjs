const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const ROOT=path.resolve(__dirname,'..'),source=fs.readFileSync(path.join(ROOT,'showcase/library.js'),'utf8');
const catalog=JSON.parse(fs.readFileSync(path.join(ROOT,'showcase/public/library-catalog.json'))),index=JSON.parse(fs.readFileSync(path.join(ROOT,'showcase/public/aligned-scenes.json')));
function fixture(){
 class Node{constructor(){this.children=[];this.attributes={};this.dataset={};this.hidden=false;this.paused=true;this.currentTime=0;this.readyState=0;this.listeners={};this.messages=[];this.contentWindow={postMessage:(data,origin)=>this.messages.push({data,origin})};}append(...nodes){this.children.push(...nodes);}replaceChildren(...nodes){this.children=[...nodes];}removeAttribute(key){delete this[key];delete this.attributes[key];}setAttribute(k,v){this.attributes[k]=String(v);}addEventListener(name,fn){(this.listeners[name]||=[]).push(fn);}fire(name,event={}){for(const fn of this.listeners[name]||[])fn({target:this,...event});}pause(){this.paused=true;}load(){}}
 const nodes=new Map(),pending=new Map(),listeners={};
 const document={currentScript:{src:'http://127.0.0.1:8766/showcase/library.js'},getElementById(id){if(['library-preview','library-cards'].includes(id))return null;if(!nodes.has(id))nodes.set(id,new Node());return nodes.get(id);},createElement(){return new Node();},querySelectorAll(){return[];}};
 const context={document,URL,URLSearchParams,TextDecoder,AbortController,Blob,location:{origin:'http://127.0.0.1:8766'},addEventListener:(name,fn)=>listeners[name]=fn,AlignedScenes:{find:source=>new Promise((resolve,reject)=>pending.set(source,{resolve,reject})),checked:async ref=>JSON.parse(fs.readFileSync(path.join(ROOT,ref.path)))},AlignedSceneContract:require('../showcase/aligned-scene-contract.js'),Sam3DModelReviews:require('../showcase/sam3d-models.js')};
 vm.runInNewContext(source.replace('\nstart();\n','\nroot.__preview={startPreview,postPreview,connectPreview,clearPreview,scrollToInspector,setCurrentForTest:record=>{current=record;}};\nstart();\n'),context);
 return{nodes,pending,listeners,context,preview:context.__preview,node(id){return document.getElementById(id);}};
}
function program(record){const b=fs.readFileSync(path.join(ROOT,record.interaction.path));return b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength);}
const jabra=catalog.records.find(r=>r.source.sha256===index.scenes[0].source_sha256),other=catalog.records.find(r=>r.id!==jabra.id);
const saved=index.scenes[0],aligned={reference:saved.manifest,manifest:JSON.parse(fs.readFileSync(path.join(ROOT,saved.manifest.path))),object_program:saved.object_program};
test('declared derivative replay labels its descriptive file and preserves original native downloads',async()=>{
 const relative='data/assembly-refinements/jabra-source-refit-20261103-v2/derived-assembly.json',bytes=fs.readFileSync(path.join(ROOT,relative)),derived=JSON.parse(bytes),binding={path:relative,sha256:require('node:crypto').createHash('sha256').update(bytes).digest('hex'),bytes:bytes.length},record=catalog.records.find(r=>r.source.sha256===derived.source.sha256),selected={reference:derived.scene_ref,manifest:JSON.parse(fs.readFileSync(path.join(ROOT,derived.scene_ref.path))),object_program:binding,base_object_program:derived.base_object_program_ref,derived_assembly:{artifact:binding,document:derived}};
 const ui=fixture(),native=JSON.parse(fs.readFileSync(path.join(ROOT,derived.base_object_program_ref.path)));ui.context.AlignedSceneContract.validateDerivedAssembly(derived,selected.manifest,{sceneReference:derived.scene_ref,baseProgramReference:derived.base_object_program_ref,baseProgram:native,baseSceneReference:native.constrained_scene,baseScene:JSON.parse(fs.readFileSync(path.join(ROOT,native.constrained_scene.path))),refinementReport:JSON.parse(fs.readFileSync(path.join(ROOT,derived.refinement_report_ref.path))),sourceAudit:JSON.parse(fs.readFileSync(path.join(ROOT,derived.source_alignment_audit_ref.path)))});const run=ui.preview.startPreview(record,program(record));ui.pending.get(record.source.sha256).resolve(selected);await run;
 assert.equal(ui.node('program-format').textContent,'Refined twin program v1');assert.equal(ui.node('description-download').href,'/files/'+relative);assert.match(ui.node('preview-scope').textContent,/Reviewed geometry refinements/);
 const links=ui.node('twin-downloads').children.filter(n=>n.href);assert.ok(links.some(n=>n.href==='/files/'+derived.base_object_program_ref.path));assert.ok(links.some(n=>n.href==='/files/'+relative));
 const meshes=links.filter(n=>n.href.endsWith('.glb')),nativeReviews=ui.context.Sam3DModelReviews.forRecording(record),refined=derived.component_assets.filter(c=>c.native_glb.sha256!==c.original_native_glb.sha256),expected=[...nativeReviews.map(r=>r.native_glb.path),...refined.map(c=>c.native_glb.path)].map(p=>'/files/'+p);
 assert.deepEqual(meshes.map(n=>n.href).sort(),expected.sort(),'Every unchanged SAM model and separately declared derivative has its exact own download');
 for(const c of refined){const row=meshes.find(n=>n.href==='/files/'+c.native_glb.path);assert.match(row.children[0].children[0].textContent,/· refined$/);assert.match(row.children[0].children[1].textContent,/original available below/);}
 const labels=links.map(n=>n.children[0].children[0].textContent);assert.ok(labels.includes('Refined twin program v1'));assert.ok(labels.includes('Original native object program'));
});
test('library preview ties its iframe, clock messages and descriptive download to the exact curated source',async()=>{
 const ui=fixture(),run=ui.preview.startPreview(jabra,program(jabra));ui.pending.get(jabra.source.sha256).resolve(aligned);await run;
 const frame=ui.node('twin-alignment-frame'),url=new URL(frame.src,'http://127.0.0.1:8766/showcase/');assert.equal(url.pathname,'/showcase/aligned-scene.html');assert.equal(url.searchParams.get('source'),jabra.source.sha256);assert.equal(url.searchParams.get('sha256'),saved.manifest.sha256);
 assert.equal(ui.node('description-download').href,'/files/'+saved.object_program.path);assert.match(ui.node('program-document').textContent,/interaction-twin-object-program/);
 ui.node('twin-source-video').currentTime=16.7;ui.preview.postPreview();const message=frame.messages.at(-1);assert.equal(message.origin,'http://127.0.0.1:8766');assert.equal(message.data.manifest_sha256,saved.manifest.sha256);assert.equal(message.data.source_sha256,jabra.source.sha256);assert.equal(message.data.time_s,16.7);
 const links=ui.node('twin-downloads').children.filter(n=>n.href).map(n=>n.href);assert.ok(links.includes('/files/'+saved.object_program.path));assert.ok(links.includes('/files/'+saved.manifest.path));assert.equal(links.filter(p=>p.endsWith('.glb')).length,jabra.sam3d_reviews.length);
});
test('a late connected-scene lookup cannot replace a newly selected recording',async()=>{
 const ui=fixture(),first=ui.preview.startPreview(jabra,program(jabra)),second=ui.preview.startPreview(other,program(other));
 ui.pending.get(other.source.sha256).resolve(null);await second;ui.pending.get(jabra.source.sha256).resolve(aligned);await first;
 assert.equal(ui.node('twin-source-video').src,'/files/'+other.source.path);assert.equal(ui.node('twin-alignment-frame').src,undefined);assert.equal(ui.node('twin-alignment-frame').hidden,true);assert.equal(ui.node('description-download').href,'/files/'+other.interaction.path);assert.match(ui.node('twin-alignment-status').textContent,/not available/);
 assert.ok(ui.node('twin-downloads').children.every(n=>!n.href||n.href!=='/files/'+saved.manifest.path));
});
test('only the active source, manifest and exact iframe may send the ready handshake',async()=>{
 const ui=fixture();ui.preview.connectPreview();const run=ui.preview.startPreview(jabra,program(jabra));ui.pending.get(jabra.source.sha256).resolve(aligned);await run;
 const frame=ui.node('twin-alignment-frame'),valid={origin:'http://127.0.0.1:8766',source:frame.contentWindow,data:{type:'interaction-twin:aligned-ready',source_sha256:jabra.source.sha256,manifest_sha256:saved.manifest.sha256}};
 for(const invalid of [{...valid,origin:'https://example.com'},{...valid,source:{}},{...valid,data:{...valid.data,source_sha256:'0'.repeat(64)}},{...valid,data:{...valid.data,manifest_sha256:'0'.repeat(64)}}])ui.listeners.message(invalid);
 assert.equal(frame.messages.length,0);ui.listeners.message(valid);assert.equal(frame.messages.length,1);ui.preview.clearPreview();ui.listeners.message(valid);assert.equal(frame.messages.length,1);
});

test('invalid or unsupported video times are not sent as valid source replay positions',async()=>{
 const ui=fixture(),run=ui.preview.startPreview(jabra,program(jabra));ui.pending.get(jabra.source.sha256).resolve(aligned);await run;
 const video=ui.node('twin-source-video'),frame=ui.node('twin-alignment-frame');for(const t of [NaN,Infinity,-1,jabra.duration_s+1]){video.currentTime=t;ui.preview.postPreview();}assert.equal(frame.messages.length,0);video.currentTime=0;ui.preview.postPreview();assert.equal(frame.messages.length,1);
});

test('a failed descriptive-program byte check withdraws scene downloads and restores the verified procedure',async()=>{
 const ui=fixture();ui.context.AlignedScenes.checked=async()=>{throw Error('Saved bytes changed.');};const run=ui.preview.startPreview(jabra,program(jabra));ui.pending.get(jabra.source.sha256).resolve(aligned);await run;
 assert.equal(ui.node('twin-alignment-frame').hidden,true);assert.equal(ui.node('twin-alignment-frame').src,undefined);assert.match(ui.node('twin-alignment-status').textContent,/Saved bytes changed/);assert.equal(ui.node('description-download').href,'/files/'+jabra.interaction.path);
 assert.ok(ui.node('twin-downloads').children.every(n=>n.href!=='/files/'+saved.manifest.path&&n.href!=='/files/'+saved.object_program.path));assert.equal(ui.node('preview-camera').disabled,true);assert.equal(ui.node('preview-orbit').disabled,true);
});

test('a late descriptive-program fetch cannot overwrite a newly selected recording',async()=>{
 const ui=fixture();let release;ui.context.AlignedScenes.checked=()=>new Promise(resolve=>release=resolve);const first=ui.preview.startPreview(jabra,program(jabra));ui.pending.get(jabra.source.sha256).resolve(aligned);await new Promise(resolve=>setImmediate(resolve));assert.ok(release);
 const second=ui.preview.startPreview(other,program(other));ui.pending.get(other.source.sha256).resolve(null);await second;release(JSON.parse(fs.readFileSync(path.join(ROOT,saved.object_program.path))));await first;
 assert.equal(ui.node('description-download').href,'/files/'+other.interaction.path);assert.equal(JSON.parse(ui.node('program-document').textContent).source.sha256,other.source.sha256);assert.equal(ui.node('twin-alignment-frame').src,undefined);
});

test('first source metadata seeks to the validated supported preview time exactly once',async()=>{
 const ui=fixture();ui.preview.connectPreview();const run=ui.preview.startPreview(jabra,program(jabra));ui.pending.get(jabra.source.sha256).resolve(aligned);await run;
 const video=ui.node('twin-source-video'),expected=ui.context.AlignedSceneContract.previewTime(aligned.manifest).time_s;assert.ok(expected>0);assert.equal(video.currentTime,0);video.readyState=1;video.fire('loadedmetadata');assert.equal(video.currentTime,expected);assert.equal(ui.node('preview-timeline').disabled,false);
 const range=ui.node('preview-timeline');range.value='4.25';range.fire('input');assert.equal(video.currentTime,4.25);video.fire('loadedmetadata');assert.equal(video.currentTime,4.25);assert.equal(ui.node('preview-time').textContent,'4.25 s');
});

test('source interaction before scene discovery prevents a delayed automatic reset',async()=>{
 const ui=fixture();ui.preview.connectPreview();const run=ui.preview.startPreview(jabra,program(jabra));const video=ui.node('twin-source-video');video.readyState=1;video.currentTime=12.5;video.fire('pointerdown');ui.pending.get(jabra.source.sha256).resolve(aligned);await run;video.fire('loadedmetadata');assert.equal(video.currentTime,12.5);
});

test('initial seek belongs only to the current recording and does not run on late metadata',async()=>{
 const ui=fixture();ui.preview.connectPreview();const first=ui.preview.startPreview(jabra,program(jabra));ui.pending.get(jabra.source.sha256).resolve(aligned);await first;const second=ui.preview.startPreview(other,program(other));ui.pending.get(other.source.sha256).resolve(null);await second;const video=ui.node('twin-source-video');video.currentTime=0;video.readyState=1;video.fire('loadedmetadata');assert.equal(video.currentTime,0);assert.equal(video.src,'/files/'+other.source.path);
});

test('card-triggered scroll waits for layout, honors reduced motion and ignores replaced selections',()=>{
 const ui=fixture(),queued=[];ui.context.requestAnimationFrame=fn=>queued.push(fn);ui.context.matchMedia=()=>({matches:true});const panel=ui.node('library-detail'),calls=[];panel.scrollIntoView=options=>calls.push(options);panel.hidden=false;ui.preview.setCurrentForTest(jabra);
 ui.preview.scrollToInspector(jabra,false);assert.equal(queued.length,0);ui.preview.scrollToInspector(jabra,true);assert.equal(queued.length,1);queued.shift()();assert.equal(calls.length,1);assert.equal(calls[0].behavior,'auto');
 ui.preview.scrollToInspector(jabra,true);ui.preview.setCurrentForTest(other);queued.shift()();assert.equal(calls.length,1);
});
