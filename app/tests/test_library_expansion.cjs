/* Exercise the actual public controller, its fetch epochs and one shared inspector. */
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),{webcrypto,createHash}=require('node:crypto');
const ROOT=path.resolve(__dirname,'..'),script=fs.readFileSync(path.join(ROOT,'showcase/library.js'),'utf8'),html=fs.readFileSync(path.join(ROOT,'showcase/library.html'),'utf8');
const saved=JSON.parse(fs.readFileSync(path.join(ROOT,'showcase/public/library-catalog.json'))),alignedIndex=JSON.parse(fs.readFileSync(path.join(ROOT,'showcase/public/aligned-scenes.json')));
const sha=raw=>createHash('sha256').update(raw).digest('hex'),clone=value=>JSON.parse(JSON.stringify(value)),deferred=()=>{let resolve;const promise=new Promise(r=>resolve=r);return{promise,resolve};};
const until=async(check,message)=>{for(let i=0;i<180;i++){if(check())return;await new Promise(resolve=>setImmediate(resolve));}throw Error(message);};

function savedAligned(source){
 const row=alignedIndex.scenes.find(r=>r.source_sha256===source);if(!row)return null;
 const read=ref=>JSON.parse(fs.readFileSync(path.join(ROOT,ref.path)));
 return require('./derived_scene_fixture.cjs')({reference:row.manifest,manifest:read(row.manifest),object_program:row.derived_assembly||row.object_program,...(row.derived_assembly?{base_object_program:row.object_program,derived_assembly:{artifact:row.derived_assembly,document:read(row.derived_assembly)}}:{}),...(row.observed_surfaces?{observed_surfaces:{artifact:row.observed_surfaces,document:read(row.observed_surfaces)}}:{})});
}
function fixture({query='',catalog=clone(saved),find}={}){
 const removals=[],requests=[],blocked=new Map(),windowListeners={},frames=new Map(),blobURLs=new Set();let serial=0;
 const location={url:new URL('http://127.0.0.1:8766/showcase/library.html'+query),get href(){return this.url.href;},get origin(){return this.url.origin;},get hash(){return this.url.hash;}};
 let document;
 function simple(node,selector){
  if(node.tagName==='#document')return false;
  const tag=selector.match(/^[a-zA-Z][\w-]*/)?.[0];if(tag&&node.tagName!==tag.toLowerCase())return false;
  const id=selector.match(/#([\w-]+)/)?.[1];if(id&&node.id!==id)return false;
  if([...selector.matchAll(/\.([\w-]+)/g)].some(m=>!node.classList.contains(m[1])))return false;
  for(const m of selector.matchAll(/\[([\w-]+)(?:=["']?([^\]"']+)["']?)?\]/g)){const value=node.getAttribute(m[1]);if(value===null||(m[2]!==undefined&&value!==m[2]))return false;}
  return true;
 }
 function matches(node,selector){
  return selector.split(',').some(part=>{const pieces=part.trim().split(/\s+/);if(!simple(node,pieces.pop()))return false;let parent=node.parentElement;while(pieces.length){const wanted=pieces.pop();while(parent&&!simple(parent,wanted))parent=parent.parentElement;if(!parent)return false;parent=parent.parentElement;}return true;});
 }
 class Node{
  constructor(tag){this.tagName=tag.toLowerCase();this.parentElement=null;this.children=[];this.attributes={};this.dataset={};this.style={};this.listeners={};this.hidden=false;this.paused=true;this.currentTime=0;this.readyState=0;this.messages=[];this.scrolls=[];this.contentWindow={postMessage:(data,origin)=>this.messages.push({data,origin})};}
  get id(){return this.attributes.id||'';}set id(v){this.attributes.id=String(v);}
  get className(){return this.attributes.class||'';}set className(v){this.attributes.class=String(v);}
  get classList(){const node=this;return{contains:c=>node.className.split(/\s+/).includes(c),add(...cs){node.className=[...new Set([...node.className.split(/\s+/).filter(Boolean),...cs])].join(' ');},remove(...cs){node.className=node.className.split(/\s+/).filter(c=>!cs.includes(c)).join(' ');},toggle(c,on){const present=this.contains(c);on=on===undefined?!present:!!on;on?this.add(c):this.remove(c);return on;}};}
  get isConnected(){let n=this;while(n){if(n===document.root)return true;n=n.parentElement;}return false;}
  get options(){return this.querySelectorAll('option');}get value(){return this._value===undefined?(this.tagName==='select'?this.options[0]?.value||'':''):this._value;}set value(v){this._value=String(v);}
  get src(){return this.attributes.src;}set src(v){this.attributes.src=String(v);if(this.tagName==='img')queueMicrotask(()=>this.onload?.());}
  get textContent(){return this._text||this.children.map(n=>n.textContent).join('');}set textContent(v){this._text=String(v);for(const child of [...this.children])child.remove();}
  set innerHTML(v){this.textContent=v;}
  getAttribute(k){if(k==='hidden')return this.hidden?'':null;if(k.startsWith('data-'))return this.dataset[k.slice(5).replace(/-([a-z])/g,(_,c)=>c.toUpperCase())]??null;return this.attributes[k]??null;}
  setAttribute(k,v){if(k==='hidden')this.hidden=true;else if(k.startsWith('data-'))this.dataset[k.slice(5).replace(/-([a-z])/g,(_,c)=>c.toUpperCase())]=String(v);else this.attributes[k]=String(v);if(k==='value')this.value=v;}
  removeAttribute(k){delete this.attributes[k];if(k==='hidden')this.hidden=false;}
  contains(node){return node===this||this.children.some(c=>c.contains(node));}
  _detach(){if(!this.parentElement)return;if(this.isConnected)for(const frame of [this,...this.querySelectorAll('iframe')])if(frame.tagName==='iframe'&&frame.src)removals.push(frame);const siblings=this.parentElement.children;siblings.splice(siblings.indexOf(this),1);this.parentElement=null;}
  append(...nodes){for(const node of nodes){if(typeof node==='string')continue;node._detach();node.parentElement=this;this.children.push(node);}}
  insertBefore(node,before){if(node===before)return;if(!before){this.append(node);return;}assert.equal(before.parentElement,this);node._detach();node.parentElement=this;this.children.splice(this.children.indexOf(before),0,node);}
  replaceChildren(...nodes){for(const child of [...this.children])child._detach();this._text='';this.append(...nodes);}
  remove(){this._detach();}replaceWith(node){const parent=this.parentElement;parent.insertBefore(node,this);this._detach();}
  cloneNode(){const node=new Node(this.tagName);node.attributes={...this.attributes};node.dataset={...this.dataset};node.hidden=this.hidden;return node;}
  querySelectorAll(selector){const result=[];for(const child of this.children){if(matches(child,selector))result.push(child);result.push(...child.querySelectorAll(selector));}return result;}
  querySelector(selector){return this.querySelectorAll(selector)[0]||null;}
  closest(selector){for(let node=this;node;node=node.parentElement)if(matches(node,selector))return node;return null;}
  addEventListener(name,fn){(this.listeners[name]||=[]).push(fn);}async fire(name,extra={}){return Promise.all((this.listeners[name]||[]).map(fn=>fn({target:this,currentTarget:this,preventDefault(){},...extra})));}
  focus(){document.activeElement=this;}scrollIntoView(value){this.scrolls.push(value);}
  pause(){this.paused=true;}load(){this.readyState=0;this.currentTime=0;this.paused=true;}play(){this.paused=false;return this.fire('play');}
 }
 document={root:new Node('#document'),currentScript:{src:location.origin+'/showcase/library.js'},activeElement:null,createElement:tag=>new Node(tag),getElementById(id){return this.root.querySelector('#'+id);},querySelectorAll(selector){return this.root.querySelectorAll(selector);}};
 const stack=[document.root],voidTags=new Set(['meta','link','img','input','br','hr','source','wbr']);
 for(const match of html.matchAll(/<\/?[^>]+>/g)){
  const token=match[0];if(/^<!/.test(token))continue;const name=token.match(/^<\/?([\w-]+)/)?.[1]?.toLowerCase();if(!name)continue;
  if(token.startsWith('</')){for(let i=stack.length-1;i>0;i--)if(stack[i].tagName===name){stack.length=i;break;}continue;}
  const node=new Node(name),attributes=token.slice(token.indexOf(name)+name.length).replace(/\/?\s*>$/,'');
  for(const item of attributes.matchAll(/([^\s=/>]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+)))?/g))node.setAttribute(item[1],item[2]??item[3]??item[4]??'');
  stack.at(-1).append(node);if(!voidTags.has(name)&&!token.endsWith('/>'))stack.push(node);
 }
 document.body=document.root.querySelector('body');
 const catalogRaw=Buffer.from(JSON.stringify(catalog)),catalogName='library-catalog-'+sha(catalogRaw)+'.json';
 const indexRaw=Buffer.from(JSON.stringify({format:'interaction-twin-library-index',version:1,catalog:{path:catalogName,sha256:sha(catalogRaw),bytes:catalogRaw.length}}));
 const response=(raw,wait)=>({ok:true,status:200,text:async()=>raw.toString(),arrayBuffer:async()=>{if(wait)await wait.promise;return raw.buffer.slice(raw.byteOffset,raw.byteOffset+raw.byteLength);}});
 const context={document,location,URL,URLSearchParams,TextDecoder,TextEncoder,Blob,AbortController,crypto:webcrypto,navigator:{clipboard:{writeText:async()=>{}}},history:{replaceState(_a,_b,value){location.url=new URL(value,location.href);}},matchMedia:()=>({matches:true}),requestAnimationFrame:fn=>{frames.set(++serial,fn);return serial;},cancelAnimationFrame:id=>frames.delete(id),addEventListener:(name,fn)=>(windowListeners[name]||=[]).push(fn),Sam3DModelReviews:require('../showcase/sam3d-models.js'),AlignedSceneContract:require('../showcase/aligned-scene-contract.js'),ObjectProgramSummaryV1:require('../showcase/object-program-summary-v1.js'),
  fetch:async(value,options={})=>{const url=new URL(value,location.href);requests.push({path:url.pathname,signal:options.signal});if(url.pathname.endsWith('/library-index.json'))return response(indexRaw);if(url.pathname.endsWith('/'+catalogName))return response(catalogRaw);if(!url.pathname.startsWith('/files/data/'))throw Error('No noncurated resource fixture: '+url.pathname);const relative=url.pathname.slice('/files/'.length),raw=fs.readFileSync(path.join(ROOT,relative));return response(raw,blocked.get(relative));},
  AlignedScenes:{find:find||(async source=>savedAligned(source)),checked:async reference=>JSON.parse(fs.readFileSync(path.join(ROOT,reference.path)))}};
 const originalCreate=URL.createObjectURL;context.URL=class extends URL{static createObjectURL(blob){const value=originalCreate(blob);blobURLs.add(value);return value;}static revokeObjectURL(value){blobURLs.delete(value);URL.revokeObjectURL(value);}};
 const injected=script.replace('\nstart();\n','\nroot.__expansion={start,select,collapse,renderCards};\n');assert.notEqual(injected,script,'Actual controller hook insertion required');vm.runInNewContext(injected,context,{filename:'showcase/library.js'});
 return{context,document,location,requests,blocked,removals,frames,hooks:context.__expansion,records:catalog.records,node:id=>document.getElementById(id),host(id){return document.querySelectorAll('.library-item').find(n=>n.dataset.twin===id);},async start(){await context.__expansion.start();},hold(relative){const value=deferred();blocked.set(relative,value);return value;},flushFrames(){const waiting=[...frames.values()];frames.clear();for(const fn of waiting)fn();},close(){context.__expansion.collapse();for(const fn of windowListeners.pagehide||[])fn();for(const value of blobURLs)URL.revokeObjectURL(value);}};
}
const jabra=saved.records.find(r=>r.sam3d_reviews?.length===4),other=saved.records.find(r=>r.id!==jabra.id);

function correctedLaptop(){const catalog=clone(saved),record=catalog.records.find(r=>r.id==='flow-capture-laptop-20261103-v1--flow-procedure-1'),relation=record.relationships.find(r=>r.id==='r_cable_to_laptop'),artifact={path:'data/semantic-relation-corrections/fixture-v1/descriptor.json',sha256:'a'.repeat(64),bytes:1200};record.semantic_correction=artifact;relation.object_track_id='laptop_base';relation.semantic_correction={artifact:{...artifact},original_object_track_id:'laptop_lid',inference:'source_reviewed_candidate_not_verified_attachment',human_reviewed:false,physical_accuracy_verified:false};return{catalog,record,relation};}
test('a source-reviewed relation is visibly declared and downloadable while the original program stays available',async t=>{const{catalog,record}=correctedLaptop(),ui=fixture({catalog});t.after(()=>ui.close());await ui.start();await ui.hooks.select(record);assert.match(ui.node('relation-list').textContent,/Source-reviewed inference/);assert.match(ui.node('object-program-summary').textContent,/Original generated files remain unchanged/);const link=ui.node('object-program-summary').querySelector('a');assert.equal(ui.node('object-program-summary').querySelectorAll('a').at(-1).href,'/files/'+record.semantic_correction.path);assert.ok(ui.node('twin-downloads').querySelectorAll('a').some(a=>a.href==='/files/'+record.semantic_correction.path));assert.equal(ui.node('download-program').href,'/files/'+record.interaction.path);});
test('correction annotations cannot invent approval, physical validation, a foreign artifact or an unbound component',()=>{const L=require('../showcase/library.js');for(const change of[({record})=>{record.semantic_correction.path='data/../private.json';},({relation})=>{relation.semantic_correction.human_reviewed=true;},({relation})=>{relation.semantic_correction.physical_accuracy_verified=true;},({relation})=>{relation.semantic_correction.artifact.sha256='b'.repeat(64);},({relation})=>{relation.semantic_correction.original_object_track_id='foreign_part';},({record})=>{delete record.semantic_correction;},({relation})=>{delete relation.semantic_correction;}]){const input=correctedLaptop();change(input);assert.throws(()=>L.validateCatalog(input.catalog),/correction|binding/i);}});

test('a curated quality note is visible on the compact card and in its expanded inspector',async t=>{
 const catalog=clone(saved),record=catalog.records.find(r=>r.id===jabra.id);
 record.public_title='Laptop + charging cable';
 record.public_note='Estimated hinge: 27 supported samples, 39 unknown. Screen texture and physical fit remain unverified.';
 const ui=fixture({catalog});t.after(()=>ui.close());await ui.start();
 assert.equal(ui.host(record.id).querySelector('h3').textContent,record.public_title);
 assert.equal(ui.host(record.id).querySelector('.library-note').textContent,record.public_note);
 await ui.hooks.select(record);
 assert.equal(ui.node('detail-quality-note').textContent,record.public_note);
 assert.equal(ui.node('detail-quality-note').hidden,false);
 assert.equal(ui.node('detail-summary').textContent,record.summary);
 assert.equal(ui.node('library-detail').parentElement,ui.host(record.id));
});

test('default browsing renders compact disclosures and does not load a program or auto-open a twin',async t=>{
 const ui=fixture();t.after(()=>ui.close());await ui.start();
 assert.equal(ui.document.querySelectorAll('.library-item').length,saved.records.length);assert.equal(ui.document.querySelectorAll('.is-expanded').length,0);assert.equal(ui.node('library-detail').parentElement,ui.node('library-detail-parking'));assert.equal(ui.node('library-detail').hidden,true);
 for(const host of ui.document.querySelectorAll('.library-item')){const button=host.querySelector('.library-card');assert.equal(button.tagName,'button');assert.equal(button.getAttribute('aria-expanded'),'false');assert.equal(button.getAttribute('aria-controls'),'library-detail');}
 assert.ok(ui.requests.every(r=>!saved.records.some(record=>r.path==='/files/'+record.interaction.path)));assert.equal(ui.location.url.searchParams.has('case'),false);
});

test('selecting a card expands that wrapper with one sibling inspector and minimize restores its trigger and filters',async t=>{
 const ui=fixture({query:'?sort=duration&direction=desc'});t.after(()=>ui.close());await ui.start();await ui.hooks.select(jabra,true);const host=ui.host(jabra.id),button=host.querySelector('.library-card'),detail=ui.node('library-detail');
 assert.equal(detail.parentElement,host);assert.equal(button.hidden,true);assert.equal(button.getAttribute('aria-expanded'),'true');assert.equal(button.contains(detail),false);assert.equal(ui.document.querySelectorAll('#library-detail').length,1);assert.equal(ui.document.querySelectorAll('.is-expanded').length,1);
 await ui.node('minimize-twin').fire('click');assert.equal(detail.parentElement,ui.node('library-detail-parking'));assert.equal(detail.hidden,true);assert.equal(button.hidden,false);assert.equal(button.getAttribute('aria-expanded'),'false');assert.equal(ui.document.activeElement,button);assert.equal(ui.location.url.searchParams.get('sort'),'duration');assert.equal(ui.location.url.searchParams.get('direction'),'desc');assert.equal(ui.location.url.searchParams.has('case'),false);ui.flushFrames();assert.equal(detail.scrolls.length,0,'A queued scroll cannot jump after minimizing');
});

test('minimizing during uncancellable program verification prevents a late response from reopening or starting media',async t=>{
 const ui=fixture();t.after(()=>ui.close());await ui.start();const held=ui.hold(jabra.interaction.path),pending=ui.hooks.select(jabra,true);await until(()=>ui.requests.some(r=>r.path==='/files/'+jabra.interaction.path),'Program verification did not begin');
 await ui.node('minimize-twin').fire('click');held.resolve();await pending;ui.flushFrames();
 assert.equal(ui.document.querySelectorAll('.is-expanded').length,0);assert.equal(ui.node('library-detail').hidden,true);assert.equal(ui.node('library-detail').parentElement,ui.node('library-detail-parking'));assert.equal(ui.node('twin-source-video').src,undefined);assert.equal(ui.node('twin-alignment-frame').src,undefined);assert.equal(ui.location.url.searchParams.has('case'),false);assert.equal(ui.requests.find(r=>r.path==='/files/'+jabra.interaction.path).signal.aborted,true);
});

test('rapid A to B selection keeps B open when A finishes verification last',async t=>{
 const ui=fixture();t.after(()=>ui.close());await ui.start();const a=ui.hold(jabra.interaction.path),b=ui.hold(other.interaction.path),first=ui.hooks.select(jabra),second=ui.hooks.select(other);b.resolve();await second;a.resolve();await first;
 assert.equal(ui.node('library-detail').parentElement,ui.host(other.id));assert.equal(ui.host(other.id).querySelector('.library-card').hidden,true);assert.equal(ui.host(jabra.id).querySelector('.library-card').hidden,false);assert.equal(ui.document.querySelectorAll('.is-expanded').length,1);assert.equal(ui.node('detail-title').textContent,other.public_title||other.title);assert.equal(ui.node('twin-source-video').src,'/files/'+other.source.path);assert.equal(ui.location.url.searchParams.get('case'),other.id);
});

test('sorting and filtering a still-visible twin retain its live expanded host, iframe and source clock',async t=>{
 const ui=fixture();t.after(()=>ui.close());await ui.start();await ui.hooks.select(jabra);await until(()=>ui.node('twin-alignment-frame').src,'Connected scene did not load');const host=ui.host(jabra.id),detail=ui.node('library-detail'),frame=ui.node('twin-alignment-frame'),video=ui.node('twin-source-video'),src=frame.src;video.currentTime=16.7;const removed=ui.removals.length;
 ui.hooks.renderCards([...ui.records].reverse());ui.hooks.renderCards([jabra]);
 assert.equal(ui.host(jabra.id),host);assert.equal(detail.parentElement,host);assert.equal(frame.isConnected,true);assert.equal(frame.src,src);assert.equal(video.currentTime,16.7);assert.equal(ui.removals.length,removed,'Reconciliation must not unmount an active iframe ancestor');assert.equal(ui.document.querySelectorAll('.is-expanded').length,1);
});

test('filtering the selected twin out closes it and cancels a pending scene discovery',async t=>{
 const lookup=deferred(),ui=fixture({find:()=>lookup.promise});t.after(()=>ui.close());await ui.start();await ui.hooks.select(jabra);ui.node('library-search').value='no-such-twin-in-curated-library';await ui.node('library-search').fire('input');lookup.resolve(null);await new Promise(r=>setImmediate(r));
 assert.equal(ui.document.querySelectorAll('.library-item').length,0);assert.equal(ui.node('library-detail').parentElement,ui.node('library-detail-parking'));assert.equal(ui.node('library-detail').hidden,true);assert.equal(ui.node('twin-source-video').paused,true);assert.equal(ui.node('twin-source-video').src,undefined);assert.equal(ui.node('twin-alignment-frame').src,undefined);assert.equal(ui.location.url.searchParams.has('case'),false);assert.equal(ui.location.url.searchParams.get('query'),'no-such-twin-in-curated-library');
});

test('explicit case aliases and part hashes open only their curated twin and collapse removes the selection hash',async t=>{
 const catalog=clone(saved),record=catalog.records.find(r=>r.id===jabra.id),alias='curated-test-historical-alias';record.other_saved_versions.push({id:alias,interaction:record.interaction});catalog.summary.eligible_saved_versions++;
 const ui=fixture({query:'?case='+alias+'#sam3d-models',catalog});t.after(()=>ui.close());await ui.start();assert.equal(ui.node('library-detail').parentElement,ui.host(jabra.id));assert.equal(ui.node('view-parts').getAttribute('aria-selected'),'true');assert.equal(ui.location.url.searchParams.get('case'),jabra.id);
 await ui.node('minimize-twin').fire('click');assert.equal(ui.location.hash,'');assert.equal(ui.location.url.searchParams.has('case'),false);assert.equal(ui.node('sam3d-native-frame').src,undefined);assert.equal(ui.node('twin-alignment-frame').src,undefined);
});

test('default collection chips omit empty classes while a requested empty class remains visible and resettable',async t=>{
 const L=require('../showcase/library.js'),empty=L.TASK_CLASSES.find(c=>!saved.records.some(r=>r.task_classes.includes(c.id)));assert(empty,'Current library has an unavailable interaction class');
 const ui=fixture();t.after(()=>ui.close());await ui.start();const shown=ui.node('library-collections').children;assert.equal(shown.length,L.TASK_CLASSES.filter(c=>saved.records.some(r=>r.task_classes.includes(c.id))).length);assert(shown.every(chip=>!chip.querySelector('.collection-count').classList.contains('empty')));assert(!shown.some(chip=>chip.textContent.includes(empty.label)));
 const selected=fixture({query:'?task_classes='+empty.id});t.after(()=>selected.close());await selected.start();const chip=selected.node('library-collections').children.find(c=>c.textContent.includes(empty.label));assert(chip);assert.equal(chip.getAttribute('aria-pressed'),'true');assert.equal(chip.querySelector('.collection-count').textContent,'0');assert.equal(selected.node('library-empty').hidden,false);await selected.node('collection-all').fire('click');assert.equal(selected.node('library-empty').hidden,true);assert(!selected.node('library-collections').children.some(c=>c.textContent.includes(empty.label)));assert.equal(selected.location.url.searchParams.has('task_classes'),false);
});

test('current Description explains the refined hinge and exact source citations before raw JSON',async t=>{
 const ui=fixture();t.after(()=>ui.close());await ui.start();await ui.hooks.select(jabra);await until(()=>ui.node('object-program-summary').querySelector('.program-summary-scope')?.textContent==='Refined estimated assembly','Refined description never arrived');await ui.node('view-description').fire('click');const box=ui.node('object-program-summary');assert.equal(box.hidden,false);assert.match(box.textContent,/Estimated hinge/);assert.match(box.textContent,/Source configuration range 0–164°/);assert.match(box.textContent,/Physical stops are unverified/);assert.equal(box.querySelectorAll('.program-summary-connections li').length,3);const links=box.querySelectorAll('.program-summary-evidence a');assert.equal(links.length,3);const allowed=jabra.steps.concat(jabra.relationships).flatMap(r=>r.citations);for(const link of links){assert(allowed.some(c=>link.href==='/files/'+c.frame.path&&link.textContent===c.timestamp_s.toFixed(2)+' s'));assert.equal(link.target,'_blank');}assert.equal(box.querySelector('img'),null,'Summary uses metadata and source links without another image dashboard');assert.equal(ui.node('view-panel-description').hidden,false);await box.querySelector('button').fire('click');assert.equal(ui.node('panel-actions').hidden,false);assert.equal(ui.node('panel-relations').hidden,true);assert.equal(ui.node('view-panel-description').hidden,false);assert(ui.node('description-download').href.endsWith('.json'));assert.equal(ui.node('program-document').tagName,'pre');
});

test('late scene discovery for another card cannot replace the current source-visible description',async t=>{
 const phone=saved.records.find(r=>r.components.some(c=>c.id==='phone_blue')),a=deferred(),b=deferred(),lookups=[];
 const ui=fixture({find:source=>{lookups.push(source);return source===jabra.source.sha256?a.promise:b.promise;}});t.after(()=>ui.close());await ui.start();await ui.hooks.select(jabra);await until(()=>lookups.includes(jabra.source.sha256),'First lookup did not start');await ui.hooks.select(phone);await until(()=>lookups.includes(phone.source.sha256),'Phone lookup did not start');b.resolve(savedAligned(phone.source.sha256));await until(()=>ui.node('object-program-summary').querySelector('.program-summary-scope')?.textContent==='Source-visible surface preview','Current phone summary did not render');const currentText=ui.node('object-program-summary').textContent;a.resolve(savedAligned(jabra.source.sha256));await new Promise(r=>setImmediate(r));assert.equal(ui.node('object-program-summary').textContent,currentText);assert.match(currentText,/blue-cased smartphone/);assert.doesNotMatch(currentText,/charging case|0–164°/);assert.equal(ui.node('library-detail').parentElement,ui.host(phone.id));assert.equal(ui.node('twin-source-video').src,'/files/'+phone.source.path);
});

test('summary labels are rendered as text and a foreign aligned context falls back to procedure-only facts',async t=>{
 const catalog=clone(saved),record=catalog.records.find(r=>r.id===jabra.id);record.components[0].label='<img src=x onerror=alert(1)>';
 const ui=fixture({catalog,find:async()=>savedAligned(saved.records.find(r=>r.id!==record.id).source.sha256)});t.after(()=>ui.close());await ui.start();await ui.hooks.select(record);await until(()=>/Saved scene unavailable/.test(ui.node('twin-alignment-status').textContent),'Foreign aligned scene was not rejected');const box=ui.node('object-program-summary');assert.equal(box.querySelector('.program-summary-scope').textContent,'Demonstrated procedure');assert.match(box.textContent,/<img src=x onerror=alert\(1\)>/);assert.equal(box.querySelector('img'),null);assert.doesNotMatch(box.textContent,/0–164°|Refined estimated assembly/);assert.equal(ui.node('twin-alignment-frame').src,undefined);
});
