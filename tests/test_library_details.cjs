const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),{webcrypto}=require('node:crypto');
const ROOT=path.resolve(__dirname,'..'),source=fs.readFileSync(path.join(ROOT,'showcase/library.js'),'utf8');
const catalog=JSON.parse(fs.readFileSync(path.join(ROOT,'showcase/public/library-catalog.json')));

function browserController({selectNative=false,withCards=false,sourceImageBytes=false,corruptSource=false}={}){
 class Node{
  constructor(tag='div'){this.tagName=tag;this.children=[];this.dataset={};this.attributes={};this.textContent='';this.loading='lazy';this.navigations=[];this.isConnected=true;}
  set src(value){this._src=value;this.navigations.push({src:value,loading:this.loading,hidden:this.hidden});}
  get src(){return this._src;}
  removeAttribute(name){delete this.attributes[name];if(name==='src')delete this._src;}
  cloneNode(){return new Node(this.tagName);}
  replaceWith(){}
  append(...nodes){this.children.push(...nodes);}
  replaceChildren(...nodes){this.children=[...nodes];}
  setAttribute(name,value){this.attributes[name]=String(value);}
  addEventListener(){}
 }
 const nodes=new Map(),requests=[];let fetches=0;
 const document={currentScript:{src:'http://127.0.0.1:8766/showcase/library.js'},
  getElementById(id){if(['library-preview','library-cards'].includes(id))return null;if(!nodes.has(id))nodes.set(id,new Node());return nodes.get(id);},
  createElement(tag){return new Node(tag);},querySelectorAll(){return[];}};
 const context={document,URL,URLSearchParams,AbortController,crypto:webcrypto,Blob,location:{origin:'http://127.0.0.1:8766'},fetch:async(route)=>{fetches++;const url=new URL(route,'http://127.0.0.1:8766');requests.push(url.pathname);if(sourceImageBytes&&url.pathname.startsWith('/files/data/')&&url.pathname.endsWith('/source-rgb.png')){const saved=fs.readFileSync(path.join(ROOT,url.pathname.slice('/files/'.length))),raw=Buffer.from(saved);if(corruptSource)raw[raw.length-2]^=1;return{ok:true,arrayBuffer:async()=>raw.buffer.slice(raw.byteOffset,raw.byteOffset+raw.byteLength)};}throw Error('Fixture source bytes intentionally unavailable');}};
 if(selectNative||withCards){const savedReviews=require('../showcase/sam3d-models.js');context.Sam3DModelReviews={forComponent:savedReviews.forComponent,...(withCards?{forRecording:savedReviews.forRecording}:{})};}
 // Test-only access to the real component-selection closure; production code
 // and its epoch/abort behavior remain unchanged.
 const instrumented=source.replace('\nstart();\n','\nroot.__setCurrentForTest=r=>{current=r;};root.__selectComponentForTest=selectComponent;\nstart();\n');
 vm.runInNewContext(instrumented,context,{filename:'showcase/library.js'});
 return{controller:context.InteractionLibrary,nodes,requests,fetches:()=>fetches,
  selectComponent(record,component){context.__setCurrentForTest(record);return context.__selectComponentForTest(component);}};
}

test('actual fresh SAM 3D detail path renders without an aggregate scene or fabricated provenance',()=>{
 const record=catalog.records.find(r=>r.scene_case.scene_ref===null&&r.components.some(c=>c.asset?.backend==='Meta SAM 3D Objects'));
 assert.ok(record,'Current automatic SAM 3D upload is required for this regression');
 const before=JSON.stringify(record),originalTitle=record.title,ui=browserController();
 assert.doesNotThrow(()=>ui.controller.renderDetails(record));
 assert.equal(ui.nodes.get('detail-title').textContent,record.public_title||record.title);
 assert.equal(record.title,originalTitle,'Curated display titles must preserve the original model title');
 assert.equal(ui.nodes.get('component-picker').children.length,record.components.length);
 const rows=ui.nodes.get('provenance-list').children;
 assert.equal(rows.some(r=>r.children[0].textContent==='Saved scene SHA256'),false);
 const geometry=rows.find(r=>r.children[0].textContent==='Geometry inspection');assert.ok(geometry);
 assert.match(geometry.children[1].textContent,/Native SAM parts.*estimated replay/);
 assert.equal(geometry.children[1].href,undefined);
 assert.equal(ui.fetches(),0);assert.equal(JSON.stringify(record),before);
});

test('public details never navigate into an isolated historical aggregate pointer even when every current twin has a connected SAM scene',()=>{
 const record=structuredClone(catalog.records[0]);assert.ok(record.sam3d_reviews.length);
 // Only this inert presentation fixture gains an obsolete pointer. Its actual
 // source/program/native bindings remain exact, and the UI must never fetch it.
 record.scene_case.scene_ref={path:'scenes/'+('0'.repeat(64))+'.json',sha256:'0'.repeat(64),bytes:100};
 const before=JSON.stringify(record);
 const ui=browserController();assert.doesNotThrow(()=>ui.controller.renderDetails(record));
 const row=ui.nodes.get('provenance-list').children.find(r=>r.children[0].textContent==='Geometry inspection');
 assert.ok(row);assert.match(row.children[1].textContent,/Native SAM parts/);assert.equal(row.children[1].href,undefined);
 assert.equal(JSON.stringify(record),before,'public presentation must preserve original historical provenance');
 assert.match(ui.nodes.get('open-workspace').href,new RegExp('source='+record.source.sha256));
 assert.equal(ui.nodes.get('offline-queue').hidden,true);assert.equal(ui.nodes.get('cached-insights').hidden,true);
 assert.equal(ui.nodes.get('component-picker').children.length,record.components.length);
 assert.equal(ui.fetches(),0);
});

test('reusing the primary native iframe navigates eagerly before either component load',async()=>{
 const record=catalog.records.find(r=>r.scene_case.scene_ref===null&&r.sam3d_reviews?.some(c=>c.component_id==='earbud_b'));
 assert.ok(record);const ui=browserController({selectNative:true});
 for(const id of ['case_base','earbud_b']){const component=record.components.find(c=>c.id===id);assert.ok(component);await ui.selectComponent(record,component);}
 const frame=ui.nodes.get('sam3d-native-frame');assert.equal(frame.navigations.length,2);
 const standalone=new URL(ui.nodes.get('open-sam3d-component').href,'http://127.0.0.1:8766/showcase/');assert.equal(standalone.pathname,'/showcase/native-model.html');assert.equal(standalone.searchParams.get('case'),record.id);assert.equal(standalone.searchParams.get('component'),'earbud_b');
 frame.navigations.forEach((navigation,index)=>{assert.equal(navigation.loading,'eager');assert.equal(navigation.hidden,false);const url=new URL(navigation.src,'http://127.0.0.1:8766/showcase/');assert.equal(url.searchParams.get('case'),record.id);assert.equal(url.searchParams.get('component'),['case_base','earbud_b'][index]);});
});


const until=async(condition,message)=>{for(let i=0;i<150;i++){if(condition())return;await new Promise(resolve=>setImmediate(resolve));}throw Error('Saved-byte verification did not settle: '+message);};
test('curated SAM cards link to the public hash-checking viewer and never depend on raw review HTML',async()=>{
 const record=catalog.records.find(r=>r.sam3d_reviews?.length>=2),ui=browserController({withCards:true,sourceImageBytes:true});assert.ok(record);ui.controller.renderDetails(record);
 const cards=ui.nodes.get('sam3d-model-cards').children;assert.equal(cards.length,record.sam3d_reviews.length);await until(()=>cards.every(card=>card.children[0].src),'source-image SHA checks');
 for(const [index,card]of cards.entries()){const link=card.children[1].children.find(child=>child.tagName==='a');assert.ok(link);const url=new URL(link.href,'http://127.0.0.1:8766/showcase/');assert.equal(url.pathname,'/showcase/native-model.html');assert.equal(url.searchParams.get('case'),record.id);assert.equal(url.searchParams.get('component'),record.sam3d_reviews[index].component_id);assert.ok(card.children[0].src?.startsWith('blob:'),'An exact source-image digest must verify before showing the thumbnail.');URL.revokeObjectURL(card.children[0].src);}
 assert.ok(ui.requests.length>0);assert.ok(ui.requests.every(route=>route.endsWith('/source-rgb.png')),'No raw review HTML or model geometry is prefetched to mark a source card ready.');
});

test('a corrupted source image disables its model card link instead of presenting a verified thumbnail',async()=>{
 const record=catalog.records.find(r=>r.sam3d_reviews?.length),ui=browserController({withCards:true,sourceImageBytes:true,corruptSource:true});ui.controller.renderDetails(record);await until(()=>ui.nodes.get('sam3d-model-cards').children.every(card=>card.children[1].children.find(child=>child.tagName==='a').attributes['aria-disabled']==='true'),'corrupted source-image withholding');
 for(const card of ui.nodes.get('sam3d-model-cards').children){const link=card.children[1].children.find(child=>child.tagName==='a');assert.equal(link.attributes['aria-disabled'],'true');assert.equal(link.textContent,'Saved preview unavailable');assert.equal(card.children[0].src,undefined);}
 assert.ok(ui.requests.every(route=>!route.endsWith('.html')));
});

test('curated quality notes are optional bounded text and malformed notes reject the catalog',()=>{
 const ui=browserController(),copy=()=>JSON.parse(JSON.stringify(catalog));
 assert.doesNotThrow(()=>ui.controller.validateCatalog(copy()));
 const valid=copy();valid.records[0].public_note='Generated draft. Connector fit is unverified.';
 assert.doesNotThrow(()=>ui.controller.validateCatalog(valid));
 for(const value of [null,{},42,'','   ','x'.repeat(401),'Hidden\u0000note','Two\nlines']){
  const invalid=copy();invalid.records[0].public_note=value;
  assert.throws(()=>ui.controller.validateCatalog(invalid),/Invalid curated display note/);
 }
});

test('inspector shows the curated note as text and clears it on a twin without one',()=>{
 const ui=browserController(),record=JSON.parse(JSON.stringify(catalog.records[0])),before=record.summary;
 record.public_title='Phone + USB-C connector';record.public_note='Generated draft. <extra plate> remains unverified.';
 ui.controller.renderDetails(record);
 assert.equal(ui.nodes.get('detail-title').textContent,record.public_title);
 assert.equal(ui.nodes.get('detail-summary').textContent,before);
 assert.equal(ui.nodes.get('detail-quality-note').textContent,record.public_note);
 assert.equal(ui.nodes.get('detail-quality-note').hidden,false);
 assert.equal(record.summary,before,'Curated presentation must not rewrite the model description');
 delete record.public_note;ui.controller.renderDetails(record);
 assert.equal(ui.nodes.get('detail-quality-note').textContent,'');
 assert.equal(ui.nodes.get('detail-quality-note').hidden,true);
});
