/* Actual public controller/catalog verification; DOM doubles are not pixel QA. */
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const H=require('./showcase_app_harness.cjs'),root=path.resolve(__dirname,'..');
const wallet=H.payload.cases.find(c=>c.id==='flow-rigid-parts-wallet-20261031-v1--flow-procedure-1');
const jabra=H.payload.cases.find(c=>c.interaction.source.sha256===H.JABRA_SOURCE);
function publicReadsOnly(env){const catalog=JSON.parse(fs.readFileSync(path.join(root,'showcase/public/library-catalog.json'),'utf8')),programs=new Set(catalog.records.map(r=>'/files/'+r.interaction.path));assert.ok(env.requests.every(route=>route==='/health'||route==='/showcase/public/library-index.json'||/^\/showcase\/public\/library-catalog-[a-f0-9]{64}\.json$/.test(route)||programs.has(route)),'Public selection may read only curated hash-bound programs, catalog and health.');assert.equal(env.legacyScenes.length,0);}
async function opened(search){const e=H.setup(false,{search});await e.initialPromise;await H.flush();return e;}

test('the default Jabra sample resolves by source SHA from the real curated cases only',async()=>{
 assert.ok(jabra);assert.notEqual(jabra.id,'flow-rigid-parts-jabra-20261031-v1--flow-procedure-1');
 const catalog=JSON.parse(fs.readFileSync(path.join(root,'showcase/public/library-catalog.json'),'utf8'));
 assert.deepEqual(H.payload.cases.map(c=>c.id).sort(),catalog.records.map(r=>r.id).sort());
 for(const c of H.payload.cases){const record=catalog.records.find(r=>r.id===c.id);assert.equal(c.original_sha256,record.interaction.sha256);assert.equal(c.interaction.source.sha256,record.source.sha256);}
 const e=await opened('');assert.equal(H.selected(e,'jabra'),true);assert.equal(e.get('sample-video').src,'/files/'+jabra.video_path);
 const model=new URL(e.get('sample-native-frame').src,'http://127.0.0.1:8765/showcase/');assert.equal(model.searchParams.get('source'),H.JABRA_SOURCE);
 assert.equal(e.get('sample-workspace').href,'library.html?case='+encodeURIComponent(jabra.id));publicReadsOnly(e);
});

test('a curated source and fractional source time load the exact source once, then ordinary selection resets time',async()=>{
 const e=await opened('?'+new URLSearchParams({source:wallet.interaction.source.sha256,t:'9.125'}));
 assert.equal(e.get('sample-panel').attributes['aria-labelledby'],'sample-title');assert.equal(e.get('sample-video').src,'/files/'+wallet.video_path);assert.equal(e.get('sample-video').currentTime,9.125);
 assert.equal(e.get('sample-workspace').href,'library.html?case='+encodeURIComponent(wallet.id));
 const model=new URL(e.get('sample-native-frame').src,'http://127.0.0.1:8765/showcase/');assert.equal(model.searchParams.get('source'),wallet.interaction.source.sha256);
 assert.ok(e.tabs.every(tab=>tab.attributes['aria-selected']==='false'),'A linked recording must not falsely select a different preset.');
 await e.tabs[0].onclick();await H.flush();assert.equal(H.selected(e,'jabra'),true);assert.equal(e.get('sample-video').currentTime,0,'The deep-linked time is consumed rather than applied to later sources.');publicReadsOnly(e);
});

test('source-time links clamp to the curated source duration and ignore invalid time syntax',async()=>{
 const clamped=await opened('?'+new URLSearchParams({source:wallet.interaction.source.sha256,t:String(wallet.duration_s+100)}));assert.equal(clamped.get('sample-video').currentTime,wallet.duration_s);publicReadsOnly(clamped);
 for(const t of['-1','NaN','Infinity','1e9','1.2.3','']){const e=await opened('?'+new URLSearchParams({source:wallet.interaction.source.sha256,t}));assert.equal(e.get('sample-video').currentTime,0,t);publicReadsOnly(e);}
});

test('uncurated or malformed source IDs cannot select a private recording or issue a new fetch',async()=>{
 for(const source of['f'.repeat(64),'data/private/source.mp4','https://example.com/private','7474']){const e=await opened('?'+new URLSearchParams({source}));assert.equal(H.selected(e,'jabra'),true);assert.equal(e.get('sample-video').src,'/files/'+jabra.video_path);publicReadsOnly(e);}
});
