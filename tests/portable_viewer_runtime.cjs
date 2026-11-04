/* Actual shipped viewer+scene executed with DOM/WebGL contract doubles.
 * Checks state/control/render decisions, not browser pixels or GPU correctness. */
'use strict';
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const data=JSON.parse(fs.readFileSync(process.argv[2],'utf8')),dir=process.argv[3];
let drawCalls=0,networkAttempts=0;
const gl=new Proxy({},{get(_target,key){
 if(/^[A-Z_0-9]+$/.test(key))return 1;
 if(key==='clear')return()=>{drawCalls=0};
 if(key==='drawArrays')return()=>{drawCalls++};
 if(key==='getShaderParameter'||key==='getProgramParameter')return()=>true;
 if(key==='getAttribLocation')return()=>0;
 if(key.startsWith('create')||key==='getUniformLocation')return()=>({});
 return()=>{};
}});
class Element{
 constructor(tag,id=''){this.tagName=tag;this.id=id;this.children=[];this.listeners={};this.hidden=false;this.value='';this.disabled=false;this.clientWidth=600;this.clientHeight=400;this.textContent='';this.classes=new Set();this.classList={toggle:(name,on)=>{if(on)this.classes.add(name);else this.classes.delete(name)}};}
 append(...children){for(const child of children){this.children.push(child);if(this.tagName==='select'&&this.children.length===1)this.value=child.value}}
 replaceChildren(...children){this.children=[];this.append(...children)}
 addEventListener(name,fn){(this.listeners[name]??=[]).push(fn)}
 removeEventListener(){}
 dispatch(name){for(const fn of this.listeners[name]||[])fn({target:this,preventDefault(){}})}
 getContext(){return gl}
}
const tags={scene:'canvas',component:'select',samples:'select',video:'video',evidence:'img',timeline:'input'};
const ids=['title','summary','clock','timeline','samples','previous','next','sample-status','video','evidence','missing-evidence','evidence-note','evidence-label','masks','component','scene','native','motion','fit','wire','pose-status','mesh-stats','steps','states','relations','assets','program-provenance','program','limitations'];
const elements=Object.fromEntries(ids.map(id=>[id,new Element(tags[id]||'div',id)]));
const document={getElementById:id=>{assert.ok(elements[id],'Unknown DOM id '+id);return elements[id]},createElement:tag=>new Element(tag),title:''};
const noNetwork=()=>{networkAttempts++;throw Error('Network access forbidden')};
const context={document,PORTABLE_TWIN:data,console,Float32Array,Uint8Array,Set,Map,Math,Number,Array,Error,
 devicePixelRatio:1,addEventListener(){},removeEventListener(){},fetch:noNetwork,WebSocket:noNetwork,XMLHttpRequest:noNetwork,
 Image:class{set src(value){assert.ok(value.startsWith('data:image/'),'Texture must be embedded');this._src=value}}};
vm.createContext(context);
vm.runInContext(fs.readFileSync(dir+'/scene.js','utf8'),context,{filename:'scene.js'});
vm.runInContext(fs.readFileSync(dir+'/viewer.js','utf8'),context,{filename:'viewer.js'});
const runtime=context.PORTABLE_RUNTIME;
const first=data.evidence[0].timestamp_s,second=data.evidence[1].timestamp_s,last=data.evidence.at(-1).timestamp_s,gap=(first+second)/2;
assert.ok(runtime,'Actual viewer initialization completed');
assert.equal(runtime.mode,'native');
assert.ok(elements.native.classes.has('active'));
assert.ok(elements['pose-status'].textContent.includes('Native asset'));
assert.ok(drawCalls>0,'Initial native asset draws actual triangles');
function text(element){return element.textContent+' '+element.children.map(text).join(' ')}
assert.ok(text(elements.states).includes('closed'),'Initial exact sample state');
elements.next.dispatch('click');
assert.equal(runtime.time,second);assert.ok(text(elements.states).includes('open'));
elements.previous.dispatch('click');assert.equal(runtime.time,first);
elements.samples.value=String(second);elements.samples.dispatch('change');assert.equal(runtime.time,second);
elements.motion.dispatch('click');assert.equal(runtime.mode,'component');
assert.ok(elements['pose-status'].textContent.includes('Conditional camera-relative pose'));
assert.ok(drawCalls>0,'Supported exact conditional sample draws geometry');
elements.component.value='holder';elements.component.dispatch('change');
assert.equal(drawCalls,0,'Selected component with no pose stays hidden');
assert.ok(elements['pose-status'].textContent.includes('No supported pose'));
elements.component.value='part';elements.component.dispatch('change');assert.ok(drawCalls>0);
elements.timeline.value=String(gap);elements.timeline.dispatch('input');
assert.equal(runtime.time,gap);assert.equal(drawCalls,0,'Gap never renders a held/interpolated pose');
assert.ok(text(elements.states).includes('unknown'));assert.ok(!text(elements.states).includes('closed'));
assert.ok(elements['sample-status'].textContent.includes('Between evidence samples'));
assert.ok(elements['evidence-note'].textContent.includes('Nearest declared sample for inspection only'));
elements.next.dispatch('click');assert.equal(runtime.time,second,'Next from a gap must use the first following sample');
elements.timeline.value=String(gap);elements.timeline.dispatch('input');elements.previous.dispatch('click');
assert.equal(runtime.time,first,'Previous from a gap must use the last preceding sample');
elements.timeline.value=String(data.source.duration_s);elements.timeline.dispatch('input');elements.previous.dispatch('click');
assert.equal(runtime.time,last,'Previous after the last sample must select that last sample');
elements.native.dispatch('click');assert.ok(drawCalls>0,'Native inspection independent from motion coverage');
elements.steps.children[0].dispatch('click');assert.equal(runtime.time,first);
elements.samples.value=String(last);elements.samples.dispatch('change');elements.motion.dispatch('click');
assert.equal(drawCalls,0,'Withheld sample does not draw geometry');
assert.ok(text(elements.states).includes('unknown'));
elements.wire.checked=true;elements.wire.dispatch('change');elements.fit.dispatch('click');
assert.equal(networkAttempts,0,'Viewer never calls external services');
assert.equal(elements.video.src,undefined,'Source video not implicitly requested');
assert.ok(elements.evidence.src.startsWith('evidence/'),'Only package-relative evidence URL');
assert.equal(elements.program.href,data.program.path);
console.log(JSON.stringify({status:'passed',checks:['initial_native_mesh','exact_state','source_navigation','gap_navigation_no_skipped_samples','previous_after_last_sample','selected_component_motion','gap_pose_hidden','withheld_pose_hidden','unknown_state','nearest_image_label','no_network'],browser_pixel_validation:false}));
