/* Local saved software checks. No fetch, network, model or processing API calls. */
(function(root){'use strict';
const MODES=new Set(['rgb','depth','faces']);
function assetURL(path,protocol=root.location?.protocol){
 if(typeof path!=='string'||!/^data\/[A-Za-z0-9_-]+\/[A-Za-z0-9_./-]+$/.test(path)||path.split('/').some(p=>!p||p==='.'||p==='..'||p.startsWith('.')))return null;
 return(protocol==='file:'?'../':'/files/')+path.split('/').map(encodeURIComponent).join('/');
}
function validReview(value){return !!(value&&value.format==='interaction-twin-native-appearance-software-review'&&value.fixture===true&&value.physical_accuracy_verified===false&&Array.isArray(value.frames)&&value.frames.length>=2&&value.frames.length<=20&&value.frames.every((f,i)=>f.index===i&&f.fixture===true&&f.known_pose===true&&[...MODES].every(m=>assetURL(f.images?.[m])))&&Array.isArray(value.examples)&&value.examples.every(x=>x.fixture===true)&&value.baseline?.fixture===false);}
function createController(review,render,clock={setInterval:root.setInterval?.bind(root),clearInterval:root.clearInterval?.bind(root)}){
 if(!validReview(review))throw new Error('The saved verification has an invalid or missing fixture contract.');
 const state={index:0,mode:'rgb',playing:false,difference:false};let timer=null;
 const snapshot=()=>({...state,frame:review.frames[state.index],count:review.frames.length});
 const emit=()=>render(snapshot());
 function seek(value){const number=Number(value);if(!Number.isFinite(number))return;state.index=Math.max(0,Math.min(review.frames.length-1,Math.round(number)));emit();}
 function pause(){if(timer!==null)clock.clearInterval(timer);timer=null;state.playing=false;emit();}
 const api={snapshot,seek,step(delta){seek(state.index+delta);},mode(value){if(MODES.has(value)){state.mode=value;emit();}},pause,
  play(){if(state.playing)return;state.playing=true;timer=clock.setInterval(()=>seek((state.index+1)%review.frames.length),650);emit();},
  toggle(){state.playing?pause():api.play();},difference(){state.difference=!state.difference;emit();},destroy(){if(timer!==null)clock.clearInterval(timer);timer=null;state.playing=false;}};
 emit();return api;
}
const exported={assetURL,validReview,createController};if(typeof module!=='undefined'&&module.exports)module.exports=exported;
root.NativeAppearanceReview=exported;if(!root.document)return;
const $=id=>root.document.getElementById(id),node=(tag,text,className)=>{const n=root.document.createElement(tag);if(text!==undefined)n.textContent=String(text);if(className)n.className=className;return n;};
function image(id,path){const url=assetURL(path);if(!url)throw new Error('A saved image path is invalid.');$(id).src=url;}
function link(path,label,className){const a=node('a',label,className);const url=assetURL(path);if(url)a.href=url;else a.removeAttribute('href');return a;}
function metric(container,value,label){const div=node('div',undefined,'metric');div.append(node('strong',value),node('span',label));container.append(div);}
function detail(container,label,value){const div=node('div');div.append(node('dt',label),node('dd',value));container.append(div);}
try{
 const payload=root.NATIVE_APPEARANCE_REVIEW,review=payload?.review;if(!validReview(review))throw new Error('Build the saved verification first with scripts/build_native_appearance_review.py.');
 const pointer=payload.pointer;if(!assetURL(pointer?.manifest_path))throw new Error('The verification pointer is invalid.');
 const manifestURL=assetURL(pointer.manifest_path);$('manifest-link').href=manifestURL;$('fixture-json').href=manifestURL;
 const controller=createController(review,s=>{
  image('fixture-image',s.frame.images[s.mode]);$('fixture-image').alt='SOFTWARE TEST FIXTURE: known calibration cube pose '+(s.index+1)+', '+s.mode+' CPU render';
  $('frame-slider').value=String(s.index);$('frame-slider').max=String(s.count-1);$('frame-number').textContent=String(s.index+1).padStart(2,'0')+' / '+String(s.count).padStart(2,'0');
  $('play').textContent=s.playing?'Pause':'Play';$('play').setAttribute('aria-pressed',String(s.playing));
  $('previous').disabled=s.index===0;$('next').disabled=s.index===s.count-1;
  for(const button of root.document.querySelectorAll('[data-mode]'))button.setAttribute('aria-pressed',String(button.dataset.mode===s.mode));
  $('mode-description').textContent={rgb:'Embedded software-test texture',depth:'Near = mint · far = violet · per-frame range',faces:'Separate primitive colours · software input'}[s.mode];
  $('pose-description').textContent='Yaw '+s.frame.yaw_deg.toFixed(1)+'° · pitch '+s.frame.pitch_deg.toFixed(1)+'°';
  image('baseline-native',review.baseline.images[s.difference?'difference':'native']);
  $('baseline-native').alt=s.difference?'Measured render difference: teal indicates unchanged rendered pixels; coral indicates changed pixels':'Native loader projection using the unchanged Jabra mesh and saved reference pose';
  $('baseline-label').textContent=s.difference?'Measured render difference':'Native loader dispatch';$('difference').textContent=s.difference?'Show native render':'Show difference';$('difference').setAttribute('aria-pressed',String(s.difference));
 });
 $('frame-slider').addEventListener('input',e=>controller.seek(e.target.value));$('previous').addEventListener('click',()=>controller.step(-1));$('next').addEventListener('click',()=>controller.step(1));$('play').addEventListener('click',controller.toggle);$('difference').addEventListener('click',controller.difference);
 for(const button of root.document.querySelectorAll('[data-mode]'))button.addEventListener('click',()=>controller.mode(button.dataset.mode));
 root.document.addEventListener('visibilitychange',()=>{if(root.document.hidden)controller.pause();});root.addEventListener('pagehide',controller.destroy);
 root.document.addEventListener('keydown',e=>{if(!['INPUT','TEXTAREA','SELECT','BUTTON','A'].includes(e.target?.tagName)&&!e.altKey&&!e.metaKey&&!e.ctrlKey){if(e.key==='ArrowLeft'){e.preventDefault();controller.step(-1);}if(e.key==='ArrowRight'){e.preventDefault();controller.step(1);}}});
 const receipt=review.fixture_renderer_receipt;
 detail($('fixture-stats'),'Known poses',review.frames.length);detail($('fixture-stats'),'Native vertices',receipt.vertex_count);detail($('fixture-stats'),'Triangles',receipt.face_count);detail($('fixture-stats'),'Model calls','0');
 for(const example of review.examples){const card=node('article',undefined,'example-card'),visual=node('div',undefined,'example-image'),img=node('img');img.src=assetURL(example.images.rgb);img.alt='SOFTWARE TEST FIXTURE: '+example.title;img.width=480;img.height=320;visual.append(img,node('span','SOFTWARE TEST FIXTURE','fixture-label'));const copy=node('div',undefined,'example-copy');copy.append(node('h3',example.title),node('p',example.summary));const foot=node('div',undefined,'example-foot');foot.append(node('span',example.id==='seams'?example.vertices+' vertices / '+example.unique_positions+' positions':example.visible_triangles+' visible triangles'),link(example.images.raw_depth,'Raw depth ↗'));copy.append(foot);card.append(visual,copy);$('example-cards').append(card);}
 image('baseline-source',review.baseline.images.source);image('baseline-legacy',review.baseline.images.legacy);$('source-time').textContent=review.baseline.source_timestamp_s.toFixed(3)+' s';
 const m=review.baseline.metrics;$('baseline-status').textContent=m.pixel_equality?'EXACT PIXEL MATCH':'INSPECT DIFFERENCE';$('baseline-status').classList.toggle('caution',!m.pixel_equality);
 metric($('baseline-metrics'),m.changed_rgb_pixels,'Changed RGB pixels');metric($('baseline-metrics'),m.rgb_max_absolute_difference_uint8,'Maximum RGB difference / 255');metric($('baseline-metrics'),m.mask_mismatched_pixels,'Mismatched silhouette pixels');metric($('baseline-metrics'),m.depth_arrays_exactly_equal_including_background?'Exact':'Different','Depth arrays, including background');
 detail($('hashes'),'Test asset SHA256',review.fixture_asset.sha256);detail($('hashes'),'Jabra asset SHA256',review.baseline.asset.sha256);detail($('hashes'),'Renderer source SHA256',receipt.loader_source_sha256);detail($('hashes'),'Decoder source SHA256',receipt.decoder_source_sha256);detail($('hashes'),'Verification JSON SHA256',pointer.manifest_sha256);
 if(review.real_fit){const fit=review.real_fit;$('real-fit-section').hidden=false;image('real-fit-preview',fit.preview_path);metric($('real-fit-metrics'),fit.wall_seconds.toFixed(1)+' s','Actual bounded CPU fit');metric($('real-fit-metrics'),fit.reference_iou.toFixed(3),'Same-source silhouette IoU');metric($('real-fit-metrics'),'Unverified','Physical pose');$('real-fit-reasons').textContent='Physical acceptance withheld: '+fit.gate_reasons.join('; ').replaceAll('_',' ')+'.';$('real-fit-links').append(link(fit.receipt_path,'Actual run receipt ↗'),link(fit.experiment_path,'Fit evidence ↗'));if(fit.tracking_observer_path)$('real-fit-links').append(link(fit.tracking_observer_path,'Actual CPU tracking observer ↗'));}
 if(review.native_cpu_upgrade){const run=review.native_cpu_upgrade,c=run.comparison;const paragraph=node('p',c?'A completed native CPU upgrade is also available: '+c.baseline.conditional_supported+' → '+c.candidate.conditional_supported+' conditional supports on '+c.matched_sample_count+' common non-reference timestamps. These internal checks do not measure physical accuracy.':'A completed native CPU upgrade is also available. Its saved receipt preserves partial results and conditional motion.','fine-print');$('real-fit-links').after(paragraph);$('real-fit-links').append(link(run.receipt_path,'Completed CPU upgrade receipt ↗'));if(run.observer_path)$('real-fit-links').append(link(run.observer_path,'CPU upgrade motion ↗'));}
 $('load-status').textContent='Saved CPU checks · '+review.frames.length+' known-pose fixture frames · real Jabra compatibility measured separately · no model inference for this review.';
}catch(error){$('load-status').textContent=error.message||String(error);$('load-status').classList.add('error');}
})(typeof window!=='undefined'?window:globalThis);
