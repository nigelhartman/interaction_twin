/* A camera diagnostic link names fixed evidence, never an arbitrary artifact path. */
const FIELDS={caseId:'camera_case',trackId:'camera_track',sourceSha256:'camera_source',hypothesisId:'camera_hypothesis'};
const ID=/^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$/;
function validate(binding){
 for(const key of ['caseId','trackId','hypothesisId'])if(!ID.test(binding[key]||''))throw new Error('Incomplete or invalid saved camera link. No other reconstruction was substituted.');
 if(!/^[a-f0-9]{64}$/.test(binding.sourceSha256||''))throw new Error('Incomplete or invalid camera source identity. No other reconstruction was substituted.');
 return binding;
}
export function readCameraPermalink(url){
 if(!Object.values(FIELDS).some(name=>url.searchParams.has(name)))return null;
 if(Object.values(FIELDS).some(name=>url.searchParams.getAll(name).length!==1))throw new Error('Incomplete or ambiguous camera link. No other reconstruction was substituted.');
 return validate(Object.fromEntries(Object.entries(FIELDS).map(([key,name])=>[key,url.searchParams.get(name)])));
}
export function writeCameraPermalink(url,item){
 const result=new URL(url.href);result.searchParams.set('run',item.id);
 for(const name of Object.values(FIELDS))result.searchParams.delete(name);
 if(item.cameraBinding){
  const binding=validate({...item.cameraBinding,hypothesisId:item.cameraHypothesisId});
  for(const [key,name]of Object.entries(FIELDS))result.searchParams.set(name,binding[key]);
 }
 return result;
}
