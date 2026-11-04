const test=require('node:test');
const assert=require('node:assert/strict');
const {selection}=require('../workspace/native-default.js');
const models=require('../showcase/sam3d-models.js');
function ref(name){return{path:'data/native/'+name,sha256:'b'.repeat(64),bytes:100};}
function fixture(){const review={component_id:'part_a',source_sha256:'a'.repeat(64),selected:false,physical_accuracy_verified:false,
 review_page:ref('index.html'),source_image:ref('image.png'),source_mask:ref('mask.png'),native_glb:ref('asset.glb')};
 return{records:[{id:'source-family',source:{sha256:'a'.repeat(64),original_sha256:'c'.repeat(64)},components:[{id:'part_a'},{id:'part_b'}],sam3d_reviews:[review]}]};}
test('native selection matches the normalized source SHA and exact component',()=>{
 const c=fixture();assert.equal(selection(c,'a'.repeat(64),'part_a',models).review,c.records[0].sam3d_reviews[0]);
 assert.equal(selection(c,'c'.repeat(64),'part_a',models).review,null);
 assert.equal(selection(c,'a'.repeat(64),'part_b',models).review,null);
 assert.equal(selection(c,'a'.repeat(64),'unknown',models).review,null);
});
test('ineligible or wrong-source native reviews cannot substitute a mesh',()=>{
 for(const [field,value]of[['selected',true],['physical_accuracy_verified',true],['source_sha256','d'.repeat(64)],['component_id','another']]){
  const c=fixture();c.records[0].sam3d_reviews[0][field]=value;
  assert.equal(selection(c,'a'.repeat(64),'part_a',models).review,null,field);
 }
});
test('missing native model remains empty even if a historical asset exists',()=>{
 const c=fixture();c.records[0].sam3d_reviews=[];c.records[0].components[0].asset=ref('old-comparison.glb');
 assert.equal(selection(c,'a'.repeat(64),'part_a',models).review,null);
 assert.equal(selection(null,'a'.repeat(64),'part_a',models).record,null);
});
