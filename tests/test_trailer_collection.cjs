const test=require('node:test'),assert=require('node:assert/strict');
const collection=require('../showcase/trailer-collection.js');
test('six story choices use distinct media and three deliberate compositions',()=>{
  const films=collection.FILMS;
  assert.equal(films.length,6);assert.equal(new Set(films.map(f=>f.id)).size,6);
  assert.equal(new Set(films.map(f=>f.genre)).size,6);
  assert.equal(new Set(films.map(f=>collection.media(f.id).video)).size,6);
  assert.deepEqual(collection.filtered('portrait').map(f=>f.id),['pocket-worlds']);
  assert.deepEqual(collection.filtered('square').map(f=>f.id),['agent']);
  assert.equal(collection.filtered('landscape').length,4);
  for(const film of films){assert.equal(film.arc.length,3);assert.ok(film.description.length>70);for(const path of Object.values(collection.media(film.id)))assert.match(path,/^media\/trailer-collection-20261104-v1\/\d{2}-[a-z-]+(?:-poster\.jpg|-transcript\.txt|\.mp4|\.vtt)$/);}
});
test('next and previous film navigation wraps; unrecognized URL/media choices fail closed',()=>{
  assert.equal(collection.adjacent('origin',-1).id,'agent');
  assert.equal(collection.adjacent('agent',1).id,'origin');
  assert.equal(collection.adjacent('blueprint',1).id,'mission');
  for(const bad of['../private','https://example.com','phone',undefined,null])assert.throws(()=>collection.media(bad),/Unknown trailer/);
  for(const bad of['16:9','external',undefined])assert.throws(()=>collection.filtered(bad),/Unknown film format/);
  for(const bad of[0,2,NaN,'1'])assert.throws(()=>collection.adjacent('origin',bad),/Invalid film direction/);
});
