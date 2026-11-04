const test=require('node:test'),assert=require('node:assert/strict');
const A=require('../showcase/auto-scene-camera-v1.js');
test('manual camera has a three-second idle window and a smooth bounded return',()=>{
 const c=A.create();assert.deepEqual(c.state(0),{phase:'source',progress:1});
 c.activity(50);assert.deepEqual(c.state(3049),{phase:'manual',progress:0});
 assert.deepEqual(c.state(3050),{phase:'return',progress:0});
 assert.deepEqual(c.state(3475),{phase:'return',progress:.5});
 assert.deepEqual(c.state(3900),{phase:'source',progress:1});
 c.activity(3500);assert.equal(c.state(3900).phase,'manual');
 c.reset();assert.equal(c.state(3900).phase,'source');
});
test('invalid clocks and unbounded timings are rejected',()=>{
 for(const v of[NaN,Infinity,-1])assert.throws(()=>A.create({idle_ms:v}));
 for(const v of[NaN,Infinity,0,-1])assert.throws(()=>A.create({return_ms:v}));
 const c=A.create();for(const v of[NaN,Infinity,'1']){assert.throws(()=>c.activity(v));assert.throws(()=>c.state(v));}
});
