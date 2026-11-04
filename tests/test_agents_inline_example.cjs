'use strict';
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const {webcrypto}=require('node:crypto');
const root=path.resolve(__dirname,'..');
const code=fs.readFileSync(path.join(root,'showcase/agent-charge-example.js'),'utf8');
function fixture(tamper){
  const nodes=new Map(),requests=[],urls=[];
  function node(tag='div'){
    return {tag,children:[],textContent:'',hidden:true,attrs:{},append(...items){this.children.push(...items);},setAttribute(k,v){this.attrs[k]=v;}};
  }
  const get=id=>{if(!nodes.has(id))nodes.set(id,node());return nodes.get(id);};
  const context={document:{getElementById:get,createElement:node,createDocumentFragment:()=>node('fragment')},crypto:webcrypto,TextDecoder,Uint8Array,Blob,
    URL:{createObjectURL(){const url='blob:verified-'+urls.length;urls.push(url);return url;},revokeObjectURL(){}},addEventListener(){},
    async fetch(url,options){
      requests.push({url,options});
      assert.deepEqual({...options},{credentials:'omit',cache:'no-cache'});
      assert.ok(!url.includes('/api/')&&!url.includes('http:')&&!url.includes('https:'));
      const file=url.startsWith('/files/data/')?url.slice('/files/'.length):'showcase/'+url;
      let raw=fs.readFileSync(path.join(root,file));
      if(tamper?.(url))raw=Buffer.concat([raw,Buffer.from(' ')]);
      return {ok:true,arrayBuffer:async()=>raw.buffer.slice(raw.byteOffset,raw.byteOffset+raw.byteLength)};
    }};
  vm.runInNewContext(code,context);
  return {context,get,requests,urls};
}
(async()=>{
  const success=fixture();await success.context.InteractionTwinChargeExample.ready;
  assert.equal(success.get('charge-example-content').hidden,false);
  assert.equal(success.get('charge-example-status').hidden,true);
  assert.equal(success.requests.length,7);
  assert.equal(success.urls.length,3);
  assert.match(success.get('charge-task-summary').textContent,/5 proposed steps · 3 connection candidates/);
  assert.match(success.get('charge-trace-summary').textContent,/42 calls.*6 original model files.*9 source frames/);
  assert.match(success.get('charge-review-summary').textContent,/separate Meta Muse metadata critique/);
  const cards=success.get('charge-example-assets').children[0].children;
  assert.equal(cards.length,3);
  for(const card of cards){const anchor=card.children[0];assert.match(anchor.href,/^\/showcase\/library.html\?case=flow-capture-(phone|charger)-20261103-v1--flow-procedure-1$/);assert.match(anchor.children[0].src,/^blob:verified-/);}
  for(const target of ['task.json','trace.json','summary.json','muse-review.json','/frames/000057.jpg']){
    const broken=fixture(url=>url.endsWith(target));await broken.context.InteractionTwinChargeExample.ready;
    assert.equal(broken.get('charge-example-content').hidden,true,target);
    assert.match(broken.get('charge-example-status').textContent,/could not be verified/,target);
    assert.equal(broken.get('charge-example-assets').children.length,0,target);
  }
  const page=fs.readFileSync(path.join(root,'showcase/agents.html'),'utf8');
  assert.ok(page.indexOf('class="agent-flow"')<page.indexOf('id="charge-phone-example"'));
  assert.equal((page.match(/class="agent-example-details"/g)||[]).length,1);
  assert.ok(!code.includes('innerHTML'));
  console.log('PASS: exact saved42-call MCP + separate Muse lineage,3 verified source images, inline task,7 bounded read-only fetches and five fail-closed tamper checks.');
})().catch(error=>{console.error(error);process.exitCode=1;});
