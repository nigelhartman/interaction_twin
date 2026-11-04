/* Public copy controls never contact a model, MCP server or installer. */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const code = fs.readFileSync(path.join(root, 'showcase/agents.js'), 'utf8');
function setup(clipboard, execCommand) {
  const nodes = new Map();
  function get(id) {
    if (!nodes.has(id)) nodes.set(id, {value:'',textContent:'',disabled:false,attrs:{},handlers:{},
      addEventListener(name, fn){this.handlers[name]=fn;},setAttribute(k,v){this.attrs[k]=v;},
      focus(){this.focused=true;},select(){this.selected=true;}});
    return nodes.get(id);
  }
  const context = {document:{getElementById:get,execCommand},location:{protocol:'https:',origin:'https://twins.example'},navigator:{clipboard},
    fetch(){throw Error('Public copy helpers must not make requests');}};
  vm.runInNewContext(code,context);
  get('charge-example-prompt').value='Prepare a task to charge this phone.';
  return {get};
}
(async()=>{
  const copied=[];
  const client=setup({writeText:async text=>copied.push(text)});
  assert.match(client.get('agent-prompt').value,/https:\/\/twins\.example\/showcase\/agent-setup\.html/);
  assert.equal(client.get('agent-prompt').value,'Connect to Interaction Twin: https://twins.example/showcase/agent-setup.html');
  await client.get('copy-agent-prompt').handlers.click();
  assert.equal(copied[0],client.get('agent-prompt').value);
  assert.equal(client.get('prompt-copy-status').textContent,'Copied.');
  assert.equal(client.get('copy-agent-prompt').attrs['data-copy-state'],'copied');
  await client.get('copy-charge-prompt').handlers.click();
  assert.equal(copied[1],'Prepare a task to charge this phone.');
  assert.equal(client.get('charge-copy-status').textContent,'Copied.');
  const fallback=setup(undefined);
  await fallback.get('copy-agent-prompt').handlers.click();
  assert.equal(fallback.get('agent-prompt').selected,true);
  assert.equal(fallback.get('agent-prompt').focused,true);
  assert.equal(fallback.get('copy-agent-prompt').attrs['data-copy-state'],'idle');
  assert.match(fallback.get('prompt-copy-status').textContent,/Ctrl\+C/);
  const legacyCommands=[];
  const legacy=setup(undefined, command=>{legacyCommands.push(command);return true;});
  await legacy.get('copy-agent-prompt').handlers.click();
  assert.deepEqual(legacyCommands,['copy']);
  assert.equal(legacy.get('agent-prompt').selected,true);
  assert.equal(legacy.get('prompt-copy-status').textContent,'Copied.');
  assert.equal(legacy.get('copy-agent-prompt').attrs['data-copy-state'],'copied');
  const denied=setup({writeText:async()=>{throw Error('Permission denied');}},()=>{throw Error('Copy unavailable');});
  await denied.get('copy-agent-prompt').handlers.click();
  assert.equal(denied.get('agent-prompt').selected,true);
  assert.match(denied.get('prompt-copy-status').textContent,/⌘C on Mac or Ctrl\+C/);
  let release;
  const pending=setup({writeText:()=>new Promise(resolve=>{release=resolve;})});
  const copying=pending.get('copy-charge-prompt').handlers.click();
  pending.get('charge-example-prompt').value='Updated task';release();await copying;
  assert.match(pending.get('charge-copy-status').textContent,/field changed/);
  const page=fs.readFileSync(path.join(root,'showcase/agents.html'),'utf8');
  assert.ok(!page.includes('/Users/nigelhartman/'));
  assert.ok(!page.includes('--private-owner'));
  assert.ok(!page.includes('mcp-checkout-path'));
  assert.ok(!page.includes('mcp-config'));
  assert.ok(!page.includes('robot-task-kit'));
  assert.ok(!page.includes('See the task draft'));
  assert.ok(!page.includes('Try the programmable'));
  assert.match(page,/href="agent-setup.html"/);
  assert.match(page,/id="charge-phone-example"/);
  assert.match(page,/id="copy-charge-prompt" type="button" aria-label="Copy phone task prompt"/);
  assert.match(page,/does not operate a robot/);
  assert.match(page,/href="robotics-lab.html"/);
  // Native button semantics provide Enter/Space activation and remain inside
  // the same highlighted field; the SVGs are decorative to assistive tech.
  assert.match(page,/<div class="agent-connection-field">\s*<input id="agent-prompt"[^>]+>\s*<button class="agent-copy-icon" id="copy-agent-prompt" type="button" aria-label="Copy agent connection"/);
  assert.match(page,/class="agent-copy-glyph"[^>]+aria-hidden="true" focusable="false"/);
  assert.match(page,/id="prompt-copy-status"[^>]+role="status" aria-live="polite"/);
  assert.ok(!page.includes('>Copy connection</button>'));
  assert.ok(!page.includes('Read-only JSON'));
  assert.ok(!page.includes('See an example response'));
  assert.ok(!page.includes('Available filters'));
  assert.ok(!page.includes('<form'));
  const css=fs.readFileSync(path.join(root,'showcase/agents.css'),'utf8');
  assert.match(css,/\.agent-connection-field\{[^}]*background:#edf2ff/);
  assert.match(css,/\.agent-copy-icon:focus-visible/);
  console.log('PASS: two accessible inline copy controls, clipboard/legacy/keyboard fallback, race handling, simplified page and request-free copy scope.');
})().catch(error=>{console.error(error);process.exitCode=1;});
