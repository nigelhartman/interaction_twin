/* Execute production insights.js against a minimal browser contract.
 * All fetches are explicitly resolved local test promises; no server/model/API.
 */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../workspace/insights.js'), 'utf8');
const scenario = process.argv[2];

function freeze(value) {
  if (value && typeof value === 'object') {
    Object.freeze(value);
    Object.values(value).forEach(freeze);
  }
  return value;
}
class Element {
  constructor(tag = 'div') {
    this.tagName = tag.toUpperCase(); this.children = []; this.dataset = {};
    this._text = ''; this._value = undefined; this.disabled = false;
    this.classes = new Set();
    this.classList = {toggle: (key, yes) => yes ? this.classes.add(key) : this.classes.delete(key)};
  }
  set textContent(value) {this._text = String(value); this.children = [];}
  get textContent() {return this._text + this.children.map(n => typeof n === 'string' ? n : n.textContent).join('');}
  set disabled(value) {this._disabled = Boolean(value);}
  get disabled() {return this._disabled;}
  set value(value) {this._value = String(value);}
  get value() {return this._value ?? (this.tagName === 'SELECT' ? this.children[0]?.value ?? '' : '');}
  append(...nodes) {this.children.push(...nodes);}
  replaceChildren(...nodes) {this.children = nodes; this._text = ''; if (this.tagName === 'SELECT') this._value = undefined;}
  querySelector(selector) {return walk(this).find(n => selector.startsWith('.') && n.className?.split(' ').includes(selector.slice(1))) ?? null;}
  focus() {this.focused = true;}
  click() {if (!this.disabled) return this.onclick?.();}
}
function walk(node) {return node.children.flatMap(n => typeof n === 'string' ? [] : [n, ...walk(n)]);}
function project(id = 'alpha') {
  const evidence = [
    {id: id + '_source_1', timestamp_s: 1.234567, frame_path: 'data/' + id + '/frame1.jpg', observations: [{track_id: 'part', audit_status: 'plausible_visible_mask', mask_path: 'data/' + id + '/mask1.png'}]},
    {id: id + '_source_2', timestamp_s: 4.25, frame_path: 'data/' + id + '/frame2.jpg', observations: [{track_id: 'part', mask_path: 'data/' + id + '/mask2.png'}]},
    {id: id + '_source_3', timestamp_s: 8.125, frame_path: 'data/' + id + '/frame3.jpg', observations: [{track_id: 'part', audit_status: 'rejected_wrong_object'}]},
    {id: id + '_source_4', timestamp_s: 10.5, frame_path: 'data/' + id + '/frame4.jpg', observations: [{track_id: 'holder', audit_status: 'plausible_visible_mask'}]},
    {id: id + '_source_5', timestamp_s: 15.75, frame_path: 'data/' + id + '/frame5.jpg', observations: []},
  ];
  return freeze({id, labels: {conditional_poses: 2}, interaction: {
    summary: id + ' original procedure', tracklets: [{id: 'part', label: 'Part'}, {id: 'holder', label: 'Holder'}],
    steps: [{id: 'step1', title: id + ' recorded step'}], evidence,
    relationships: [{subject_track_id: 'part', object_track_id: 'holder', predicate: 'removable_from', status: 'inferred', evidence_ids: [id + '_source_2']}],
  }});
}
function analysis(id) {return {result: {title: id + ' analysis', summary: id + ' grounded summary', evidence_ids: [id + '_source_1'], claims: [], step_insights: []}};}
function answer(question, id, extra = {}) {return {contract: {question}, result: {answer: id + ' answer: ' + question, answer_type: 'evidence_based', status: 'visual_model_proposal', evidence_ids: [id + '_source_2'], limitations: ['Hidden contact remains unknown.']}, ...extra};}
function snapshot(id, extra = {}) {return freeze({analysis: analysis(id), answers: [], readiness: {checks: [{id: 'contact', title: 'Contact verification', status: 'unknown'}]}, job: null, ...extra});}

function setup(initial = project()) {
  let active = initial, timerId = 0;
  const nodes = new Map(), events = new Map(), pending = [], timers = new Map(), seeks = [], tabs = [], blobs = [];
  const get = id => {
    if (!nodes.has(id)) {
      const node = new Element(['intel-track-filter', 'intel-audit-filter', 'intel-answer-history'].includes(id) ? 'select' : 'div');
      node.id = id; nodes.set(id, node);
    }
    return nodes.get(id);
  };
  get('intel-audit-filter').value = 'all';
  const W = {get active() {return active;}, seek: value => seeks.push(value), tab: value => tabs.push(value), url: value => '/files/' + value};
  const context = {
    console, Blob, window: {TwinWorkspace: W},
    document: {getElementById: get, createElement: tag => new Element(tag), querySelectorAll: () => [], addEventListener: (name, fn) => events.set(name, fn)},
    URL: {createObjectURL: blob => {blobs.push(blob); return 'blob:offline-' + blobs.length;}, revokeObjectURL() {}},
    setTimeout: (fn, delay) => {const id = ++timerId; timers.set(id, {fn, delay}); return id;},
    clearTimeout: id => timers.delete(id),
    fetch: (route, options = {}) => new Promise((resolve, reject) => pending.push({route, options, resolve, reject, done: false})),
  };
  vm.runInNewContext(source, context, {filename: 'workspace/insights.js'});
  function take(method = 'GET', route) {
    const item = pending.find(p => !p.done && (p.options.method || 'GET') === method && (route === undefined || p.route === route));
    assert(item, 'Expected pending ' + method + ' ' + route); item.done = true; return item;
  }
  function respond(item, data, ok = true) {item.resolve({ok, json: async () => data});}
  return {get, nodes, pending, timers, seeks, tabs, blobs, take, respond,
    switchProject(next) {active = next; events.get('twin:casechange')();},
    tick() {const next = timers.entries().next().value; assert(next, 'Expected a scheduled poll'); timers.delete(next[0]); return next[1].fn();},
    submit(question) {get('intel-question').value = question; get('intel-question-form').onsubmit({preventDefault() {}});},
  };
}
const flush = async () => {await new Promise(resolve => setImmediate(resolve));};
const posts = env => env.pending.filter(p => p.options.method === 'POST');
async function load(env, data = snapshot('alpha')) {env.respond(env.take('GET'), data); await flush();}
function text(env, id) {return env.get(id).textContent;}

async function run() {
  if (scenario === 'stale_cached_project') {
    const env = setup(); await load(env);
    env.submit('Question for alpha');
    env.respond(env.take('POST'), {status: 'cached'}); await flush();
    const stale = env.take('GET', '/api/intelligence?case=alpha');
    env.switchProject(project('beta'));
    await load(env, snapshot('beta', {answers: [answer('Question for beta', 'beta')]}));
    const betaBefore = text(env, 'intel-overview') + text(env, 'intel-answer');
    env.respond(stale, snapshot('alpha', {answers: [answer('Question for alpha', 'alpha')]})); await flush();
    assert.equal(text(env, 'intel-overview') + text(env, 'intel-answer'), betaBefore, 'Stale cached GET must not render into another project');
    assert.doesNotMatch(text(env, 'intel-answer'), /alpha/);
    env.get('intel-export').click();
    const downloaded = JSON.parse(await env.blobs[0].text());
    assert.equal(downloaded.case_id, 'beta');
    assert.equal(downloaded.answers[0].contract.question, 'Question for beta');
    assert.equal(posts(env).length, 1);
  } else if (scenario === 'stale_submit_project') {
    const env = setup(); await load(env);
    env.submit('Old project question'); const old = env.take('POST');
    env.switchProject(project('beta')); await load(env, snapshot('beta'));
    env.respond(old, {status: 'queued'}); await flush();
    assert.equal(env.timers.size, 0, 'An old paid response must not schedule polling for the new project');
    assert.match(text(env, 'intel-overview'), /beta analysis/);
    assert.equal(env.get('intel-ask').disabled, false);
  } else if (scenario === 'duplicate_paid_submit') {
    const env = setup(); await load(env);
    env.submit('Can we prove seating?'); env.submit('Can we prove seating?');
    assert.equal(posts(env).length, 1, 'Repeated form submission must reserve only one request');
    assert.equal(env.get('intel-ask').disabled, true); assert.equal(env.get('intel-analyze').disabled, true);
    const submitted = posts(env)[0];
    assert.deepEqual(JSON.parse(submitted.options.body), {case_id: 'alpha', mode: 'ask', question: 'Can we prove seating?', hosted_reasoning_consent: true});
    env.respond(env.take('POST'), {status: 'queued'}); await flush();
    env.submit('Another question'); env.get('intel-analyze').click();
    assert.equal(posts(env).length, 1, 'Queued work must keep both paid actions guarded');
    env.tick();
    env.respond(env.take('GET'), snapshot('alpha', {answers: [answer('Can we prove seating?', 'alpha')], job: {status: 'complete', mode: 'ask'}})); await flush();
    assert.equal(env.get('intel-ask').disabled, false); assert.equal(env.get('intel-analyze').disabled, false);
    assert.match(text(env, 'intel-answer'), /Can we prove seating/);
  } else if (scenario === 'immutable_question_history') {
    const env = setup(), original = snapshot('alpha', {answers: [answer('Canonical first question', 'alpha', {question: 'Mutable legacy display'}), answer('Canonical second question', 'alpha')]});
    await load(env, original);
    const select = env.get('intel-answer-history');
    assert.deepEqual(select.children.map(o => o.textContent), ['Canonical first question', 'Canonical second question']);
    assert.match(text(env, 'intel-answer'), /Canonical second question/);
    select.value = '0'; select.onchange();
    assert.match(text(env, 'intel-answer'), /Canonical first question/);
    assert.equal(original.answers[0].contract.question, 'Canonical first question');
    assert.equal(original.answers[0].question, 'Mutable legacy display');
    assert(Object.isFrozen(original.answers[0].contract));
    env.get('intel-export').click();
    const saved = JSON.parse(await env.blobs[0].text());
    assert.equal(saved.answers[0].contract.question, 'Canonical first question');
    assert.equal(posts(env).length, 0, 'Selecting or exporting saved history must not invoke paid work');
  } else if (scenario === 'exact_citations') {
    const env = setup();
    const record = analysis('alpha');
    record.result.evidence_ids = ['alpha_source_2', 'made_up', 'alpha_source_1'];
    record.result.claims = [{title: 'Visible handling', description: '<script>not executable</script>', status: 'inferred', evidence_ids: ['alpha_source_3', 'other_project_source_1']}];
    record.result.step_insights = [{step_id: 'unknown_step', interpretation: 'Must not render'}, {step_id: 'step1', interpretation: 'Recorded interpretation', visual_check: 'Inspect visible contact', evidence_ids: ['alpha_source_4']}];
    await load(env, snapshot('alpha', {analysis: record}));
    const citations = [...env.nodes.values()].flatMap(walk).filter(n => n.className === 'citation');
    assert(citations.length >= 4);
    for (const citation of citations) {
      const source = project().interaction.evidence.find(e => e.id === citation.title);
      assert(source, 'Only exact current-source evidence IDs may become citations');
      citation.click();
      assert.equal(env.seeks.at(-1), source.timestamp_s, 'Display rounding must not alter the seek timestamp');
      assert.equal(env.tabs.at(-1), 'replay');
    }
    assert.doesNotMatch(text(env, 'intel-step-insights'), /Must not render/);
    assert.match(text(env, 'intel-claims'), /<script>not executable<\/script>/, 'Model text must remain text');
    assert.equal(posts(env).length, 0);
  } else if (scenario === 'mask_filters') {
    const env = setup(); await load(env);
    const filter = (track, audit, expected) => {
      env.get('intel-track-filter').value = track; env.get('intel-audit-filter').value = audit; env.get('intel-audit-filter').onchange();
      assert.equal(text(env, 'intel-evidence-count'), expected + ' / 5 source samples', track + ' / ' + audit);
    };
    filter('all', 'all', 5); filter('all', 'plausible', 2); filter('all', 'attention', 2); filter('all', 'missing', 1);
    filter('part', 'all', 3); filter('part', 'plausible', 1); filter('part', 'attention', 2);
    assert.match(text(env, 'intel-evidence-grid'), /not audited/, 'Unknown audit status must require attention');
    assert.match(text(env, 'intel-evidence-grid'), /rejected wrong object/);
    filter('part', 'missing', 2); filter('holder', 'attention', 0);
    assert.match(text(env, 'intel-evidence-grid'), /No samples match/);
    filter('part', 'plausible', 1);
    const card = env.get('intel-evidence-grid').children[0], image = walk(card).find(n => n.tagName === 'IMG');
    assert.equal(image.src, '/files/data/alpha/frame1.jpg');
    const mask = walk(card).find(n => n.tagName === 'A');
    assert.equal(mask.href, '/files/data/alpha/mask1.png'); assert.equal(mask.rel, 'noopener');
    assert.equal(env.pending.length, 1, 'Filtering uses saved evidence without new requests');
  } else if (scenario === 'resume_and_failure') {
    const env = setup(); await load(env, snapshot('alpha', {job: {status: 'running', mode: 'analysis'}}));
    assert.equal(env.get('intel-ask').disabled, true); env.submit('Do not duplicate running work');
    assert.equal(posts(env).length, 0);
    env.tick(); env.respond(env.take('GET'), snapshot('alpha', {job: {status: 'failed', error: 'Budget reservation declined'}})); await flush();
    assert.equal(env.get('intel-ask').disabled, false); assert.match(text(env, 'intel-status'), /Budget reservation declined/);
    assert.equal(env.timers.size, 0, 'A failure must not retry paid work automatically');
    env.submit('Try explicitly'); env.respond(env.take('POST'), {error: 'Daily guard blocked this request'}, false); await flush();
    assert.equal(env.get('intel-ask').disabled, false); assert.equal(env.get('intel-analyze').disabled, false);
    assert.match(text(env, 'intel-status'), /Daily guard blocked/); assert.equal(posts(env).length, 1); assert.equal(env.timers.size, 0);
  } else if (scenario === 'cached_question') {
    const env = setup(); await load(env);
    env.submit('Immutable question'); env.respond(env.take('POST'), {status: 'cached'}); await flush();
    env.respond(env.take('GET'), snapshot('alpha', {answers: [answer('Immutable question', 'alpha')]})); await flush();
    assert.match(text(env, 'intel-answer'), /Immutable question/);
    assert.match(text(env, 'intel-status'), /Identical saved result reused/);
    assert.equal(env.get('intel-ask').disabled, false); assert.equal(env.timers.size, 0);
    assert.equal(posts(env).length, 1);
  } else if (scenario === 'interrupted_and_cancelled') {
    for (const terminal of ['interrupted', 'cancelled']) {
      const env = setup(); await load(env, snapshot('alpha', {job: {status: 'running', mode: 'ask', question: 'Pending question'}}));
      env.tick(); env.respond(env.take('GET'), snapshot('alpha', {job: {status: terminal, mode: 'ask'}})); await flush();
      assert.match(text(env, 'intel-status'), new RegExp(terminal));
      assert.doesNotMatch(text(env, 'intel-status'), /Saved result ready/);
      assert.equal(env.get('intel-ask').disabled, false);
      assert.equal(env.get('intel-analyze').disabled, false);
      assert.equal(posts(env).length, 0, 'Terminal jobs must not start paid work');
      assert.equal(env.timers.size, 0, 'Terminal jobs must not resume polling or retry');
      const reopened = setup(); await load(reopened, snapshot('alpha', {job: {status: terminal, mode: 'ask'}}));
      assert.match(text(reopened, 'intel-status'), new RegExp(terminal));
      assert.equal(posts(reopened).length, 0); assert.equal(reopened.timers.size, 0);
    }
  } else if (scenario === 'missing_requested_answer') {
    const env = setup(); await load(env);
    env.submit('Requested question'); env.respond(env.take('POST'), {status: 'queued'}); await flush();
    env.tick(); env.respond(env.take('GET'), snapshot('alpha', {answers: [answer('Different old question', 'alpha')], job: {status: 'complete', mode: 'ask', question: 'Requested question'}})); await flush();
    assert.match(text(env, 'intel-status'), /matching saved answer is unavailable/);
    assert.doesNotMatch(text(env, 'intel-status'), /Saved result ready/);
    assert.match(text(env, 'intel-answer'), /Requested question/);
    assert.doesNotMatch(text(env, 'intel-answer'), /Different old question/, 'An unrelated history answer must not impersonate the requested answer');
    assert.equal(env.get('intel-ask').disabled, false); assert.equal(posts(env).length, 1); assert.equal(env.timers.size, 0);
    const reopened = setup(); await load(reopened, snapshot('alpha', {answers: [], job: {status: 'complete', mode: 'ask', question: 'Requested question'}}));
    assert.match(text(reopened, 'intel-status'), /matching saved answer is unavailable/);
    assert.equal(posts(reopened).length, 0);
  } else if (scenario === 'missing_cached_answer') {
    const env = setup(); await load(env);
    env.submit('Saved but unavailable'); env.respond(env.take('POST'), {status: 'cached'}); await flush();
    env.respond(env.take('GET'), snapshot('alpha', {answers: [answer('Other saved question', 'alpha')]})); await flush();
    assert.match(text(env, 'intel-status'), /matching saved answer is unavailable/);
    assert.doesNotMatch(text(env, 'intel-status'), /Identical saved result reused/);
    assert.doesNotMatch(text(env, 'intel-answer'), /Other saved question/);
    assert.equal(env.get('intel-ask').disabled, false); assert.equal(posts(env).length, 1); assert.equal(env.timers.size, 0);
  } else if (scenario === 'persistent_omission_disclosure') {
    for (const count of [0, 1, 3]) {
      const env = setup(), record = analysis('alpha'), question = 'Already completed question';
      if (count === 3) record.normalization = {removed_count: count};
      else record.recovery = {removed_count: count};
      const saved = snapshot('alpha', {analysis: record, answers: [answer(question, 'alpha')], job: {status: 'complete', mode: 'ask', question}});
      await load(env, saved);
      assert.match(text(env, 'intel-status'), /Saved Meta answer/);
      if (count) assert.match(text(env, 'intel-overview'), new RegExp(count + ' unsupported proposal item'));
      else assert.doesNotMatch(text(env, 'intel-overview'), /unsupported proposal item/);
      env.submit(question); env.respond(env.take('POST'), {status: 'cached'}); await flush();
      env.respond(env.take('GET'), saved); await flush();
      assert.match(text(env, 'intel-status'), /Identical saved result reused/);
      const notes = walk(env.get('intel-overview')).filter(node => node.className === 'analysis-normalization-note');
      assert.equal(notes.length, count ? 1 : 0, 'Omission disclosure must persist exactly once, independent of the global status');
      if (count) assert.match(notes[0].textContent, new RegExp(count + ' unsupported proposal item'));
      assert.equal(posts(env).length, 1); assert.equal(env.timers.size, 0);
    }
  } else throw new Error('Unknown scenario ' + scenario);
  console.log(scenario + ' passed; no network or models');
}
run().catch(error => {console.error(error); process.exitCode = 1;});
