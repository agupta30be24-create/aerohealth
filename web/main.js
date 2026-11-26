// AeroHealth 3D. Displays results computed in Python (src/export_frontend.py for NASA fleets, app/server.py for uploads).
// No model code runs here: every number shown comes from the payload.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';

window.addEventListener('error', (e) => { document.body.dataset.error = String(e.message); });
window.addEventListener('unhandledrejection', (e) => { document.body.dataset.error = String(e.reason); });

const $ = (id) => document.getElementById(id);
const sgn = (v, d = 1) => `${v < 0 ? '−' : '+'}${Math.abs(v).toFixed(d)}`;
const LEVEL = { HIGH: new THREE.Color(0xc8504a), MEDIUM: new THREE.Color(0xd39b3a), LOW: new THREE.Color(0x4f9a72) };
const NEG = new THREE.Color(0xc8504a), POS = new THREE.Color(0x4a78c2), STEEL = new THREE.Color(0x8a939e);
const COMP_KEYS = ['fan', 'lpc', 'hpc', 'combustor', 'hpt', 'lpt'];
const SHORT = { fan: 'Fan', lpc: 'LPC', hpc: 'HPC', combustor: 'Combustor', hpt: 'HPT', lpt: 'LPT' };
const R0 = 2.4, R1 = 14, RUL_MAX = 140;                       // radar: radius = predicted RUL
const rOf = (rul) => R0 + (Math.min(Math.max(rul, 0), RUL_MAX) / RUL_MAX) * (R1 - R0);
const HERO_Y = 4.2;

const state = { payload: null, source: 'FD001', sel: null, evalMode: false, activeComp: null,
  exploded: 0, explodeTarget: 0, view: 'fleet', scaleMax: 1, upload: null, cache: {} };

// ---------------------------------------------------------------- renderer, scene, lights
const stage = $('stage');
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
stage.appendChild(renderer.domElement);
const labelRenderer = new CSS2DRenderer();
Object.assign(labelRenderer.domElement.style, { position: 'absolute', top: '0', left: '0', pointerEvents: 'none' });
stage.appendChild(labelRenderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x12171e);
scene.fog = new THREE.Fog(0x12171e, 34, 70);
const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 300);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.maxPolarAngle = 1.42;
controls.minDistance = 4;
controls.maxDistance = 55;

scene.add(new THREE.HemisphereLight(0xdde6f0, 0x1a2028, 1.1));
const keyLight = new THREE.DirectionalLight(0xffffff, 1.8);
keyLight.position.set(-8, 16, 10);
keyLight.castShadow = true;
keyLight.shadow.mapSize.set(2048, 2048);
Object.assign(keyLight.shadow.camera, { left: -16, right: 16, top: 16, bottom: -16, near: 1, far: 50 });
scene.add(keyLight);
const fill = new THREE.DirectionalLight(0xc9d6e8, 0.5); fill.position.set(10, 6, -8); scene.add(fill);

function resize() {
  const w = window.innerWidth, h = window.innerHeight;
  renderer.setSize(w, h); labelRenderer.setSize(w, h);
  camera.aspect = w / h; camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize); resize();

function label(text, cls = 'label3d') {
  const el = document.createElement('div'); el.className = cls; el.textContent = text;
  return new CSS2DObject(el);
}
function circle(r, mat, y = 0.01) {
  const pts = []; for (let i = 0; i <= 160; i++) { const a = (i / 160) * Math.PI * 2; pts.push(new THREE.Vector3(Math.cos(a) * r, y, Math.sin(a) * r)); }
  const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), mat);
  if (mat.isLineDashedMaterial) line.computeLineDistances();
  return line;
}

// ---------------------------------------------------------------- radar (polar chart of remaining life)
const radar = new THREE.Group(); scene.add(radar);
{
  const floor = new THREE.Mesh(new THREE.CircleGeometry(R1 + 2, 96), new THREE.MeshStandardMaterial({ color: 0x161c24, roughness: 1, metalness: 0 }));
  floor.rotation.x = -Math.PI / 2; floor.receiveShadow = true; radar.add(floor);
  const gridMat = new THREE.LineBasicMaterial({ color: 0x2a3442 });
  for (let rul = 20; rul <= 140; rul += 20) {
    radar.add(circle(rOf(rul), gridMat));
    if (rul % 40 === 0) { const l = label(`${rul} cycles`, 'label3d ring'); l.position.set(0, 0.05, rOf(rul) + 0.35); radar.add(l); }
  }
  for (let i = 0; i < 8; i++) {
    const a = (i / 8) * Math.PI * 2;
    radar.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(Math.cos(a) * R0, 0.01, Math.sin(a) * R0), new THREE.Vector3(Math.cos(a) * R1, 0.01, Math.sin(a) * R1)]), gridMat));
  }
  radar.add(circle(R0 - 0.3, new THREE.LineBasicMaterial({ color: 0x3b4757 })));
}
const thresholdGroup = new THREE.Group(); radar.add(thresholdGroup);
function buildThresholds(p) {
  thresholdGroup.clear();
  for (const [rul, col, txt] of [[p.cutoff_high, LEVEL.HIGH, `HIGH ≤ ${p.cutoff_high} cycles`], [p.cutoff_medium, LEVEL.MEDIUM, `MEDIUM ≤ ${p.cutoff_medium}`]]) {
    const r = rOf(rul);
    thresholdGroup.add(circle(r, new THREE.LineDashedMaterial({ color: col, dashSize: 0.35, gapSize: 0.2 }), 0.02));
    const l = label(txt, 'label3d ring'); l.element.style.color = `#${col.getHexString()}`; l.position.set(r * 0.71, 0.05, -r * 0.71); thresholdGroup.add(l);
  }
}

// ---------------------------------------------------------------- fleet engines (instanced)
let pods = null;
const podGeo = new THREE.CylinderGeometry(0.22, 0.17, 0.55, 20).rotateZ(Math.PI / 2);
const selRing = circle(0.45, new THREE.LineBasicMaterial({ color: 0xdfe6ee }), 0.03);
scene.add(selRing);
let link = null;

function buildFleet(p) {
  if (pods) { scene.remove(pods); pods.material.dispose(); }
  const E = p.engines, golden = Math.PI * (3 - Math.sqrt(5));
  pods = new THREE.InstancedMesh(podGeo, new THREE.MeshStandardMaterial({ roughness: 0.55, metalness: 0.2 }), E.length);
  pods.castShadow = true;
  const m = new THREE.Matrix4(), q = new THREE.Quaternion(), s = new THREE.Vector3(1, 1, 1), up = new THREE.Vector3(0, 1, 0);
  [...E].sort((a, b) => a.id - b.id).forEach((e, k) => { e._angle = k * golden; });   // stable position per engine id
  E.forEach((e, i) => {
    const r = rOf(e.pred_rul), a = e._angle;
    e._pos = new THREE.Vector3(Math.cos(a) * r, 0.22, Math.sin(a) * r);
    pods.setMatrixAt(i, m.compose(e._pos, q.setFromAxisAngle(up, -a), s));
    pods.setColorAt(i, LEVEL[e.level]);
  });
  scene.add(pods);
  buildThresholds(p);
}

// ---------------------------------------------------------------- engine model
const hero = new THREE.Group(); hero.position.y = HERO_Y; hero.scale.setScalar(1.3); scene.add(hero);
const comps = {};
const EXPLODE = { fan: -1.5, lpc: -0.85, hpc: -0.2, combustor: 0.45, hpt: 1.0, lpt: 1.65 };
const BASE_X = { fan: -2.06, lpc: -1.6, hpc: -0.8, combustor: 0.02, hpt: 0.43, lpt: 1.25 };
const frustum = (a, b, len, seg = 48, open = false) => new THREE.CylinderGeometry(a, b, len, seg, 1, open).rotateZ(Math.PI / 2);
const ringG = (r) => new THREE.CylinderGeometry(r, r, 0.04, 48).rotateZ(Math.PI / 2);

function addPart(key, geo, x) {
  const c = comps[key], mesh = new THREE.Mesh(geo, c.mat);
  mesh.position.x = x; mesh.userData.comp = key; mesh.castShadow = true; c.group.add(mesh); return mesh;
}
COMP_KEYS.forEach((k, i) => {
  const group = new THREE.Group(); group.position.x = BASE_X[k]; hero.add(group);
  const l = label(SHORT[k]); l.position.set(0, i % 2 ? -1.0 : 1.05, 0); group.add(l);
  comps[k] = { group, mat: new THREE.MeshStandardMaterial({ color: STEEL.clone(), metalness: 0.55, roughness: 0.42 }), label: l.element };
});
const shaft = new THREE.Mesh(frustum(0.07, 0.07, 4.4, 16), new THREE.MeshStandardMaterial({ color: 0x5b636e, metalness: 0.6, roughness: 0.4 }));
hero.add(shaft);
const nacelle = new THREE.Mesh(frustum(1.22, 1.1, 3.0, 72, true), new THREE.MeshStandardMaterial({
  color: 0xb8c2cf, metalness: 0.3, roughness: 0.5, transparent: true, opacity: 0.1, side: THREE.DoubleSide, depthWrite: false }));
nacelle.position.x = -0.95; hero.add(nacelle);
const lip = new THREE.Mesh(new THREE.TorusGeometry(1.2, 0.03, 12, 96).rotateY(Math.PI / 2), new THREE.MeshStandardMaterial({ color: 0x9aa4b0, metalness: 0.6, roughness: 0.35 }));
lip.position.x = -2.45; hero.add(lip);
// fan
addPart('fan', new THREE.ConeGeometry(0.3, 0.55, 32).rotateZ(Math.PI / 2), -0.39);
addPart('fan', frustum(0.3, 0.34, 0.25), 0);
const fanBlades = new THREE.Group(); comps.fan.group.add(fanBlades);
for (let i = 0; i < 22; i++) {
  const a = (i / 22) * Math.PI * 2, b = new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.84, 0.2), comps.fan.mat);
  b.position.set(0, Math.cos(a) * 0.76, Math.sin(a) * 0.76); b.rotation.x = -a; b.rotation.y = 0.5; b.userData.comp = 'fan'; b.castShadow = true; fanBlades.add(b);
}
addPart('lpc', frustum(0.6, 0.56, 0.5), 0); [-0.18, 0, 0.18].forEach((x) => addPart('lpc', ringG(0.64), x));
addPart('hpc', frustum(0.53, 0.38, 1.1), 0); for (let s = 0; s < 6; s++) addPart('hpc', ringG(0.56 - s * 0.03), -0.47 + s * 0.19);
addPart('combustor', new THREE.TorusGeometry(0.46, 0.14, 20, 64).rotateY(Math.PI / 2), 0); addPart('combustor', frustum(0.38, 0.4, 0.42), 0);
addPart('hpt', frustum(0.42, 0.48, 0.34), 0); [-0.1, 0.07].forEach((x) => addPart('hpt', ringG(0.5), x));
addPart('lpt', frustum(0.5, 0.66, 0.8), -0.23); [-0.5, -0.3, -0.1, 0.07].forEach((x, i) => addPart('lpt', ringG(0.55 + i * 0.035), x));
addPart('lpt', frustum(0.66, 0.5, 0.45, 48, true), 0.39); addPart('lpt', new THREE.ConeGeometry(0.3, 0.6, 32).rotateZ(-Math.PI / 2), 0.53);
// stand: a simple pylon so the engine does not float without reason
const pylon = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.12, HERO_Y - 1.4, 16), new THREE.MeshStandardMaterial({ color: 0x3b4450, metalness: 0.5, roughness: 0.5 }));
pylon.position.y = (HERO_Y - 1.4) / 2; pylon.castShadow = true; scene.add(pylon);

function colourHero() {
  const e = state.sel; if (!e) return;
  COMP_KEYS.forEach((k) => {
    const v = e.component_shap[k] ?? 0, t = Math.min(Math.abs(v) / state.scaleMax, 1), c = comps[k];
    c.mat.color.copy(STEEL).lerp(v < 0 ? NEG : POS, 0.9 * t);           // tint proportional to the part's contribution
    c.mat.emissive.setHex(state.activeComp === k ? 0x2a2f36 : 0x000000);
    c.label.innerHTML = `${SHORT[k]} <span class="${v < 0 ? 'neg' : 'pos'}">${sgn(v)}</span>`;
    c.label.classList.toggle('active', state.activeComp === k);
  });
}

// ---------------------------------------------------------------- camera views
const flight = { t: 1, dur: 0.9, fromP: new THREE.Vector3(), toP: new THREE.Vector3(), fromT: new THREE.Vector3(), toT: new THREE.Vector3() };
const ease = (x) => (x < 0.5 ? 2 * x * x : 1 - Math.pow(-2 * x + 2, 2) / 2);
const VIEWS = {
  fleet: () => [new THREE.Vector3(0, 18, 24), new THREE.Vector3(0, 1.6, 0)],
  inspect: () => { const f = 1 + 0.45 * state.explodeTarget;       // back off when the engine is exploded (it is wider)
    return [new THREE.Vector3(6.5 * f, HERO_Y + 2.6 * f, 9.5 * f), new THREE.Vector3(0, HERO_Y - 0.3, 0)]; },
};
function setView(v, instant = false) {
  state.view = v;
  const [pos, target] = VIEWS[v]();
  if (instant) { camera.position.copy(pos); controls.target.copy(target); flight.t = 1; }
  else { Object.assign(flight, { t: 0 }); flight.fromP.copy(camera.position); flight.toP.copy(pos); flight.fromT.copy(controls.target); flight.toT.copy(target); }
  $('btnFleet').classList.toggle('active', v === 'fleet');
  $('btnInspect').classList.toggle('active', v === 'inspect');
}

// ---------------------------------------------------------------- selection + panels
function select(e) {
  state.sel = e;
  selRing.position.set(e._pos.x, 0, e._pos.z);
  if (link) { scene.remove(link); link.geometry.dispose(); }
  const curve = new THREE.QuadraticBezierCurve3(e._pos.clone(), new THREE.Vector3(e._pos.x * 0.5, HERO_Y + 0.8, e._pos.z * 0.5), new THREE.Vector3(0, HERO_Y - 0.9, 0));
  link = new THREE.Line(new THREE.BufferGeometry().setFromPoints(curve.getPoints(48)),
    new THREE.LineDashedMaterial({ color: 0x8b96a6, dashSize: 0.15, gapSize: 0.12 }));
  link.computeLineDistances();
  scene.add(link);
  colourHero(); renderEnginePanel(); renderFleetPanel();
}

function notesOf(e) {
  const n = [];
  if (e.unknown_condition_share > 0) n.push(['red', `Operating settings not seen by the ${state.payload.dataset} model in training (${Math.round(e.unknown_condition_share * 100)}% of rows). Prediction not reliable.`]);
  if (e.out_of_range_sensors?.length) n.push(['amber', `Sensor readings beyond the training range (${e.out_of_range_sensors.join(', ')}). Treat with caution.`]);
  if (e.padded) n.push(['amber', 'Fewer than 30 recorded cycles: window padded by repeating the first cycle, so the prediction is less informed.']);
  return n;
}

function renderFleetPanel() {
  const p = state.payload, up = p.source === 'upload', ev = p.evaluation;
  const title = up ? `Uploaded engines <span class="muted">· ${state.upload?.name ?? 'upload'}</span>` : `NASA ${p.dataset} test fleet`;
  const flagged = p.engines.filter((e) => notesOf(e).length).length;
  let h = `<h2>${title}</h2><div class="muted small">${up ? `${p.engines.length} engine(s), ${p.n_rows} rows · ` : ''}
    model trained on ${p.dataset} (${p.n_regimes} condition${p.n_regimes > 1 ? 's' : ''}, ${p.fault_mode})</div>
    <div class="counts">${['HIGH', 'MEDIUM', 'LOW'].map((l) => `<div class="count ${l}"><div class="n">${p.counts[l]}</div><div class="l">${{ HIGH: 'HIGH · act now', MEDIUM: 'MEDIUM · plan', LOW: 'LOW' }[l]}</div></div>`).join('')}</div>`;
  if (up && p.suggested_model) h += `<div class="alert info">Suggested model: <b>${p.suggested_model}</b>. ${p.suggestion_reason}</div>`;
  if (up && p.engines.some((e) => e.unknown_condition_share > 0)) h += `<div class="alert red">Some engines run at flight conditions this model never saw. Switch to the suggested model.</div>`;
  (p.notes || []).forEach((n) => { h += `<div class="alert info">${n}</div>`; });
  h += `<h3>Reading the chart</h3><div class="small muted">Distance from the centre is the <b style="color:var(--text)">predicted remaining life</b>.
    Dashed rings are the calibrated cut-offs: HIGH if P(fail within ${p.H}) ≥ ${p.p_cut} (predicted RUL ≤ ${p.cutoff_high}),
    MEDIUM if P(fail within ${2 * p.H}) ≥ ${p.p_cut} (≤ ${p.cutoff_medium}). Derived from validation engines, not chosen.</div>`;
  if (p.test_rmse != null) h += `<div class="small muted" style="margin-top:6px">Test RMSE ${p.test_rmse} cycles.</div>`;
  if (state.evalMode && ev) h += `<div class="alert info">Evaluation: HIGH caught <b>${ev.high_caught} of ${ev.fail_within_H}</b> engines failing within ${p.H};
    <b>${ev.high_caught} of ${ev.high_total}</b> flags correct; failing within ${p.H} but rated LOW: <b>${ev.fail_within_H_rated_low}</b>.</div>`;
  h += `<h3>Highest risk${flagged ? ` · ${flagged} with data warnings` : ''}</h3><div class="list">`;
  p.engines.slice(0, 12).forEach((e) => {
    h += `<div class="row ${state.sel === e ? 'sel' : ''}" data-id="${e.id}"><b>#${e.id}</b><span class="badge ${e.level}">${e.level}</span>
      <span class="p">${e.pred_rul.toFixed(1)} cycles · ${(e.p_h * 100).toFixed(0)}%</span><span class="warnmark">${notesOf(e).length ? '⚠' : ''}</span></div>`;
  });
  h += `</div>`;
  if (up) h += `<div style="margin-top:12px;display:flex;gap:8px"><button class="btn" id="dlCsv">Download predictions (CSV)</button><button class="btn" id="newUpload">New file</button></div>`;
  $('fleetPanel').innerHTML = h;
  $('fleetPanel').querySelectorAll('.row').forEach((r) => r.addEventListener('click', () => select(p.engines.find((e) => String(e.id) === r.dataset.id))));
  $('dlCsv')?.addEventListener('click', downloadCsv);
  $('newUpload')?.addEventListener('click', () => $('uploadPanel').classList.remove('hidden'));
}

function renderEnginePanel() {
  const p = state.payload, e = state.sel, H = p.H;
  const compSum = COMP_KEYS.reduce((s, k) => s + (e.component_shap[k] ?? 0), 0);
  const truth = e.true_rul == null ? ['n/a', 'true RUL (unknown for uploads)'] : state.evalMode ? [e.true_rul, 'true RUL (benchmark only)'] : ['hidden', 'true RUL (turn on evaluation mode)'];
  let h = `<h2>Engine #${e.id} <span class="badge ${e.level}">${e.level}</span></h2>
    <div class="kpis"><div class="kpi"><div class="v">${e.pred_rul.toFixed(1)}</div><div class="l">predicted remaining life (cycles)</div></div>
    <div class="kpi"><div class="v">${(e.p_h * 100).toFixed(0)}%</div><div class="l">chance of failing within ${H} cycles</div></div>
    <div class="kpi"><div class="v">${(e.p_2h * 100).toFixed(0)}%</div><div class="l">within ${2 * H} cycles</div></div>
    <div class="kpi"><div class="v">${truth[0]}</div><div class="l">${truth[1]}</div></div></div>`;
  notesOf(e).forEach(([cls, t]) => { h += `<div class="alert ${cls}">${t}</div>`; });
  h += `<h3>Why this prediction</h3><div class="small">The ${p.dataset} model's average prediction is <b>${p.base_value.toFixed(1)}</b> cycles. Biggest reasons this engine differs:</div>
    <ul class="drivers">${e.drivers.map((r) => `<li>${r.text}: <span class="${r.shap < 0 ? 'neg' : 'pos'}">${r.shap < 0 ? 'shortens' : 'extends'} life by ${Math.abs(r.shap).toFixed(1)}</span> cycles</li>`).join('')}</ul>
    <h3>Contribution by engine part · click to expand</h3>`;
  COMP_KEYS.forEach((k) => {
    const v = e.component_shap[k] ?? 0, c = p.components[k], w = Math.min(Math.abs(v) / state.scaleMax, 1) * 50;
    h += `<div class="comp ${state.activeComp === k ? 'active' : ''}" data-comp="${k}"><span>${c.label}</span>
      <div class="track"><div class="fill" style="left:${v < 0 ? 50 - w : 50}%;width:${w}%;background:${v < 0 ? 'var(--neg)' : 'var(--pos)'}"></div></div>
      <span class="num ${v < 0 ? 'neg' : 'pos'}">${sgn(v)}</span></div>`;
    if (state.activeComp === k) h += `<div class="sensors">${c.sensors.length ? c.sensors.map((s) => `${s} ${c.sensor_names[s]}: <span class="${e.sensor_shap[s] < 0 ? 'neg' : 'pos'}">${sgn(e.sensor_shap[s], 2)}</span>`).join('<br>') : 'no sensors kept for this part in this dataset'}</div>`;
  });
  h += `<div class="check">Check: model average ${p.base_value.toFixed(1)} + parts ${sgn(compSum)} = <b>${(p.base_value + compSum).toFixed(1)}</b>
      vs prediction <b>${e.pred_rul.toFixed(1)}</b> (SHAP values add up exactly; small gaps are rounding).</div>
    <div class="check">Part tint = summed SHAP of the sensors located there (Saxena et al. 2008, Table 2; core speed and bleed are drawn on the HPC).
      The HPC has the most sensors, so compare individual sensors before concluding.</div>`;
  $('enginePanel').innerHTML = h;
  $('enginePanel').querySelectorAll('.comp').forEach((el) => el.addEventListener('click', () => setActiveComp(el.dataset.comp)));
}

function setActiveComp(k) {
  state.activeComp = state.activeComp === k ? null : k;
  colourHero(); renderEnginePanel();
  if (state.activeComp && state.view !== 'inspect') setView('inspect');
}

function downloadCsv() {
  const p = state.payload, rows = [['engine', 'risk', `p_fail_within_${p.H}`, `p_fail_within_${2 * p.H}`, 'predicted_RUL', 'driver1', 'driver2', 'driver3', 'data_check', 'model']];
  p.engines.forEach((e) => rows.push([e.id, e.level, e.p_h, e.p_2h, e.pred_rul, ...e.drivers.map((d) => d.feature), notesOf(e).map((n) => n[1].split(':')[0].split('(')[0].trim()).join('; '), p.dataset]));
  const blob = new Blob([rows.map((r) => r.map((c) => `"${String(c).replaceAll('"', '""')}"`).join(',')).join('\n')], { type: 'text/csv' });
  const a = Object.assign(document.createElement('a'), { href: URL.createObjectURL(blob), download: `aerohealth_predictions_${p.dataset}.csv` });
  a.click(); URL.revokeObjectURL(a.href);
}

// ---------------------------------------------------------------- data
function show(payload) {
  state.payload = payload; state.activeComp = null;
  const vals = payload.engines.flatMap((e) => COMP_KEYS.map((k) => Math.abs(e.component_shap[k] ?? 0))).sort((a, b) => a - b);
  state.scaleMax = Math.max(vals[Math.floor(vals.length * 0.95)] || 1, 1e-6);    // 95th pct so one outlier does not wash out colours
  buildFleet(payload);
  const hasTruth = payload.engines[0]?.true_rul != null;
  $('evalWrap').classList.toggle('disabled', !hasTruth); $('evalMode').disabled = !hasTruth;
  if (!hasTruth) { $('evalMode').checked = false; state.evalMode = false; }
  select(payload.engines[0]);
}

async function loadBenchmark(fd) {
  if (!state.cache[fd]) {
    const res = await fetch(`data/${fd}.json`);
    if (!res.ok) throw new Error(`Could not load data/${fd}.json`);
    state.cache[fd] = await res.json();
  }
  state.source = fd; markSource(fd); show(state.cache[fd]);
}

function markSource(src) { document.querySelectorAll('#sources button').forEach((b) => b.classList.toggle('active', b.dataset.src === src)); }

function status(html) { $('uploadStatus').innerHTML = html; }
async function analyse(file) {
  state.upload = { name: file.name, bytes: await file.arrayBuffer() };
  return runUpload();
}
async function runUpload() {
  const model = $('modelSelect').value;
  status(`<div class="alert info">Analysing <b>${state.upload.name}</b> with the ${model === 'auto' ? 'suggested' : model} model…</div>`);
  let res;
  try {
    res = await fetch(`api/predict?model=${model}`, { method: 'POST', body: state.upload.bytes });
  } catch {
    status(`<div class="alert red">The analysis server is not running. Start it with <code>python app/server.py</code> and open <code>http://localhost:8765</code>.</div>`);
    return;
  }
  const data = await res.json().catch(() => ({ error: `Server error (${res.status}). Start it with python app/server.py.` }));
  if (!res.ok || data.error) { status(`<div class="alert red">This file cannot be used: ${data.error}</div>`); return; }
  status(`<div class="alert info">${data.engines.length} engine(s) analysed with the <b>${data.dataset}</b> model.</div>`);
  state.source = 'upload'; markSource('upload'); show(data);
  $('uploadPanel').classList.add('hidden');
  setView('fleet');
}

// ---------------------------------------------------------------- interaction
const ray = new THREE.Raycaster(), mouse = new THREE.Vector2();
function pick(ev) {
  mouse.set((ev.clientX / window.innerWidth) * 2 - 1, -(ev.clientY / window.innerHeight) * 2 + 1);
  ray.setFromCamera(mouse, camera);
  const hits = ray.intersectObjects([pods, hero].filter(Boolean), true);
  for (const h of hits) {
    if (h.object === pods) return { engine: state.payload.engines[h.instanceId] };
    if (h.object.userData.comp) return { comp: h.object.userData.comp };
  }
  return {};
}
renderer.domElement.addEventListener('pointermove', (ev) => {
  const { engine, comp } = pick(ev), tip = $('tooltip');
  renderer.domElement.style.cursor = engine || comp ? 'pointer' : '';
  if (!engine && !comp) { tip.classList.add('hidden'); return; }
  tip.innerHTML = engine
    ? `<b>Engine #${engine.id}</b> <span class="badge ${engine.level}">${engine.level}</span><br>${engine.pred_rul.toFixed(1)} cycles left · ${(engine.p_h * 100).toFixed(0)}% within ${state.payload.H}`
      + (state.evalMode && engine.true_rul != null ? `<br>true RUL ${engine.true_rul}` : '') + (notesOf(engine).length ? '<br>⚠ data warning' : '')
    : `<b>${state.payload.components[comp].label}</b><br>${sgn(state.sel.component_shap[comp])} cycles for engine #${state.sel.id}`;
  Object.assign(tip.style, { left: `${ev.clientX + 14}px`, top: `${ev.clientY + 14}px` });
  tip.classList.remove('hidden');
});
let down = null;
renderer.domElement.addEventListener('pointerdown', (ev) => { down = [ev.clientX, ev.clientY]; });
renderer.domElement.addEventListener('pointerup', (ev) => {
  if (!down || Math.hypot(ev.clientX - down[0], ev.clientY - down[1]) > 5) return;
  const { engine, comp } = pick(ev);
  if (engine) select(engine); else if (comp) setActiveComp(comp);
});

document.querySelectorAll('#sources button').forEach((b) => b.addEventListener('click', () => {
  if (b.dataset.src === 'upload') { $('uploadPanel').classList.remove('hidden'); return; }
  loadBenchmark(b.dataset.src); setView('fleet');
}));
$('evalMode').addEventListener('change', (e) => { state.evalMode = e.target.checked; renderFleetPanel(); renderEnginePanel(); });
$('btnFleet').addEventListener('click', () => setView('fleet'));
$('btnInspect').addEventListener('click', () => setView('inspect'));
$('btnExplode').addEventListener('click', () => {
  state.explodeTarget = state.explodeTarget ? 0 : 1;
  $('btnExplode').classList.toggle('active', !!state.explodeTarget);
  if (state.explodeTarget || state.view === 'inspect') setView('inspect');
});
$('uploadClose').addEventListener('click', () => $('uploadPanel').classList.add('hidden'));
$('fileInput').addEventListener('change', (e) => { if (e.target.files[0]) analyse(e.target.files[0]); });
$('modelSelect').addEventListener('change', () => { if (state.upload) runUpload(); });
document.querySelectorAll('[data-example]').forEach((b) => b.addEventListener('click', async () => {
  const res = await fetch(`examples/${b.dataset.example}`);
  if (!res.ok) { status(`<div class="alert red">Examples need the analysis server: <code>python app/server.py</code>.</div>`); return; }
  analyse(new File([await res.blob()], b.dataset.example));
}));
let dragDepth = 0;
window.addEventListener('dragenter', (e) => { e.preventDefault(); dragDepth++; $('dropOverlay').classList.remove('hidden'); });
window.addEventListener('dragleave', () => { if (--dragDepth <= 0) { dragDepth = 0; $('dropOverlay').classList.add('hidden'); } });
window.addEventListener('dragover', (e) => e.preventDefault());
window.addEventListener('drop', (e) => {
  e.preventDefault(); dragDepth = 0; $('dropOverlay').classList.add('hidden');
  const f = e.dataTransfer.files[0];
  if (f) { $('uploadPanel').classList.remove('hidden'); analyse(f); }
});

// ---------------------------------------------------------------- loop
const clock = new THREE.Clock();
function animate() {
  const dt = Math.min(clock.getDelta(), 0.05);
  if (flight.t < 1) {
    flight.t = Math.min(1, flight.t + dt / flight.dur);
    const k = ease(flight.t);
    camera.position.lerpVectors(flight.fromP, flight.toP, k); controls.target.lerpVectors(flight.fromT, flight.toT, k);
  }
  controls.update();
  fanBlades.rotation.x += dt * 1.5;                                   // slow idle rotation of the fan only
  state.exploded += (state.explodeTarget - state.exploded) * Math.min(1, dt * 5);
  COMP_KEYS.forEach((k) => { comps[k].group.position.x = BASE_X[k] + EXPLODE[k] * state.exploded; });
  nacelle.visible = lip.visible = state.exploded < 0.3;
  renderer.render(scene, camera);
  labelRenderer.render(scene, camera);
  requestAnimationFrame(animate);
}

// ---------------------------------------------------------------- start (URL options for prepared views)
// ?fd=FD004 &eval=1 &engine=10 &comp=hpc &view=inspect &exploded=1 &example=sample_engines_6conditions.csv
const params = new URLSearchParams(location.search);
(async () => {
  const fd = ['FD001', 'FD002', 'FD003', 'FD004'].includes(params.get('fd')) ? params.get('fd') : 'FD001';
  if (params.get('eval') === '1') { state.evalMode = true; $('evalMode').checked = true; }
  await loadBenchmark(fd);
  if (params.get('example')) {
    const res = await fetch(`examples/${params.get('example')}`);
    if (res.ok) await analyse(new File([await res.blob()], params.get('example')));
  }
  const eng = state.payload.engines.find((e) => String(e.id) === params.get('engine'));
  if (eng) select(eng);
  if (COMP_KEYS.includes(params.get('comp'))) setActiveComp(params.get('comp'));
  if (params.get('exploded') === '1') { state.explodeTarget = 1; state.exploded = 1; $('btnExplode').classList.add('active'); }
  const inspect = params.get('view') === 'inspect' || params.get('comp') || params.get('exploded') === '1';
  setView(inspect ? 'inspect' : 'fleet', true);
  $('loader').classList.add('done');
  document.body.dataset.ready = '1';
  animate();
})();
