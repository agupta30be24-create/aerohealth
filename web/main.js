// AeroHealth 3D. Displays results computed in Python (src/export_frontend.py for NASA fleets, app/server.py for uploads).
// No model code runs here: every number shown comes from the payload.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

window.addEventListener('error', (e) => { document.body.dataset.error = String(e.message); });
window.addEventListener('unhandledrejection', (e) => { document.body.dataset.error = String(e.reason); });

const $ = (id) => document.getElementById(id);
const sgn = (v, d = 1) => `${v < 0 ? '−' : '+'}${Math.abs(v).toFixed(d)}`;
const LEVEL = { HIGH: new THREE.Color(0xff4d5e), MEDIUM: new THREE.Color(0xffb020), LOW: new THREE.Color(0x2fd38c) };
const NEG = new THREE.Color(0xff3045), POS = new THREE.Color(0x3d7dff), NEUTRAL = new THREE.Color(0x3a4256);
const COMP_KEYS = ['fan', 'lpc', 'hpc', 'combustor', 'hpt', 'lpt'];
const SHORT = { fan: 'Fan', lpc: 'LPC', hpc: 'HPC', combustor: 'Combustor', hpt: 'HPT', lpt: 'LPT' };
const R0 = 2.4, R1 = 14, RUL_MAX = 140;                       // radar: radius = predicted RUL
const rOf = (rul) => R0 + (Math.min(Math.max(rul, 0), RUL_MAX) / RUL_MAX) * (R1 - R0);
const HERO_Y = 4.4;

const state = { payload: null, source: 'FD001', sel: null, evalMode: false, activeComp: null,
  exploded: 0, explodeTarget: 0, view: 'fleet', scaleMax: 1, upload: null, cache: {} };

// ---------------------------------------------------------------- renderer, scene, bloom
const stage = $('stage');
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
stage.appendChild(renderer.domElement);
const labelRenderer = new CSS2DRenderer();
Object.assign(labelRenderer.domElement.style, { position: 'absolute', top: '0', left: '0', pointerEvents: 'none' });
stage.appendChild(labelRenderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x04060b);
scene.fog = new THREE.FogExp2(0x04060b, 0.022);
const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 400);
camera.position.set(0, 38, 52);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.maxPolarAngle = 1.42;
controls.minDistance = 4;
controls.maxDistance = 60;
controls.target.set(0, 2, 0);

scene.add(new THREE.HemisphereLight(0x9fc4ff, 0x0a0f1a, 0.9));
const key = new THREE.DirectionalLight(0xffffff, 2.2); key.position.set(8, 14, 10); scene.add(key);
const rim = new THREE.PointLight(0x5cc8ff, 40, 30); rim.position.set(-6, 7, -6); scene.add(rim);

const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.5, 0.4, 0.55);
composer.addPass(bloom);
composer.addPass(new OutputPass());

function resize() {
  const w = window.innerWidth, h = window.innerHeight;
  renderer.setSize(w, h); composer.setSize(w, h); labelRenderer.setSize(w, h);
  camera.aspect = w / h; camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize); resize();

function label(text, cls = 'label3d') {
  const el = document.createElement('div'); el.className = cls; el.textContent = text;
  return new CSS2DObject(el);
}

// ---------------------------------------------------------------- background dust
{
  const n = 1800, pos = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const r = 40 + Math.random() * 90, t = Math.random() * Math.PI * 2, y = (Math.random() - 0.3) * 60;
    pos.set([Math.cos(t) * r, y, Math.sin(t) * r], i * 3);
  }
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  scene.add(new THREE.Points(g, new THREE.PointsMaterial({ color: 0x6f8fc8, size: 0.18, transparent: true, opacity: 0.55, fog: false })));
}

// ---------------------------------------------------------------- radar
const radar = new THREE.Group(); scene.add(radar);
const glowMat = (hex, mult = 1, opacity = 1) => new THREE.MeshBasicMaterial({
  color: new THREE.Color(hex).multiplyScalar(mult), transparent: opacity < 1, opacity, toneMapped: false,
  side: THREE.DoubleSide, depthWrite: opacity >= 1 });
{
  const disc = new THREE.Mesh(new THREE.CircleGeometry(R1 + 1.5, 96), new THREE.MeshBasicMaterial({ color: 0x081124, transparent: true, opacity: 0.85 }));
  disc.rotation.x = -Math.PI / 2; radar.add(disc);
  for (let rul = 20; rul <= 140; rul += 20) {                         // faint grid rings every 20 cycles
    const r = rOf(rul), ring = new THREE.Mesh(new THREE.RingGeometry(r - 0.015, r + 0.015, 128), glowMat(0x5cc8ff, 0.35, 0.5));
    ring.rotation.x = -Math.PI / 2; ring.position.y = 0.01; radar.add(ring);
    if (rul % 40 === 0) { const l = label(`${rul} cycles`, 'label3d ring'); l.element.style.color = '#6f8fc8'; l.position.set(0, 0.05, r); radar.add(l); }
  }
  for (let i = 0; i < 12; i++) {                                     // spokes
    const a = (i / 12) * Math.PI * 2, g = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(Math.cos(a) * R0, 0.012, Math.sin(a) * R0), new THREE.Vector3(Math.cos(a) * R1, 0.012, Math.sin(a) * R1)]);
    radar.add(new THREE.Line(g, new THREE.LineBasicMaterial({ color: 0x1d3155 })));
  }
  const pedestal = new THREE.Mesh(new THREE.RingGeometry(R0 - 0.35, R0 - 0.2, 96), glowMat(0x5cc8ff, 1.4));
  pedestal.rotation.x = -Math.PI / 2; pedestal.position.y = 0.02; radar.add(pedestal);
}
const sweep = new THREE.Mesh(new THREE.CircleGeometry(R1 + 0.6, 64, 0, 0.55),
  new THREE.MeshBasicMaterial({ color: 0x5cc8ff, transparent: true, opacity: 0.07, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide }));
sweep.rotation.x = -Math.PI / 2; sweep.position.y = 0.03; radar.add(sweep);

// threshold rings (derived cut-offs), rebuilt per payload
const thresholdGroup = new THREE.Group(); radar.add(thresholdGroup);
function buildThresholds(p) {
  thresholdGroup.clear();
  for (const [rul, hex, txt] of [[p.cutoff_high, 0xff4d5e, `HIGH ≤ ${p.cutoff_high} cycles`], [p.cutoff_medium, 0xffb020, `MEDIUM ≤ ${p.cutoff_medium}`]]) {
    const r = rOf(rul), ring = new THREE.Mesh(new THREE.RingGeometry(r - 0.04, r + 0.04, 160), glowMat(hex, 1.6));
    ring.rotation.x = -Math.PI / 2; ring.position.y = 0.02; thresholdGroup.add(ring);
    const l = label(txt, 'label3d ring'); l.element.style.color = `#${hex.toString(16)}`; l.position.set(r * 0.71, 0.08, -r * 0.71); thresholdGroup.add(l);
  }
}

// ---------------------------------------------------------------- fleet pods (instanced)
let pods = null;
const podGeo = new THREE.CylinderGeometry(0.26, 0.2, 0.62, 18).rotateZ(Math.PI / 2);
const beam = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.28, 2.6, 24, 1, true),
  new THREE.MeshBasicMaterial({ color: 0x9fdcff, transparent: true, opacity: 0.16, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide }));
scene.add(beam);
const selRing = new THREE.Mesh(new THREE.RingGeometry(0.42, 0.5, 40), glowMat(0xffffff, 1.6));
selRing.rotation.x = -Math.PI / 2; scene.add(selRing);
let link = null;

function buildFleet(p) {
  if (pods) { scene.remove(pods); pods.material.dispose(); }
  const E = p.engines, n = E.length, golden = Math.PI * (3 - Math.sqrt(5));
  pods = new THREE.InstancedMesh(podGeo, new THREE.MeshBasicMaterial({ toneMapped: false }), n);
  const m = new THREE.Matrix4(), q = new THREE.Quaternion(), s = new THREE.Vector3(1, 1, 1);
  [...E].sort((a, b) => a.id - b.id).forEach((e, k) => { e._angle = k * golden; });   // stable angle per engine id
  E.forEach((e, i) => {
    const r = rOf(e.pred_rul), a = e._angle;
    e._pos = new THREE.Vector3(Math.cos(a) * r, 0.4, Math.sin(a) * r);
    q.setFromAxisAngle(new THREE.Vector3(0, 1, 0), -a);
    pods.setMatrixAt(i, m.compose(e._pos, q, s));
    pods.setColorAt(i, LEVEL[e.level].clone().multiplyScalar(e.level === 'LOW' ? 0.55 : e.level === 'MEDIUM' ? 0.9 : 1.1));
  });
  scene.add(pods);
  buildThresholds(p);
}

// ---------------------------------------------------------------- hero engine
const hero = new THREE.Group(); hero.position.y = HERO_Y; hero.scale.setScalar(1.35); scene.add(hero);
const comps = {};
const EXPLODE = { fan: -1.5, lpc: -0.85, hpc: -0.2, combustor: 0.45, hpt: 1.0, lpt: 1.65 };
const BASE_X = { fan: -2.06, lpc: -1.6, hpc: -0.8, combustor: 0.02, hpt: 0.43, lpt: 1.25 };
const frustum = (a, b, len, seg = 48, open = false) => new THREE.CylinderGeometry(a, b, len, seg, 1, open).rotateZ(Math.PI / 2);
const ringG = (r) => new THREE.CylinderGeometry(r, r, 0.04, 48).rotateZ(Math.PI / 2);

function compMat() { return new THREE.MeshStandardMaterial({ color: NEUTRAL.clone(), metalness: 0.75, roughness: 0.3, emissive: new THREE.Color(0) }); }
function addPart(key, geo, x) {
  const c = comps[key], mesh = new THREE.Mesh(geo, c.mat);
  mesh.position.x = x; mesh.userData.comp = key; c.group.add(mesh); return mesh;
}
COMP_KEYS.forEach((k, i) => {
  const group = new THREE.Group(); group.position.x = BASE_X[k]; hero.add(group);
  const l = label(SHORT[k]); l.position.set(0, i % 2 ? -1.0 : 1.05, 0); group.add(l);
  comps[k] = { group, mat: compMat(), label: l.element };
});
// shaft + nacelle (not components)
hero.add(new THREE.Mesh(frustum(0.07, 0.07, 4.4, 16), new THREE.MeshStandardMaterial({ color: 0x2a3142, metalness: 0.8, roughness: 0.3 })));
const nacelle = new THREE.Mesh(frustum(1.22, 1.1, 3.0, 72, true), new THREE.MeshPhysicalMaterial({
  color: 0x9fc4ff, metalness: 0.2, roughness: 0.15, transparent: true, opacity: 0.12, side: THREE.DoubleSide, depthWrite: false }));
nacelle.position.x = -0.95; hero.add(nacelle);
const lip = new THREE.Mesh(new THREE.TorusGeometry(1.2, 0.035, 12, 96).rotateY(Math.PI / 2), glowMat(0x5cc8ff, 1.8));
lip.position.x = -2.45; hero.add(lip);
// fan
addPart('fan', new THREE.ConeGeometry(0.3, 0.55, 32).rotateZ(Math.PI / 2), -0.39);
addPart('fan', frustum(0.3, 0.34, 0.25), 0);
const fanBlades = new THREE.Group(); comps.fan.group.add(fanBlades);
for (let i = 0; i < 22; i++) {
  const a = (i / 22) * Math.PI * 2, b = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.84, 0.2), comps.fan.mat);
  b.position.set(0, Math.cos(a) * 0.76, Math.sin(a) * 0.76); b.rotation.x = -a; b.rotation.y = 0.5; b.userData.comp = 'fan'; fanBlades.add(b);
}
// LPC
addPart('lpc', frustum(0.6, 0.56, 0.5), 0); [-0.18, 0, 0.18].forEach((x) => addPart('lpc', ringG(0.64), x));
// HPC
addPart('hpc', frustum(0.53, 0.38, 1.1), 0); for (let s = 0; s < 6; s++) addPart('hpc', ringG(0.56 - s * 0.03), -0.47 + s * 0.19);
// combustor
addPart('combustor', new THREE.TorusGeometry(0.46, 0.14, 20, 64).rotateY(Math.PI / 2), 0); addPart('combustor', frustum(0.38, 0.4, 0.42), 0);
// HPT
addPart('hpt', frustum(0.42, 0.48, 0.34), 0); [-0.1, 0.07].forEach((x) => addPart('hpt', ringG(0.5), x));
// LPT + exhaust
addPart('lpt', frustum(0.5, 0.66, 0.8), -0.23); [-0.5, -0.3, -0.1, 0.07].forEach((x, i) => addPart('lpt', ringG(0.55 + i * 0.035), x));
addPart('lpt', frustum(0.66, 0.5, 0.45, 48, true), 0.39); addPart('lpt', new THREE.ConeGeometry(0.3, 0.6, 32).rotateZ(-Math.PI / 2), 0.53);

// airflow particles (visual only, not data): bypass air stays cool, core air heats after the combustor
const FLOW_N = 1100, flowPos = new Float32Array(FLOW_N * 3), flowCol = new Float32Array(FLOW_N * 3), flowMeta = [];
for (let i = 0; i < FLOW_N; i++) {
  const core = Math.random() < 0.45, r = core ? 0.12 + Math.random() * 0.3 : 0.72 + Math.random() * 0.4, a = Math.random() * Math.PI * 2;
  flowMeta.push({ core, r, a, x: -3 + Math.random() * 6.2, v: 1.2 + Math.random() * 0.9 });
}
const flowGeo = new THREE.BufferGeometry();
flowGeo.setAttribute('position', new THREE.BufferAttribute(flowPos, 3));
flowGeo.setAttribute('color', new THREE.BufferAttribute(flowCol, 3));
const flow = new THREE.Points(flowGeo, new THREE.PointsMaterial({ size: 0.045, vertexColors: true, transparent: true, opacity: 0.85,
  blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false }));
hero.add(flow);
const COOL = new THREE.Color(0x5cc8ff).multiplyScalar(1.6), HOT = new THREE.Color(0xff7a2a).multiplyScalar(2.2), tmp = new THREE.Color();
function updateFlow(dt) {
  for (let i = 0; i < FLOW_N; i++) {
    const f = flowMeta[i];
    f.x += f.v * dt; f.a += dt * 0.6;
    if (f.x > 3.2) f.x = -3;
    const squeeze = f.core ? 1 - 0.35 * Math.max(0, Math.min(1, (f.x + 1.3) / 1.3)) : 1;
    flowPos[i * 3] = f.x; flowPos[i * 3 + 1] = Math.cos(f.a) * f.r * squeeze; flowPos[i * 3 + 2] = Math.sin(f.a) * f.r * squeeze;
    tmp.copy(COOL); if (f.core && f.x > 0) tmp.lerp(HOT, Math.min(1, f.x / 0.6));
    flowCol[i * 3] = tmp.r; flowCol[i * 3 + 1] = tmp.g; flowCol[i * 3 + 2] = tmp.b;
  }
  flowGeo.attributes.position.needsUpdate = true; flowGeo.attributes.color.needsUpdate = true;
}

function colourHero() {
  const e = state.sel; if (!e) return;
  COMP_KEYS.forEach((k) => {
    const v = e.component_shap[k] ?? 0, t = Math.min(Math.abs(v) / state.scaleMax, 1), c = comps[k];
    const col = NEUTRAL.clone().lerp(v < 0 ? NEG : POS, Math.sqrt(t));
    c.mat.color.copy(col);
    c.mat.emissive.copy(v < 0 ? NEG : POS).multiplyScalar((state.activeComp === k ? 0.9 : 0.1 + 0.6 * t) * Math.sqrt(t));
    c.label.innerHTML = `${SHORT[k]} <span class="${v < 0 ? 'neg' : 'pos'}">${sgn(v)}</span>`;
    c.label.classList.toggle('active', state.activeComp === k);
  });
}

// ---------------------------------------------------------------- camera flights
const flight = { t: 1, dur: 1.4, fromP: new THREE.Vector3(), toP: new THREE.Vector3(), fromT: new THREE.Vector3(), toT: new THREE.Vector3() };
const ease = (x) => (x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2);
function flyTo(pos, target, dur = 1.4) {
  Object.assign(flight, { t: 0, dur });
  flight.fromP.copy(camera.position); flight.toP.copy(pos); flight.fromT.copy(controls.target); flight.toT.copy(target);
}
const VIEWS = {
  fleet: () => [new THREE.Vector3(0, 17, 23), new THREE.Vector3(0, 1.8, 0)],
  inspect: () => [new THREE.Vector3(6.8, HERO_Y + 3.0, 10.5), new THREE.Vector3(0, HERO_Y - 0.3, 0)],
};
function setView(v, instant = false) {
  state.view = v;
  const [pos, target] = VIEWS[v]();
  if (instant) { camera.position.copy(pos); controls.target.copy(target); flight.t = 1; } else flyTo(pos, target);
  $('btnFleet').classList.toggle('active', v === 'fleet');
  $('btnInspect').classList.toggle('active', v === 'inspect');
}

// ---------------------------------------------------------------- selection + panels
function select(e) {
  state.sel = e;
  beam.position.set(e._pos.x, 1.3, e._pos.z);
  selRing.position.set(e._pos.x, 0.03, e._pos.z);
  if (link) { scene.remove(link); link.geometry.dispose(); }
  const curve = new THREE.QuadraticBezierCurve3(e._pos.clone(), new THREE.Vector3(e._pos.x * 0.45, HERO_Y + 1.5, e._pos.z * 0.45), new THREE.Vector3(0, HERO_Y - 0.6, 0));
  link = new THREE.Line(new THREE.BufferGeometry().setFromPoints(curve.getPoints(48)),
    new THREE.LineBasicMaterial({ color: LEVEL[e.level].clone().multiplyScalar(1.8), transparent: true, opacity: 0.8, toneMapped: false }));
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
  const title = up ? `Your engines <span class="muted">· ${state.upload?.name ?? 'upload'}</span>` : `NASA ${p.dataset} test fleet`;
  const flagged = p.engines.filter((e) => notesOf(e).length).length;
  let h = `<h2>${title}</h2><div class="muted small">${up ? `${p.engines.length} engine(s), ${p.n_rows} rows · ` : ''}
    model trained on ${p.dataset} (${p.n_regimes} condition${p.n_regimes > 1 ? 's' : ''}, ${p.fault_mode})</div>
    <div class="counts">${['HIGH', 'MEDIUM', 'LOW'].map((l) => `<div class="count ${l}"><div class="n">${p.counts[l]}</div><div class="l">${{ HIGH: 'act now', MEDIUM: 'plan maintenance', LOW: 'no action' }[l]}</div></div>`).join('')}</div>`;
  if (up && p.suggested_model) h += `<div class="alert info">Suggested model: <b>${p.suggested_model}</b>. ${p.suggestion_reason}</div>`;
  if (up && p.engines.some((e) => e.unknown_condition_share > 0)) h += `<div class="alert red">Some engines run at flight conditions this model never saw. Switch to the suggested model.</div>`;
  (p.notes || []).forEach((n) => { h += `<div class="alert info">${n}</div>`; });
  h += `<h3>Radar</h3><div class="small muted">Distance from the centre = <b style="color:var(--text)">predicted remaining life</b>.
    Glowing rings are the calibrated cut-offs: HIGH if P(fail within ${p.H}) ≥ ${p.p_cut} (= predicted RUL ≤ ${p.cutoff_high}),
    MEDIUM if P(fail within ${2 * p.H}) ≥ ${p.p_cut} (≤ ${p.cutoff_medium}). Derived from validation engines, not chosen.</div>`;
  if (p.test_rmse != null) h += `<div class="small muted" style="margin-top:6px">Test RMSE ${p.test_rmse} cycles.</div>`;
  if (state.evalMode && ev) h += `<div class="alert info">Evaluation: HIGH caught <b>${ev.high_caught} of ${ev.fail_within_H}</b> engines failing within ${p.H};
    <b>${ev.high_caught} of ${ev.high_total}</b> flags correct; failing within ${p.H} but rated LOW: <b>${ev.fail_within_H_rated_low}</b>.</div>`;
  h += `<h3>Highest risk${flagged ? ` · ⚠ ${flagged} with data warnings` : ''}</h3><div class="list">`;
  p.engines.slice(0, 12).forEach((e) => {
    h += `<div class="row ${state.sel === e ? 'sel' : ''}" data-id="${e.id}"><b>#${e.id}</b><span class="badge ${e.level}">${e.level}</span>
      <span class="p">${e.pred_rul.toFixed(1)} cyc · ${e.p_h.toFixed(2)}</span><span class="warnmark">${notesOf(e).length ? '⚠' : ''}</span></div>`;
  });
  h += `</div>`;
  if (up) h += `<div style="margin-top:12px;display:flex;gap:8px"><button class="btn" id="dlCsv">⇩ Download predictions (CSV)</button><button class="btn" id="newUpload">⇪ New file</button></div>`;
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
    <h3>Where in the engine · click a part</h3>`;
  COMP_KEYS.forEach((k) => {
    const v = e.component_shap[k] ?? 0, c = p.components[k], w = Math.min(Math.abs(v) / state.scaleMax, 1) * 50;
    h += `<div class="comp ${state.activeComp === k ? 'active' : ''}" data-comp="${k}"><span>${c.label}</span>
      <div class="track"><div class="fill" style="left:${v < 0 ? 50 - w : 50}%;width:${w}%;background:${v < 0 ? 'var(--neg)' : 'var(--pos)'}"></div></div>
      <span class="num ${v < 0 ? 'neg' : 'pos'}">${sgn(v)}</span></div>`;
    if (state.activeComp === k) h += `<div class="sensors">${c.sensors.length ? c.sensors.map((s) => `${s} ${c.sensor_names[s]}: <span class="${e.sensor_shap[s] < 0 ? 'neg' : 'pos'}">${sgn(e.sensor_shap[s], 2)}</span>`).join('<br>') : 'no sensors kept for this part in this dataset'}</div>`;
  });
  h += `<div class="check">Check: model average ${p.base_value.toFixed(1)} + parts ${sgn(compSum)} = <b>${(p.base_value + compSum).toFixed(1)}</b>
      vs prediction <b>${e.pred_rul.toFixed(1)}</b> (SHAP values add up exactly; tiny gaps are rounding).</div>
    <div class="check">Part colours = summed SHAP of the sensors located there (Saxena et al. 2008, Table 2; core speed and bleed are drawn on the HPC).
      The HPC has the most sensors, so compare individual sensors before concluding. Airflow particles are decorative.</div>`;
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
  status(`<div class="alert info"><span class="spinner" style="display:inline-block;width:14px;height:14px;border-width:2px;vertical-align:-2px"></span> Analysing <b>${state.upload.name}</b> with the ${model === 'auto' ? 'suggested' : model} model…</div>`);
  let res;
  try {
    res = await fetch(`api/predict?model=${model}`, { method: 'POST', body: state.upload.bytes });
  } catch {
    status(`<div class="alert red">The analysis server is not running. Start it with <code>python app/server.py</code> and open <code>http://localhost:8765</code>.</div>`);
    return;
  }
  const data = await res.json().catch(() => ({ error: `Server error (${res.status}). Start it with python app/server.py.` }));
  if (!res.ok || data.error) { status(`<div class="alert red">This file cannot be used: ${data.error}</div>`); return; }
  status(`<div class="alert info">✓ ${data.engines.length} engine(s) analysed with the <b>${data.dataset}</b> model.</div>`);
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
  if (state.explodeTarget && state.view !== 'inspect') setView('inspect');
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
  const dt = Math.min(clock.getDelta(), 0.05), t = clock.elapsedTime;
  if (flight.t < 1) {
    flight.t = Math.min(1, flight.t + dt / flight.dur);
    const k = ease(flight.t);
    camera.position.lerpVectors(flight.fromP, flight.toP, k); controls.target.lerpVectors(flight.fromT, flight.toT, k);
  }
  controls.update();
  sweep.rotation.z -= dt * 0.9;
  fanBlades.rotation.x += dt * 9;
  if (state.view === 'fleet' && !state.activeComp) hero.rotation.y += dt * 0.18;
  hero.position.y = HERO_Y + Math.sin(t * 1.2) * 0.08;
  state.exploded += (state.explodeTarget - state.exploded) * Math.min(1, dt * 4);
  COMP_KEYS.forEach((k) => { comps[k].group.position.x = BASE_X[k] + EXPLODE[k] * state.exploded; });
  nacelle.material.opacity = 0.12 * (1 - state.exploded); lip.visible = state.exploded < 0.5;
  updateFlow(dt); flow.visible = state.exploded < 0.3;
  if (pods && state.payload) {                                       // HIGH engines pulse
    const c = new THREE.Color(), pulse = 0.9 + 0.7 * (0.5 + 0.5 * Math.sin(t * 4));
    state.payload.engines.forEach((e, i) => { if (e.level === 'HIGH') pods.setColorAt(i, c.copy(LEVEL.HIGH).multiplyScalar(pulse)); });
    pods.instanceColor.needsUpdate = true;
  }
  beam.material.opacity = 0.12 + 0.06 * Math.sin(t * 3);
  selRing.rotation.z += dt;
  composer.render();
  labelRenderer.render(scene, camera);
  requestAnimationFrame(animate);
}

// ---------------------------------------------------------------- start (URL options for prepared demos)
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
  // plain load: cinematic fly-in; prepared demo links (any URL option): jump straight to the view
  setView(params.get('view') === 'inspect' || params.get('comp') ? 'inspect' : 'fleet', [...params.keys()].length > 0);
  $('loader').classList.add('done');
  document.body.dataset.ready = '1';
  animate();
})();
