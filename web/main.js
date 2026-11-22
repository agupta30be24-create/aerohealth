// AeroHealth 3D frontend. Displays results exported by src/export_frontend.py (no model code runs here).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';

window.addEventListener('error', (e) => { document.body.dataset.error = String(e.message); });
window.addEventListener('unhandledrejection', (e) => { document.body.dataset.error = String(e.reason); });

const LEVEL_HEX = { HIGH: 0xe5484d, MEDIUM: 0xf5a524, LOW: 0x30a46c };
const NEG = new THREE.Color(0xef4444), POS = new THREE.Color(0x3b82f6), NEUTRAL = new THREE.Color(0x6b7280);
const state = { data: null, sel: null, evalMode: false, activeComp: null, scaleMax: 1 };
const cache = {};
const $ = (id) => document.getElementById(id);
const sgn = (v, d = 1) => `${v < 0 ? '−' : '+'}${Math.abs(v).toFixed(d)}`;

// ---------------------------------------------------------------- generic 3D view
function makeView(container, camPos) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x161b22);
  container.appendChild(renderer.domElement);
  const labels = new CSS2DRenderer();
  Object.assign(labels.domElement.style, { position: 'absolute', top: '0', left: '0', pointerEvents: 'none' });
  container.appendChild(labels.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 500);
  camera.position.set(...camPos);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  scene.add(new THREE.HemisphereLight(0xffffff, 0x223344, 1.3));
  const sun = new THREE.DirectionalLight(0xffffff, 1.7);
  sun.position.set(6, 10, 8);
  scene.add(sun);

  const view = { container, renderer, labels, scene, camera, controls };
  const resize = () => {
    const w = container.clientWidth, h = container.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h);
    labels.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe(container);
  resize();
  return view;
}

// Distinguish a click from an orbit drag.
function onClick(el, handler) {
  let down = null;
  el.addEventListener('pointerdown', (e) => { down = [e.clientX, e.clientY]; });
  el.addEventListener('pointerup', (e) => {
    if (down && Math.hypot(e.clientX - down[0], e.clientY - down[1]) < 5) handler(e);
    down = null;
  });
}

function ndc(e, el) {
  const r = el.getBoundingClientRect();
  return new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
}

// ---------------------------------------------------------------- fleet view
const fleet = makeView($('fleetView'), [0, 12, 16]);
const grid = new THREE.GridHelper(60, 60, 0x30363d, 0x21262d);
fleet.scene.add(grid);
let fleetMesh = null, stems = null;
const selRing = new THREE.Mesh(new THREE.RingGeometry(0.5, 0.62, 40), new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide }));
selRing.rotation.x = -Math.PI / 2;
fleet.scene.add(selRing);
const axisLabel = new CSS2DObject(Object.assign(document.createElement('div'), { className: 'label3d' }));
fleet.scene.add(axisLabel);

function buildFleet() {
  for (const obj of [fleetMesh, stems]) if (obj) { fleet.scene.remove(obj); obj.geometry.dispose(); obj.material.dispose(); }
  const E = state.data.engines, n = E.length;
  const cols = Math.ceil(Math.sqrt(n * 1.6)), rows = Math.ceil(n / cols), sp = 1.25;

  const geo = new THREE.CylinderGeometry(0.4, 0.3, 0.95, 24);
  geo.rotateZ(Math.PI / 2);                                   // engine pod lies along x
  fleetMesh = new THREE.InstancedMesh(geo, new THREE.MeshStandardMaterial({ metalness: 0.35, roughness: 0.45 }), n);
  const m = new THREE.Matrix4(), c = new THREE.Color(), stemPos = [];
  E.forEach((e, i) => {
    const x = ((i % cols) - (cols - 1) / 2) * sp, z = (Math.floor(i / cols) - (rows - 1) / 2) * sp;
    const y = 0.45 + e.p_h * 3.0;                              // height = P(fail within H)
    e._pos = new THREE.Vector3(x, y, z);
    m.makeTranslation(x, y, z);
    fleetMesh.setMatrixAt(i, m);
    fleetMesh.setColorAt(i, c.setHex(LEVEL_HEX[e.level]));
    stemPos.push(x, 0, z, x, y - 0.35, z);
  });
  fleet.scene.add(fleetMesh);
  const sg = new THREE.BufferGeometry();
  sg.setAttribute('position', new THREE.Float32BufferAttribute(stemPos, 3));
  stems = new THREE.LineSegments(sg, new THREE.LineBasicMaterial({ color: 0x484f58 }));
  fleet.scene.add(stems);

  const size = Math.max(cols, rows) * sp;
  fleet.camera.position.set(0, size * 0.85, size * 1.3);
  fleet.controls.target.set(0, 0.9, 0);
  axisLabel.position.set(0, 4.2, -(rows / 2) * sp - 0.5);
  axisLabel.element.textContent = `↑ higher = more likely to fail within ${state.data.H} cycles (sorted: riskiest first)`;
}

const ray = new THREE.Raycaster();
function engineAt(e) {
  if (!fleetMesh) return null;
  ray.setFromCamera(ndc(e, fleet.renderer.domElement), fleet.camera);
  const hit = ray.intersectObject(fleetMesh)[0];
  return hit ? state.data.engines[hit.instanceId] : null;
}

fleet.renderer.domElement.addEventListener('pointermove', (e) => {
  const eng = engineAt(e), tip = $('tooltip');
  if (!eng) { tip.classList.add('hidden'); fleet.renderer.domElement.style.cursor = ''; return; }
  fleet.renderer.domElement.style.cursor = 'pointer';
  const r = $('fleetPane').getBoundingClientRect();
  tip.innerHTML = `<b>Engine ${eng.id}</b> <span class="badge ${eng.level}">${eng.level}</span><br>
    Predicted RUL ${eng.pred_rul.toFixed(1)} cycles<br>P(fail ≤ ${state.data.H}) ${eng.p_h.toFixed(2)}`
    + (state.evalMode ? `<br>True RUL ${eng.true_rul}` : '') + (eng.padded ? '<br><i>short history (padded)</i>' : '');
  tip.style.left = `${e.clientX - r.left + 14}px`;
  tip.style.top = `${e.clientY - r.top + 14}px`;
  tip.classList.remove('hidden');
});
fleet.renderer.domElement.addEventListener('pointerleave', () => $('tooltip').classList.add('hidden'));
onClick(fleet.renderer.domElement, (e) => { const eng = engineAt(e); if (eng) select(eng); });

// ---------------------------------------------------------------- engine cutaway view
const engine = makeView($('engineView'), [3.2, 2.4, 6.2]);
engine.controls.target.set(-0.1, 0, 0);
const comps = {};   // key -> { mats: [], label }
const COMP_KEYS = ['fan', 'lpc', 'hpc', 'combustor', 'hpt', 'lpt'];
COMP_KEYS.forEach((k) => { comps[k] = { mats: [], meshes: [] }; });

function addMesh(geo, comp, x, opts = {}) {
  const mat = new THREE.MeshStandardMaterial({ color: NEUTRAL.clone(), metalness: 0.5, roughness: 0.4, ...opts });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.x = x;
  mesh.userData.comp = comp;
  engine.scene.add(mesh);
  if (comp) { comps[comp].mats.push(mat); comps[comp].meshes.push(mesh); }
  return mesh;
}
// frustum along x: radius rFront at the intake side (−x), rBack at the exhaust side (+x)
const frustum = (rFront, rBack, len, seg = 48, open = false) =>
  new THREE.CylinderGeometry(rFront, rBack, len, seg, 1, open).rotateZ(Math.PI / 2);
const ring = (r, thick = 0.03, len = 0.04) => new THREE.CylinderGeometry(r, r, len, 48).rotateZ(Math.PI / 2);

function buildEngineModel() {
  // shaft (not a component)
  addMesh(frustum(0.07, 0.07, 4.3, 16), null, -0.15, { color: 0x3a3f47 });
  // FAN: spinner, hub, blades, nacelle/bypass duct
  addMesh(new THREE.ConeGeometry(0.3, 0.55, 32).rotateZ(Math.PI / 2), 'fan', -2.45);
  addMesh(frustum(0.3, 0.34, 0.25), 'fan', -2.06);
  const fanGroup = new THREE.Group();
  fanGroup.position.x = -2.06;
  for (let i = 0; i < 18; i++) {
    const a = (i / 18) * Math.PI * 2;
    const blade = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.82, 0.2), null);
    blade.position.set(0, Math.cos(a) * 0.74, Math.sin(a) * 0.74);
    blade.rotation.x = -a;
    blade.rotation.y = 0.45;
    blade.userData.comp = 'fan';
    fanGroup.add(blade);
    comps.fan.meshes.push(blade);
  }
  const bladeMat = new THREE.MeshStandardMaterial({ color: NEUTRAL.clone(), metalness: 0.6, roughness: 0.35 });
  fanGroup.children.forEach((b) => { b.material = bladeMat; });
  comps.fan.mats.push(bladeMat);
  engine.scene.add(fanGroup);
  engine.userData = { fanGroup };
  addMesh(frustum(1.22, 1.12, 3.0, 64, true), 'fan', -0.95, { transparent: true, opacity: 0.16, side: THREE.DoubleSide, depthWrite: false });

  // LPC
  addMesh(frustum(0.6, 0.56, 0.5), 'lpc', -1.6);
  [-1.78, -1.6, -1.42].forEach((x) => addMesh(ring(0.64), 'lpc', x));
  // HPC: tapering, 6 stages
  addMesh(frustum(0.53, 0.38, 1.1), 'hpc', -0.8);
  for (let s = 0; s < 6; s++) addMesh(ring(0.56 - s * 0.03), 'hpc', -1.27 + s * 0.19);
  // Combustor: annular torus around the core
  addMesh(new THREE.TorusGeometry(0.46, 0.14, 20, 56).rotateY(Math.PI / 2), 'combustor', 0.02);
  addMesh(frustum(0.38, 0.4, 0.42), 'combustor', 0.02);
  // HPT
  addMesh(frustum(0.42, 0.48, 0.34), 'hpt', 0.43);
  [0.33, 0.5].forEach((x) => addMesh(ring(0.5), 'hpt', x));
  // LPT + exhaust
  addMesh(frustum(0.5, 0.66, 0.8), 'lpt', 1.02);
  [0.75, 0.95, 1.15, 1.32].forEach((x, i) => addMesh(ring(0.55 + i * 0.035), 'lpt', x));
  addMesh(frustum(0.66, 0.5, 0.45, 48, true), 'lpt', 1.64, { side: THREE.DoubleSide });
  addMesh(new THREE.ConeGeometry(0.3, 0.6, 32).rotateZ(-Math.PI / 2), 'lpt', 1.78);

  // labels, alternating above/below to avoid overlap
  const labelPos = { fan: [-2.06, 1.5], lpc: [-1.6, -0.95], hpc: [-0.8, 0.95], combustor: [0.02, -0.9], hpt: [0.43, 1.05], lpt: [1.25, -1.0] };
  COMP_KEYS.forEach((k) => {
    const el = document.createElement('div');
    el.className = 'label3d';
    const obj = new CSS2DObject(el);
    obj.position.set(labelPos[k][0], labelPos[k][1], 0);
    engine.scene.add(obj);
    comps[k].label = el;
  });
}
buildEngineModel();

const SHORT = { fan: 'Fan', lpc: 'LPC', hpc: 'HPC', combustor: 'Combustor', hpt: 'HPT', lpt: 'LPT' };
function colourEngine() {
  const e = state.sel;
  COMP_KEYS.forEach((k) => {
    const v = e.component_shap[k] ?? 0;
    const t = Math.min(Math.abs(v) / state.scaleMax, 1);
    const col = NEUTRAL.clone().lerp(v < 0 ? NEG : POS, Math.sqrt(t));
    const active = state.activeComp === k;
    comps[k].mats.forEach((mat) => {
      mat.color.copy(col);
      mat.emissive.copy(col).multiplyScalar(active ? 0.55 : 0.25 * t);
    });
    comps[k].label.innerHTML = `${SHORT[k]} <span class="${v < 0 ? 'neg' : 'pos'}">${sgn(v)}</span>`;
    comps[k].label.style.borderColor = active ? '#e6edf3' : '';
  });
}

const engRay = new THREE.Raycaster();
onClick(engine.renderer.domElement, (e) => {
  engRay.setFromCamera(ndc(e, engine.renderer.domElement), engine.camera);
  const hit = engRay.intersectObjects(engine.scene.children, true).find((h) => h.object.userData.comp);
  setActiveComp(hit ? hit.object.userData.comp : null);
});

// ---------------------------------------------------------------- panels
function setActiveComp(k) {
  state.activeComp = state.activeComp === k ? null : k;
  colourEngine();
  renderInfo();
}

function renderSummary() {
  const d = state.data, ev = d.evaluation;
  $('summary').innerHTML =
    `<span><b style="color:var(--high)">${d.counts.HIGH}</b> HIGH</span>
     <span><b style="color:var(--medium)">${d.counts.MEDIUM}</b> MEDIUM</span>
     <span><b style="color:var(--low)">${d.counts.LOW}</b> LOW</span>
     <span>test RMSE <b>${d.test_rmse}</b></span>`
    + (state.evalMode ? `<span>HIGH caught <b>${ev.high_caught}/${ev.fail_within_H}</b>, failing-but-LOW <b>${ev.fail_within_H_rated_low}</b></span>` : '');
}

function renderInfo() {
  const d = state.data, e = state.sel, H = d.H;
  const compSum = COMP_KEYS.reduce((s, k) => s + (e.component_shap[k] ?? 0), 0);
  const kpi = (v, l) => `<div class="kpi"><div class="v">${v}</div><div class="l">${l}</div></div>`;

  let html = `<h3>Engine ${e.id}</h3><div class="kpis">
      ${kpi(`<span class="badge ${e.level}">${e.level}</span>`, 'risk level')}
      ${kpi(e.pred_rul.toFixed(1), 'predicted RUL (cycles)')}
      ${kpi(e.p_h.toFixed(2), `P(fail within ${H})`)}
      ${kpi(state.evalMode ? e.true_rul : 'hidden', state.evalMode ? 'true RUL (benchmark only)' : 'true RUL (turn on evaluation mode)')}
    </div>`;
  if (e.padded) html += `<div class="warn">Fewer than 30 recorded cycles: window padded by repeating the first cycle, so this prediction is less informed.</div>`;

  html += `<h3>Why this prediction</h3>
    <div>Fleet-average prediction is <b>${d.base_value.toFixed(1)}</b> cycles. Largest reasons this engine differs:</div>
    <ul class="drivers">${e.drivers.map((r) =>
      `<li>${r.text}: <span class="${r.shap < 0 ? 'neg' : 'pos'}">${r.shap < 0 ? 'shortens' : 'extends'} life by ${Math.abs(r.shap).toFixed(1)}</span> cycles</li>`).join('')}</ul>`;

  html += `<h3>Where in the engine (SHAP summed per component, click to expand)</h3>`;
  COMP_KEYS.forEach((k) => {
    const v = e.component_shap[k] ?? 0, c = d.components[k];
    const w = Math.min(Math.abs(v) / state.scaleMax, 1) * 50;
    html += `<div class="comp ${state.activeComp === k ? 'active' : ''}" data-comp="${k}">
        <span>${c.label}</span>
        <div class="track"><div class="fill" style="left:${v < 0 ? 50 - w : 50}%;width:${w}%;background:${v < 0 ? 'var(--neg)' : 'var(--pos)'}"></div></div>
        <span class="num ${v < 0 ? 'neg' : 'pos'}">${sgn(v)}</span></div>`;
    if (state.activeComp === k) {
      html += `<div class="sensors">${c.sensors.length ? c.sensors.map((s) =>
        `${s} ${c.sensor_names[s]}: <span class="${e.sensor_shap[s] < 0 ? 'neg' : 'pos'}">${sgn(e.sensor_shap[s], 2)}</span>`).join('<br>')
        : 'no sensors kept for this component in this dataset'}</div>`;
    }
  });
  html += `<div class="check">Integrity check: fleet average ${d.base_value.toFixed(1)} + component sum ${sgn(compSum)}
    = <b>${(d.base_value + compSum).toFixed(1)}</b>, model prediction <b>${e.pred_rul.toFixed(1)}</b> (SHAP values add up exactly; small gaps are rounding).</div>`;

  html += `<h3>Fleet: ${d.dataset} (${d.fault_mode}, ${d.n_regimes} operating condition${d.n_regimes > 1 ? 's' : ''})</h3>
    <div>Model ${d.model}, test RMSE <b>${d.test_rmse}</b> cycles. HIGH if P(fail within ${H}) ≥ ${d.p_cut}
    (= predicted RUL ≤ <b>${d.cutoff_high}</b>), MEDIUM if P(fail within ${2 * H}) ≥ ${d.p_cut} (≤ <b>${d.cutoff_medium}</b>).
    Cut-offs are derived from the calibration, not chosen.</div>`;
  if (state.evalMode) {
    const ev = d.evaluation;
    html += `<div style="margin-top:6px">Evaluation (test): HIGH caught <b>${ev.high_caught} of ${ev.fail_within_H}</b> engines failing within ${H};
      <b>${ev.high_caught} of ${ev.high_total}</b> HIGH flags correct; failing within ${H} but rated LOW: <b>${ev.fail_within_H_rated_low}</b>.</div>`;
  }
  html += `<div style="margin-top:6px">Average |SHAP| share by component: ${COMP_KEYS.map((k) => `${SHORT[k]} ${(d.components[k].fleet_share * 100).toFixed(0)}%`).join(' · ')}</div>
    <p class="caveat">Component totals grow with the number of sensors assigned (HPC has the most), so compare individual sensors
    (click a component) before concluding that one part dominates. SHAP explains what the model uses, not physical causation.</p>`;

  $('info').innerHTML = html;
  $('info').querySelectorAll('.comp').forEach((el) => el.addEventListener('click', () => setActiveComp(el.dataset.comp)));
}

function select(e) {
  state.sel = e;
  selRing.position.set(e._pos.x, 0.02, e._pos.z);
  $('engineTitle').textContent = `${e.id}: ${e.level}, predicted RUL ${e.pred_rul.toFixed(1)}`;
  colourEngine();
  renderInfo();
}

// ---------------------------------------------------------------- data loading
async function loadDataset(fd) {
  if (!cache[fd]) {
    const res = await fetch(`data/${fd}.json`);
    if (!res.ok) throw new Error(`Could not load data/${fd}.json (run python src/export_frontend.py)`);
    cache[fd] = await res.json();
  }
  state.data = cache[fd];
  state.activeComp = null;
  // colour scale: 95th percentile of |component SHAP| over the fleet, so one outlier does not wash out the rest
  const vals = state.data.engines.flatMap((e) => COMP_KEYS.map((k) => Math.abs(e.component_shap[k] ?? 0))).sort((a, b) => a - b);
  state.scaleMax = Math.max(vals[Math.floor(vals.length * 0.95)] || 1, 1e-6);
  buildFleet();
  renderSummary();
  select(state.data.engines[0]);   // riskiest engine
}

$('dataset').addEventListener('change', (e) => loadDataset(e.target.value));
$('evalMode').addEventListener('change', (e) => { state.evalMode = e.target.checked; renderSummary(); renderInfo(); });

// ---------------------------------------------------------------- render loop
const clock = new THREE.Clock();
function animate() {
  const dt = clock.getDelta();
  engine.userData.fanGroup.rotation.x += dt * 2.2;
  selRing.rotation.z += dt * 0.8;
  for (const v of [fleet, engine]) {
    v.controls.update();
    v.renderer.render(v.scene, v.camera);
    v.labels.render(v.scene, v.camera);
  }
  requestAnimationFrame(animate);
}

window.__aero = { state, select, loadDataset };

// Optional URL parameters for prepared demo views, e.g. index.html?fd=FD004&eval=1&engine=10&comp=hpc
const params = new URLSearchParams(location.search);
const startFd = ['FD001', 'FD002', 'FD003', 'FD004'].includes(params.get('fd')) ? params.get('fd') : 'FD001';
$('dataset').value = startFd;
if (params.get('eval') === '1') { state.evalMode = true; $('evalMode').checked = true; }
loadDataset(startFd).then(() => {
  const eng = state.data.engines.find((e) => String(e.id) === params.get('engine'));
  if (eng) select(eng);
  if (COMP_KEYS.includes(params.get('comp'))) setActiveComp(params.get('comp'));
  renderSummary();
  document.body.dataset.ready = '1';
  animate();
});
