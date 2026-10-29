// SO-101 QC Cell: in-browser replay of the MuJoCo scene and the scripted episodes.
// scene/*.bin is generated from the sim by site/export_scene.py; no physics here, the sim's frames are played back.
import * as THREE from "three";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

const D = window.FLYWHEEL;
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) {} },
};
const TYPE_NAME = { rim: "rim", gear: "gear", pad: "brake pad", bracket: "bracket" };
const DEFECT_NAME = {
  rim: { missing_hole: "no hub bore", broken_corner: "broken spoke" },
  gear: { missing_hole: "no shaft bore", broken_corner: "missing tooth" },
  pad: { missing_hole: "no ear holes", broken_corner: "broken corner" },
  bracket: { missing_hole: "no hole", broken_corner: "broken corner" },
};
const partById = Object.fromEntries(D.parts.map((p) => [p.id, p]));
const decide = (p, order) => (p.defect === null && p.variant === order ? "ok" : "reject");
const describe = (p) => `${TYPE_NAME[p.type]} · ${p.variant === "W" ? "white" : "orange"} · ${p.defect ? DEFECT_NAME[p.type][p.defect] : "sound"}`;
const COLOR = { W: new THREE.Color().setRGB(0.93, 0.93, 0.91, THREE.SRGBColorSpace), T: new THREE.Color().setRGB(0.9, 0.45, 0.1, THREE.SRGBColorSpace) };
const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const W = (x, y, z) => new THREE.Vector3(x, z, -y); // MuJoCo (z up) → three (y up)

/* ---------------------------------------------------------------- loading */
async function fetchBuf(url, weight, prog) {
  const res = await fetch(url);
  const total = +res.headers.get("content-length") || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    prog(weight, total ? got / total : Math.min(0.95, got / 3e6));
  }
  const out = new Uint8Array(got);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  prog(weight, 1);
  return out.buffer;
}
const parts = {};
const prog = (k, v) => {
  parts[k] = v;
  const p = Math.round(((parts.m || 0) * 0.75 + (parts.f || 0) * 0.2 + (parts.s || 0) * 0.05) * 100);
  $("#load-pct").textContent = p;
  $("#load-bar").style.width = p + "%";
};
const [S, MESH, FRAMES] = await Promise.all([
  fetch("scene/scene.json").then((r) => r.json()).then((j) => (prog("s", 1), j)),
  fetchBuf("scene/meshes.bin", "m", prog),
  fetchBuf("scene/frames.bin", "f", prog),
]);
const F = new Float32Array(FRAMES);
const NB = S.nbody, NF = S.frames;

/* ---------------------------------------------------------------- renderer + scene */
const canvas = $("#gl");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x08080a);
scene.fog = new THREE.Fog(0x08080a, 1.8, 4.6);
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
scene.environmentIntensity = 0.32;

const key = new THREE.DirectionalLight(0xfff4e8, 2.4);
key.position.set(-1.1, 2.2, 0.9);
key.castShadow = true;
key.shadow.mapSize.set(2048, 2048);
Object.assign(key.shadow.camera, { left: -0.8, right: 0.8, top: 0.8, bottom: -0.8, near: 0.5, far: 5 });
key.shadow.bias = -0.0004;
key.shadow.normalBias = 0.01;
scene.add(key);
const rim = new THREE.DirectionalLight(0x9fb4ff, 0.9);
rim.position.set(1.6, 0.9, -1.4);
scene.add(rim);
const warm = new THREE.PointLight(0xff7a26, 0.0, 1.4, 2);
scene.add(warm);
scene.add(new THREE.HemisphereLight(0xbfc6d6, 0x0a0a0c, 0.35));

// floor: dark studio, disc fading toward the edges
const floorTex = (() => {
  const c = document.createElement("canvas");
  c.width = c.height = 512;
  const g = c.getContext("2d");
  const grd = g.createRadialGradient(256, 256, 0, 256, 256, 256);
  grd.addColorStop(0, "#1a1a1e");
  grd.addColorStop(0.55, "#111114");
  grd.addColorStop(1, "#08080a");
  g.fillStyle = grd;
  g.fillRect(0, 0, 512, 512);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
})();
const floor = new THREE.Mesh(new THREE.CircleGeometry(14, 96), new THREE.MeshStandardMaterial({ map: floorTex, roughness: 0.92, metalness: 0 }));
floor.rotation.x = -Math.PI / 2;
floor.position.set(0.15, -0.0008, 0);
floor.receiveShadow = true;
scene.add(floor);

// MuJoCo world: z up → y up
const root = new THREE.Group();
root.rotation.x = -Math.PI / 2;
scene.add(root);
const bodies = S.bodies.map((name, i) => {
  const g = new THREE.Group();
  g.name = name;
  (i === 0 ? root : root).add(g);
  return g;
});

/* ---------------------------------------------------------------- geometry */
function meshGeo(m) {
  const q = new Int16Array(MESH, m.vOff, m.vCount * 3);
  const pos = new Float32Array(m.vCount * 3);
  for (let i = 0; i < pos.length; i++) pos[i] = (q[i] + 32768) * m.scale[i % 3] + m.min[i % 3];
  const idx = m.i32 ? new Uint32Array(MESH, m.iOff, m.iCount) : new Uint16Array(MESH, m.iOff, m.iCount);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geo.setIndex(new THREE.BufferAttribute(m.i32 ? new Uint32Array(idx) : new Uint16Array(idx), 1));
  const flat = geo.toNonIndexed();
  flat.computeVertexNormals();
  return flat;
}
const texLoader = new THREE.TextureLoader();
const texCache = {};
function tex(name) {
  if (!texCache[name]) {
    const t = texLoader.load(`scene/tex/${name}.png`);
    t.colorSpace = THREE.SRGBColorSpace;
    t.anisotropy = 8;
    if (name !== "baseplate") { t.magFilter = THREE.NearestFilter; t.generateMipmaps = false; t.minFilter = THREE.LinearFilter; }
    texCache[name] = t;
  }
  return texCache[name];
}
const ARM = new Set(["base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw"]);
const isPart = (name) => /^[JDBL]-[WT]-\d\d$/.test(name);
const binMats = { ok: [], reject: [] };
const armMats = [];

for (const g of S.geoms) {
  const [a, b, c] = g.size;
  let geo;
  if (g.type === "box") geo = new THREE.BoxGeometry(2 * a, 2 * b, 2 * c);
  else if (g.type === "plane") geo = new THREE.PlaneGeometry(2 * a || 0.01, 2 * b || 0.01);
  else if (g.type === "cylinder") { geo = new THREE.CylinderGeometry(a, a, 2 * b, 48); geo.rotateX(Math.PI / 2); }
  else if (g.type === "capsule") { geo = new THREE.CapsuleGeometry(a, 2 * b, 8, 24); geo.rotateX(Math.PI / 2); }
  else if (g.type === "sphere") geo = new THREE.SphereGeometry(a, 24, 16);
  else if (g.type === "ellipsoid") { geo = new THREE.SphereGeometry(1, 24, 16); geo.scale(a, b, c); }
  else geo = meshGeo(S.meshes[g.mesh]);
  const bodyName = S.bodies[g.body];
  const color = new THREE.Color().setRGB(g.rgba[0], g.rgba[1], g.rgba[2], THREE.SRGBColorSpace);
  let mat;
  if (ARM.has(bodyName)) {
    mat = new THREE.MeshPhysicalMaterial({ color, roughness: 0.42, metalness: 0.0, clearcoat: 0.35, clearcoatRoughness: 0.4 });
  } else if (isPart(bodyName)) {
    mat = new THREE.MeshPhysicalMaterial({ color, roughness: 0.5, clearcoat: 0.2 });
  } else {
    mat = new THREE.MeshStandardMaterial({ color, roughness: 0.82, metalness: 0.02 });
  }
  if (g.tex) {
    const t = tex(g.tex.name).clone();
    if (g.tex.name === "baseplate") {
      t.wrapS = t.wrapT = THREE.RepeatWrapping;
      t.repeat.set(2 * a * g.tex.repeat[0], 2 * b * g.tex.repeat[1]);
    }
    mat.map = t;
    mat.color.set(0xffffff);
    if (g.tex.name.startsWith("tag") || g.tex.name.startsWith("pocket")) mat.roughness = 1;
  }
  if (g.rgba[3] < 1) { mat.transparent = true; mat.opacity = g.rgba[3]; }
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.fromArray(g.pos);
  mesh.quaternion.set(g.quat[1], g.quat[2], g.quat[3], g.quat[0]);
  mesh.castShadow = g.type !== "plane";
  mesh.receiveShadow = true;
  bodies[g.body].add(mesh);
  if (bodyName === "bin_ok" || bodyName === "bin_reject") binMats[bodyName.slice(4)].push(mat);
  if (ARM.has(bodyName)) { armMats.push(mat); mesh.userData.arm = true; }
}

// sim cameras (top and wrist); a MuJoCo camera also looks along -z, +y up
const simCams = S.cameras.map((c) => {
  const cam = new THREE.PerspectiveCamera(c.fovy, 4 / 3, 0.005, 6);
  cam.position.fromArray(c.pos);
  cam.quaternion.set(c.quat[1], c.quat[2], c.quat[3], c.quat[0]);
  bodies[c.body].add(cam);
  cam.userData.name = c.name;
  return cam;
});

/* ---------------------------------------------------------------- playback */
const qa = new THREE.Quaternion(), qb = new THREE.Quaternion();
function setFrame(t) {
  t = ((t % NF) + NF) % NF;
  const f0 = Math.floor(t), f1 = (f0 + 1) % NF, a = t - f0;
  for (let b = 1; b < NB; b++) {
    const o = (f0 * NB + b) * 7, p = (f1 * NB + b) * 7;
    const g = bodies[b];
    g.position.set(F[o] + (F[p] - F[o]) * a, F[o + 1] + (F[p + 1] - F[o + 1]) * a, F[o + 2] + (F[p + 2] - F[o + 2]) * a);
    qa.set(F[o + 4], F[o + 5], F[o + 6], F[o + 3]);
    qb.set(F[p + 4], F[p + 5], F[p + 6], F[p + 3]);
    g.quaternion.copy(qa.slerp(qb, a));
  }
}
const frame0 = (b) => ({ pos: [F[b * 7], F[b * 7 + 1], F[b * 7 + 2]], quat: [F[b * 7 + 3], F[b * 7 + 4], F[b * 7 + 5], F[b * 7 + 6]] });
const partBodies = S.bodies.map((n, i) => (isPart(n) ? i : -1)).filter((i) => i > 0);
const phaseAt = (f) => { let ph = S.phases[0][1]; for (const [s, n] of S.phases) { if (s <= f) ph = n; else break; } return ph; };
const episodeAt = (f) => Math.max(0, S.episodes.findIndex((e) => f >= e.start && f <= e.end));
setFrame(0);

/* ---------------------------------------------------------------- parts showcase and generic parts */
const showMesh = {};
for (const s of S.showcase) showMesh[`${s.type}|${s.defect}`] = meshGeo(s);
const partMat = { W: new THREE.MeshPhysicalMaterial({ color: COLOR.W, roughness: 0.5, clearcoat: 0.2 }), T: new THREE.MeshPhysicalMaterial({ color: COLOR.T, roughness: 0.5, clearcoat: 0.2 }) };
function genericPart(type, defect, variant) {
  const m = new THREE.Mesh(showMesh[`${type}|${defect}`], partMat[variant]);
  m.castShadow = m.receiveShadow = true;
  return m;
}
const SHOW_CENTER = [0.12, -0.66];
const SHOW_COLS = [["W", null], ["W", "missing_hole"], ["W", "broken_corner"], ["T", null], ["T", "missing_hole"], ["T", "broken_corner"]];
const SHOW_ROWS = ["rim", "gear", "pad", "bracket"];
const showcase = new THREE.Group();
root.add(showcase);
const showItems = [];
const pedestal = new THREE.CylinderGeometry(0.026, 0.028, 0.006, 48).rotateX(Math.PI / 2);
const pedMat = new THREE.MeshStandardMaterial({ color: 0x17171b, roughness: 0.6, metalness: 0.3 });
SHOW_ROWS.forEach((type, r) => SHOW_COLS.forEach(([variant, defect], c) => {
  const x = SHOW_CENTER[0] + (r - 1.5) * 0.085, y = SHOW_CENTER[1] + (2.5 - c) * 0.075;
  const ped = new THREE.Mesh(pedestal, pedMat);
  ped.position.set(x, y, 0.004);
  ped.receiveShadow = true;
  const holder = new THREE.Group();
  holder.position.set(x, y, 0.009);
  holder.add(genericPart(type, defect, variant));
  showcase.add(ped, holder);
  showItems.push({ holder, type, variant, defect, pos: [x, y, 0.06], phase: (r * 6 + c) * 0.7 });
}));

/* ---------------------------------------------------------------- recording assistant: 3D tray, ring, arrow */
const trayBodies = S.tray.pockets.map((pid) => S.bodies.indexOf(pid));
const pocketPos = trayBodies.map((b) => frame0(b).pos);
const typeQuat = {};
trayBodies.forEach((b) => { const t = partById[S.bodies[b]].type; if (!typeQuat[t]) typeQuat[t] = frame0(b).quat; });
const BIN_C = { ok: [0.063, -0.147], reject: [0.063, 0.147] };
const opGroup = new THREE.Group();
root.add(opGroup);
const ringMat = new THREE.MeshBasicMaterial({ color: 0xff7a26, transparent: true, opacity: 0.95, depthWrite: false });
const ring = new THREE.Mesh(new THREE.RingGeometry(0.022, 0.0255, 64), ringMat);
const arrowMat = new THREE.MeshBasicMaterial({ color: 0xff7a26, transparent: true, opacity: 0.9 });
let arrow = null;
ring.layers.set(1);
opGroup.add(ring);

function placeGeneric(id, pos, quat, scale = 1) {
  const p = partById[id];
  const m = genericPart(p.type, p.defect, p.variant);
  m.position.fromArray(pos);
  m.quaternion.set(quat[1], quat[2], quat[3], quat[0]);
  m.scale.setScalar(scale);
  return m;
}

/* ---------------------------------------------------------------- camera */
const cam = new THREE.PerspectiveCamera(32, 1, 0.02, 20);
cam.layers.enable(1);
const view = { pos: new THREE.Vector3(1.6, 1.4, 1.6), target: new THREE.Vector3(0.1, 0.05, 0) };
let tween = null;
const user = { theta: 0, phi: 0, zoom: 1 };
function flyTo(pos, target, dur = 2.0) {
  user.theta = user.phi = 0;
  user.zoom = 1;
  if (STILL) { view.pos.copy(pos); view.target.copy(target); tween = null; return; }
  tween = { p0: view.pos.clone(), t0: view.target.clone(), p1: pos.clone(), t1: target.clone(), s: performance.now(), dur: dur * 1000 };
  user.theta = user.phi = 0;
  user.zoom = 1;
}
function updateCamera(dt) {
  if (tween) {
    const k = Math.min(1, (performance.now() - tween.s) / tween.dur), e = ease(k);
    view.pos.lerpVectors(tween.p0, tween.p1, e);
    view.target.lerpVectors(tween.t0, tween.t1, e);
    if (k >= 1) tween = null;
  }
  const ch = CHAPTERS[cur];
  if (!tween && ch.drift && !STILL) user.theta += dt * ch.drift;
  const target = innerWidth < 900 ? 0 : ch.shift || 0;
  st.shift += (target - st.shift) * (STILL ? 1 : Math.min(1, dt * 2.2));
  cam.setViewOffset(innerWidth, innerHeight, st.shift * innerWidth, 0, innerWidth, innerHeight);
  const off = view.pos.clone().sub(view.target);
  const sph = new THREE.Spherical().setFromVector3(off);
  const mx = STILL ? 0 : mouse.x, my = STILL ? 0 : mouse.y;
  sph.theta += user.theta + mx * 0.06;
  sph.phi = THREE.MathUtils.clamp(sph.phi + user.phi + my * 0.03, 0.15, 1.45);
  sph.radius *= user.zoom;
  cam.position.copy(view.target).add(new THREE.Vector3().setFromSpherical(sph));
  cam.lookAt(view.target);
}
const mouse = { x: 0, y: 0 };
let drag = null;
canvas.addEventListener("pointerdown", (e) => { drag = { x: e.clientX, y: e.clientY }; canvas.setPointerCapture(e.pointerId); });
canvas.addEventListener("pointermove", (e) => {
  mouse.x = (e.clientX / innerWidth - 0.5) * 2;
  mouse.y = (e.clientY / innerHeight - 0.5) * 2;
  if (!drag) return;
  user.theta -= (e.clientX - drag.x) * 0.005;
  user.phi -= (e.clientY - drag.y) * 0.004;
  drag = { x: e.clientX, y: e.clientY };
});
canvas.addEventListener("pointerup", () => (drag = null));
canvas.addEventListener("wheel", (e) => { e.preventDefault(); user.zoom = THREE.MathUtils.clamp(user.zoom * (1 + e.deltaY * 0.0008), 0.55, 1.6); }, { passive: false });

/* ---------------------------------------------------------------- hotspots */
const hotLayer = $("#hot");
let hots = [];
function setHots(list) {
  hotLayer.innerHTML = list.map((h, i) => `<div class="hs ${h.cls || ""}" data-i="${i}"><i></i><span>${h.label}</span></div>`).join("");
  hots = list.map((h, i) => ({ ...h, el: hotLayer.children[i], v: W(...h.pos) }));
  requestAnimationFrame(() => hots.forEach((h) => h.el.classList.add("show")));
}
const tmpV = new THREE.Vector3();
function updateHots() {
  for (const h of hots) {
    if (h.follow) h.v.copy(h.follow());
    tmpV.copy(h.v).project(cam);
    const vis = tmpV.z < 1;
    h.el.style.left = ((tmpV.x + 1) / 2) * innerWidth + "px";
    h.el.style.top = ((1 - tmpV.y) / 2) * innerHeight + "px";
    h.el.style.visibility = vis ? "" : "hidden";
  }
}

/* ---------------------------------------------------------------- chapters */
const P = D.progress;
const st = { t: 0, playing: false, order: store.get("order", "W"), shift: 0, ghost: 1 };
const STILL = new URLSearchParams(location.search).has("still"); // test: frame without transitions or drift
const CHAPTERS = [
  {
    id: "project", title: "Quality control that learns from its own mistakes", rail: "Project",
    line: "A desktop model of end-of-line inspection at an automotive supplier. The robot picks the part, shows it to the camera, puts it in its bin; every round it learns from its own mistakes.",
    pos: W(-1.0, -0.95, 0.9), target: W(0.12, 0.02, -0.07), drift: 0.02, shift: 0.06, play: true, intro: true,
  },
  {
    id: "cell", title: "Quality control cell", rail: "Cell",
    line: "SO-101, two bins, a six-pocket tray and four AprilTags. Every dimension in the sim comes from the same file as the real cell.",
    pos: W(-0.66, -0.6, 0.5), target: W(0.12, 0, 0.12), drift: 0.03, shift: -0.17, play: true,
    hots: [
      { label: "SO-101", pos: [0.0, 0.0, 0.2] },
      { label: "Tray · 6 pockets", pos: [0.168, 0, 0.03] },
      { label: "OK bin", pos: [0.063, -0.147, 0.06] },
      { label: "REJECT bin", pos: [0.063, 0.147, 0.06] },
      { label: "AprilTag · fix placement", pos: [-0.052, 0.168, 0.01], cls: "warn" },
      { label: "Top camera · 660 mm", pos: [0.3, 0, 0.67] },
    ],
  },
  {
    id: "simulation", title: "One tray, six episodes", rail: "Simulation",
    line: "Scripted expert in MuJoCo: pick, show to the camera, put in the bin. On the right, the arm's two cameras.",
    pos: W(-0.6, -0.42, 0.66), target: W(0.13, 0.02, -0.02), drift: 0, play: true, hud: true, pip: true, shift: 0.07,
  },
  {
    id: "parts", title: "Color is not a defect", rail: "Parts",
    line: "OK if sound and in the work order's color, otherwise REJECT. Defects come in both colors; the model can't memorize color.",
    pos: W(SHOW_CENTER[0] - 0.66, SHOW_CENTER[1] - 0.02, 0.62), target: W(SHOW_CENTER[0], SHOW_CENTER[1], -0.02), drift: 0, shift: -0.17,
  },
  {
    id: "recording", title: "Recording", rail: "Recording",
    line: "Session: 30 episodes with one command. Tray: 6 parts laid in the tray. Episode: one part, pick, show, drop.",
    pos: W(-0.3, 0, 0.78), target: W(0.12, 0, -0.02), drift: 0, op: true, shift: 0.14,
  },
  {
    id: "loop", title: "Iteration", rail: "Iteration",
    line: "Every round, the same four steps: record, train, measure, collect failures. Measurement always uses the same held-out parts.",
    pos: W(-0.95, -1.05, 1.05), target: W(0.14, 0, 0.1), drift: 0.05, shift: 0.14, play: true, loop: true,
  },
];
let cur = Math.max(0, CHAPTERS.findIndex((c) => c.id === location.hash.slice(1)));

$("#rail").innerHTML = CHAPTERS.map((c, i) => `<button data-i="${i}"><span class="n">${String(i).padStart(2, "0")}</span><span class="t">${c.rail}</span></button>`).join("");
$("#rail").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) go(+b.dataset.i); });
addEventListener("keydown", (e) => {
  if (e.target.closest("input, select, textarea") || $("#drawer").classList.contains("open")) return;
  if (CHAPTERS[cur].op && (e.code === "Space" || e.key === "Enter")) { e.preventDefault(); opNext(); return; }
  if (e.key === "ArrowRight") go(cur + 1);
  if (e.key === "ArrowLeft") go(cur - 1);
});

function go(i, instant = false) {
  i = Math.max(0, Math.min(CHAPTERS.length - 1, i));
  const ch = CHAPTERS[i];
  cur = i;
  history.replaceState(null, "", "#" + ch.id);
  $$("#rail button").forEach((b, k) => b.classList.toggle("on", k === i));
  const cap = $("#cap");
  cap.classList.add("out");
  setTimeout(() => {
    $("#cap-no").textContent = `${String(i).padStart(2, "0")} / ${String(CHAPTERS.length - 1).padStart(2, "0")}`;
    $("#cap-title").textContent = ch.title;
    $("#cap-line").textContent = ch.line;
    $("#cap-extra").innerHTML = extraFor(ch);
    bindExtra(ch);
    cap.classList.remove("out");
  }, instant || STILL ? 0 : 420);
  flyTo(ch.pos, ch.target, instant || STILL ? 0.001 : 2.1);
  st.playing = !!ch.play;
  if (!ch.play) { st.t = 0; setFrame(0); }
  showcase.visible = ch.id === "parts";
  partBodies.forEach((b) => (bodies[b].visible = !ch.op));
  opGroup.visible = !!ch.op;
  $("#op").classList.toggle("on", !!ch.op);
  $("#loop").classList.toggle("on", !!ch.loop);
  $(".simhud").classList.toggle("on", !!ch.hud);
  $$(".pipf").forEach((p) => p.classList.toggle("on", !!ch.pip));
  $("#cap").classList.toggle("compact", !!(ch.op || ch.loop));
  document.body.classList.toggle("side", !!(ch.op || ch.loop || ch.intro));
  $("#intro").classList.toggle("on", !!ch.intro);
  $("#cap").classList.toggle("hero", !!ch.intro);
  $("#cap").classList.toggle("withhud", !!ch.hud);
  if (ch.op) opRender();
  else { setBinGlow(null); fly = null; setHots(ch.id === "parts" ? partHots() : ch.hots || []); }
  if (ch.loop) loopRender();
  if (ch.intro) introRender();
}

function extraFor(ch) {
  if (ch.id === "parts") return `<div class="seg" id="order"><button data-v="W" class="${st.order === "W" ? "on" : ""}">Work order · white</button><button data-v="T" class="${st.order === "T" ? "on" : ""}">Work order · orange</button></div>`;
  return "";
}
function bindExtra(ch) {
  if (ch.id !== "parts") return;
  $("#order").onclick = (e) => {
    const b = e.target.closest("button"); if (!b) return;
    st.order = b.dataset.v; store.set("order", st.order);
    $$("#order button").forEach((x) => x.classList.toggle("on", x === b));
    setHots(partHots());
  };
}
function partHots() {
  return showItems.map((s) => {
    const b = decide({ variant: s.variant, defect: s.defect }, st.order);
    return { label: b.toUpperCase(), pos: s.pos, cls: `chip ${b}` };
  });
}

/* ---------------------------------------------------------------- simulation HUD + small cameras */
const hud = $("#hud");
hud.innerHTML = `<div class="simhud"><div class="task"><b id="task">pick the part and show it to the camera</b><em id="oh">[0, 0]</em></div>
  <div class="tl" id="tl" style="grid-template-columns:${S.episodes.map((e) => e.end - e.start + 1).join("fr ")}fr">${S.episodes.map(() => `<div class="tl-seg"><span></span></div>`).join("")}</div>
  <div class="tl-lbl" style="grid-template-columns:${S.episodes.map((e) => e.end - e.start + 1).join("fr ")}fr">${S.episodes.map((e) => `<span>${e.part} → <b class="${e.command}">${e.command.toUpperCase()}</b></span>`).join("")}</div></div>
  <div class="op" id="op"></div><div class="op loop" id="loop"></div><div class="op intro" id="intro"></div>`;
$("#tl").addEventListener("click", (e) => {
  const r = $("#tl").getBoundingClientRect();
  st.t = ((e.clientX - r.left) / r.width) * NF;
});
$("#pip").innerHTML = simCams.map((c) => `<div class="pipf" data-c="${c.userData.name}"><span>${c.userData.name === "top" ? "top camera" : "wrist camera"}</span></div>`).join("");
let lastEp = -1, lastPh = "";
function updateSimHud() {
  const f = Math.floor(st.t) % NF, ep = episodeAt(f), ph = phaseAt(f), e = S.episodes[ep];
  const p2 = ph === "place" || ph === "home";
  if (ph !== lastPh || ep !== lastEp) {
    $("#task").textContent = p2 ? `put the part in the ${e.command.toUpperCase()} bin` : "pick the part and show it to the camera";
    $("#task").classList.toggle("p2", p2);
    $("#oh").textContent = p2 ? (e.command === "ok" ? "[1, 0]" : "[0, 1]") : "[0, 0]";
    $$(".tl-lbl span").forEach((s, i) => s.classList.toggle("on", i === ep));
    lastPh = ph; lastEp = ep;
  }
  $$(".tl-seg span").forEach((s, i) => {
    const E = S.episodes[i];
    s.style.width = (f > E.end ? 100 : f < E.start ? 0 : ((f - E.start) / (E.end - E.start)) * 100) + "%";
  });
  // warm light lights up the part during inspection
  warm.intensity = THREE.MathUtils.lerp(warm.intensity, ph === "inspect" ? 0.9 : 0, 0.08);
  const g = bodies[S.bodies.indexOf("gripper")];
  if (g) warm.position.copy(g.getWorldPosition(tmpV)).add(new THREE.Vector3(-0.1, 0.12, 0));
}
function pipRects() {
  const small = innerWidth < 900;
  const w = small ? 150 : Math.min(320, innerWidth * 0.2), h = (w * 3) / 4, gap = 12;
  const left = innerWidth - w - (small ? 12 : 40);
  const top0 = small ? 74 : innerHeight / 2 - h - gap / 2;
  return simCams.map((c, i) => ({ left, top: top0 + i * (h + gap), w, h }));
}

/* ---------------------------------------------------------------- recording assistant */
const PLAN = "record-W-seed0";
const REC = (() => {
  const R = D.recording[PLAN];
  let k = 0;
  const picks = D.plans[PLAN].trays.flatMap((tr) => tr.picks.map((p, s) => ({ ...p, tray: tr.tray, pockets: tr.pockets, slot: s + 1, index: k++ })));
  const done = new Set(R.recorded);
  const sess = R.sessions.map((trays, i) => {
    const ps = picks.filter((p) => trays.includes(p.tray));
    return { n: i + 1, trays, picks: ps, done: ps.filter((p) => done.has(p.index)).length };
  });
  return { R, picks, done, sess, order: D.plans[PLAN].order };
})();
// same as the terminal flow: one ENTER per tray, three per episode (start, shown, dropped)
function sessionSteps(s) {
  const out = [{ k: "prep" }];
  for (const tray of s.trays) {
    out.push({ k: "tray", tray });
    for (const p of s.picks.filter((q) => q.tray === tray)) for (const sub of [0, 1, 2]) out.push({ k: "ep", p, sub });
  }
  out.push({ k: "end" });
  return out;
}
const SUBS = [
  { t: () => "Return the arm to start", k: "recording starts", btn: "Started" },
  { t: (p) => `Pick from pocket ${p.pocket}, show it to the camera`, k: "command arrives", btn: "Shown" },
  { t: (p) => `Drop into the ${p.command.toUpperCase()} bin, return the arm to start`, k: "episode is saved", btn: "Dropped" },
];
const Z_UP = new THREE.Vector3(0, 0, 1);
const INSPECT = [0.13, 0, 0.12]; // params.yaml inspection.pose: where the part is shown to the camera
const N_EP = REC.picks.length;
const N_TRAY = D.plans[PLAN].trays.length;
const firstOpen = (REC.sess.find((s) => s.done < s.picks.length) || REC.sess[0]).n;
const op = store.get("op4", { sess: firstOpen, at: 0 });
if (new URLSearchParams(location.search).has("at")) op.at = +new URLSearchParams(location.search).get("at"); // test: ?at=7
let fly = null;
function opCur() {
  op.sess = Math.min(Math.max(1, op.sess), REC.sess.length);
  const s = REC.sess[op.sess - 1], steps = sessionSteps(s);
  op.at = Math.max(0, Math.min(steps.length - 1, op.at));
  return { s, steps, step: steps[op.at] };
}
function setBinGlow(k) {
  for (const key of ["ok", "reject"]) for (const m of binMats[key]) {
    m.emissive = new THREE.Color(key === "reject" ? 0xff6a10 : 0xb8c4d8);
    m.userData.glow = key === k;
  }
}
const binSlot = (cmd, j) => { const c = BIN_C[cmd]; return [c[0] + ((j % 2) - 0.5) * 0.042, c[1] + (Math.floor(j / 2) - 1) * 0.036, 0.012]; };
const V = (a) => new THREE.Vector3(...a);
function flight(from, to, lift) {
  const mid = V(from).lerp(V(to), 0.5);
  mid.z = Math.max(from[2], to[2]) + lift;
  return new THREE.QuadraticBezierCurve3(V(from), mid, V(to));
}
function drawArrow(curve) {
  arrow = new THREE.Mesh(new THREE.TubeGeometry(curve, 96, 0.0016, 8), arrowMat);
  arrow.layers.set(1);
  arrow.userData.born = performance.now();
  opGroup.add(arrow);
}
function opScene(step, s) {
  opGroup.children.filter((c) => c !== ring).forEach((c) => opGroup.remove(c));
  arrow = fly = null;
  ring.visible = false;
  setBinGlow(null);
  const tray = step.k === "prep" ? s.trays[0] : step.k === "end" ? s.trays.at(-1) : step.k === "tray" ? step.tray : step.p.tray;
  const trayPicks = s.picks.filter((q) => q.tray === tray);
  const pockets = trayPicks[0].pockets;
  const slot = step.k === "ep" ? step.p.slot : step.k === "end" ? 7 : 0;
  const gone = trayPicks.filter((q) => q.slot < slot);
  if (step.k !== "prep") pockets.forEach((id, k) => {
    if (gone.some((q) => q.pocket === k + 1)) return;
    if (step.k === "ep" && step.sub > 0 && step.p.pocket === k + 1) return; // in the air, see below
    opGroup.add(placeGeneric(id, pocketPos[k], typeQuat[partById[id].type]));
  });
  const inBin = { ok: 0, reject: 0 };
  gone.forEach((q) => opGroup.add(placeGeneric(q.part, binSlot(q.command, inBin[q.command]++), typeQuat[partById[q.part].type])));

  if (step.k === "prep") {
    setHots([{ label: "Tray · 6 pockets", pos: [0.168, 0, 0.04] }, { label: "OK bin", pos: [0.063, -0.147, 0.06] }, { label: "REJECT bin", pos: [0.063, 0.147, 0.06] }]);
  } else if (step.k === "tray") {
    setHots(pockets.map((id, k) => ({ label: `${k + 1} · ${id}`, pos: [pocketPos[k][0], pocketPos[k][1], 0.07], cls: "lbl" })));
  } else if (step.k === "end") {
    setHots([]);
  } else {
    const p = step.p, pp = [...pocketPos[p.pocket - 1]], no = `Episode ${p.index + 1}`;
    pp[2] += 0.012;
    const dest = binSlot(p.command, inBin[p.command]);
    if (step.sub === 0) {
      ring.visible = true;
      ring.position.set(pp[0], pp[1], pocketPos[p.pocket - 1][2] + 0.0035);
      setHots([{ label: no, pos: [pp[0], pp[1], 0.075], cls: "lbl" }]);
    } else {
      const curve = step.sub === 1 ? flight(pp, INSPECT, 0.03) : flight(INSPECT, dest, 0.09);
      drawArrow(curve);
      const mesh = placeGeneric(p.part, step.sub === 1 ? pp : INSPECT, typeQuat[partById[p.part].type]);
      opGroup.add(mesh);
      fly = { mesh, curve, born: performance.now(), dur: 1200, hover: step.sub === 1, base: mesh.quaternion.clone() };
      if (step.sub === 2) setBinGlow(p.command);
      setHots([step.sub === 1
        ? { label: "Show to the top camera", pos: [INSPECT[0], INSPECT[1], INSPECT[2] + 0.05], cls: "lbl" }
        : { label: p.command.toUpperCase(), pos: [BIN_C[p.command][0], BIN_C[p.command][1], 0.08], cls: `chip ${p.command}` }]);
    }
  }
}
/* ---------------------------------------------------------------- recording panel */
// step context: which session, tray and episode; every text in the panel comes from here
function opCtx() {
  const { s, steps, step } = opCur();
  const p = step.k === "ep" ? step.p : null;
  const tray = step.k === "tray" ? step.tray : p ? p.tray : step.k === "prep" ? s.trays[0] : s.trays.at(-1);
  const ps = s.picks.filter((q) => q.tray === tray);
  const gone = new Set(ps.filter((q) => (p ? q.slot < p.slot : step.k === "end")).map((q) => q.pocket));
  const cmd = `uv run flywheel-robot record --session ${op.sess}`;
  const next = REC.sess[op.sess];
  const now = step.k === "ep" ? { n: step.sub, t: SUBS[step.sub].t(p), k: SUBS[step.sub].k, go: SUBS[step.sub].btn }
    : step.k === "tray" ? { t: "Empty the tray, lay the 6 parts in the pockets", k: "tray ready", go: "Laid out" }
    : step.k === "prep" ? { go: "Start" }
    : { go: next ? `Session ${next.n}` : "Go to training" };
  // last episode counted as "passed" in the matrix: everything before the current step in this session
  const passed = p ? p.index : step.k === "tray" ? ps[0].index : step.k === "end" ? s.picks.at(-1).index + 1 : s.picks[0].index;
  return { s, steps, step, p, part: p && partById[p.part], tray, ps, gone, cmd, now, passed,
    nParts: new Set(s.picks.map((q) => q.part)).size, range: `episodes ${s.picks[0].index + 1}–${s.picks.at(-1).index + 1}`,
    trayRange: `episodes ${ps[0].index + 1}–${ps.at(-1).index + 1}` };
}
const atOf = (c, q) => c.steps.findIndex((x) => x.k === "ep" && x.p.index === q.index && x.sub === 0);
const atTray = (c, t) => c.steps.findIndex((x) => x.k === "tray" && x.tray === t);
const ui = {
  sess: () => `<select id="op-sess" aria-label="Session">${REC.sess.map((x) => `<option value="${x.n}"${x.n === op.sess ? " selected" : ""}>Session ${x.n} · ${x.done}/${x.picks.length}</option>`).join("")}</select>`,
  nav: (c) => `<div class="op-nav"><button class="btn" id="op-back" aria-label="Back"${op.at === 0 ? " disabled" : ""}>←</button><button class="btn main" id="op-go"><span>${c.now.go}</span><kbd>ENTER</kbd></button></div>`,
  copy: (c) => `<div class="cmd"><code>${c.cmd}</code><button class="copy" data-copy="cd ~/www/so101-flywheel/python && ${c.cmd}">Copy</button></div>`,
  probe: (c) => (c.p && c.p.probe ? `<span class="probe">deliberately opposite · normally ${decide(c.part, REC.order).toUpperCase()}</span>` : ""),
  // tray map seen from behind the arm: far row 4-5-6 on top, REJECT on the left, OK on the right
  map: (c, codes) => {
    const ids = c.ps[0].pockets;
    return `<div class="tmap${codes ? " codes" : ""}"><div class="tm-bin reject${c.p && c.p.command === "reject" ? " on" : ""}">REJECT</div>
      <div class="tm-grid">${[4, 5, 6, 1, 2, 3].map((n) => `<div class="tm-c${c.p && c.p.pocket === n ? " hl" : ""}${c.gone.has(n) ? " gone" : ""}"><b>${n}</b>${codes ? `<code>${ids[n - 1]}</code>` : ""}</div>`).join("")}</div>
      <div class="tm-bin ok${c.p && c.p.command === "ok" ? " on" : ""}">OK</div></div>`;
  },
  stepper: (c) => `<div class="stepper">${["Start", "Show", "Drop"].map((t, i) => `<div class="${i < c.now.n ? "d" : i === c.now.n ? "c" : ""}"><i></i><span>${t}</span></div>`).join("")}</div>`,
  now: (c) => `<div class="a-now"><span class="lbl">Now</span><b>${c.now.t}</b><span class="k"><kbd>ENTER</kbd> ${c.now.k}</span></div>`,
  chips: (c) => `<div class="chips">${c.ps.map((q) => {
    const cls = REC.done.has(q.index) ? "rec" : c.p && q.index === c.p.index ? "c" : c.gone.has(q.pocket) ? "d" : "";
    return `<button class="chip-ep ${cls}" data-at="${atOf(c, q)}" title="Episode ${q.index + 1} · Pocket ${q.pocket} → ${q.command.toUpperCase()}"><b>${q.index + 1}</b><span class="${q.command}">${q.pocket}→${q.command.toUpperCase()}</span></button>`;
  }).join("")}</div>`,
  trayChips: (c) => `<div class="chips">${c.s.trays.map((t) => {
    const ps = c.s.picks.filter((q) => q.tray === t), n = ps.filter((q) => REC.done.has(q.index)).length;
    return `<button class="chip-ep${n === 6 ? " rec" : ""}${c.step.k === "end" ? " d" : ""}" data-at="${atTray(c, t)}"><b>T${t}</b><span>${ps[0].index + 1}–${ps.at(-1).index + 1}</span></button>`;
  }).join("")}</div>`,
  // progress: three levels, three bars; each slice of a bar is one unit of that level (120 episodes, 20 trays, 4 sessions)
  progress: (c) => {
    const ep = c.p ? c.p.index : c.passed, tray = c.tray - 1, sess = c.s.n - 1;
    const recT = (t) => REC.picks.filter((q) => q.tray === t + 1).every((q) => REC.done.has(q.index));
    const row = (name, n, at, total, rec, unit) => `<div class="pg"><span>${name}</span>
      <div class="pg-bar${total > 40 ? " fine" : ""}" style="--n:${total}">${Array.from({ length: total }, (_, i) => `<i class="${rec(i) ? "rec" : i < at ? "d" : i === at ? "c" : ""}"></i>`).join("")}</div>
      <b>${n}<small> / ${total}</small></b><em>${unit}</em></div>`;
    return `<div class="pgs"><div class="pg-h"><span>Progress</span><span><b>${REC.done.size}</b> recorded</span></div>
      ${row("Episode", Math.min(ep + 1, N_EP), ep, N_EP, (i) => REC.done.has(i), "one part")}
      ${row("Tray", tray + 1, tray, N_TRAY, recT, "6 episodes")}
      ${row("Session", sess + 1, sess, REC.sess.length, (i) => REC.sess[i].done === REC.sess[i].picks.length, "5 trays")}</div>`;
  },
  prepList: (c) => `<ol class="plist2">
    <li><i>1</i><div><b>Set aside ${c.nParts} parts from the box</b><button class="link" id="op-parts">parts list</button></div></li>
    <li><i>2</i><div><b>Return the arm to start, connect both cameras</b><span>top camera 0 · wrist 1</span></div></li>
    <li><i>3</i><div><b>Start recording in the terminal</b>${ui.copy(c)}</div></li></ol>`,
  endList: (c) => `<ol class="plist2">
    <li><i>✓</i><div><b>Dataset is uploaded as private</b><a href="https://huggingface.co/datasets/${REC.R.repo}" target="_blank" rel="noopener">huggingface.co/${REC.R.repo}</a></div></li>
    <li><i>↻</i><div><b>If interrupted, the same command resumes where it left off</b>${ui.copy(c)}</div></li></ol>`,
};
function panelHTML(c) {
  let body;
  if (c.p) {
    body = `<div class="a-hero"><div class="a-tile"><span>Pocket</span><b>${c.p.pocket}</b></div><div class="a-arrow"><i></i></div>
        <div class="a-tile bin ${c.p.command}"><span>Bin · ${c.p.command === "ok" ? "right" : "left"}</span><b>${c.p.command.toUpperCase()}</b></div></div>
      <div class="a-part"><code>${c.p.part}</code><span>${describe(c.part)}</span>${ui.probe(c)}</div>
      ${ui.map(c, false)}${ui.now(c)}${ui.stepper(c)}`;
  } else if (c.step.k === "tray") {
    body = `<div class="a-title"><span>Tray</span><b>${c.tray}<small>/${N_TRAY}</small></b><em>${c.trayRange} · tray ${c.s.trays.indexOf(c.tray) + 1} of the session</em></div>
      ${ui.map(c, true)}${ui.now(c)}`;
  } else {
    body = `<div class="a-title"><span>Session</span><b>${c.s.n}<small>/${REC.sess.length}</small></b><em>${c.step.k === "end" ? "done · " : ""}${c.range}</em></div>
      ${c.step.k === "prep" ? ui.prepList(c) : ui.endList(c)}`;
  }
  const where = c.p ? `<b>${c.p.index + 1}</b> / ${N_EP}` : c.step.k === "tray" ? `tray <b>${c.tray}</b> / ${N_TRAY}` : c.step.k === "prep" ? "prep" : "done";
  const dockH = c.p || c.step.k === "tray" ? `Tray ${c.tray} · ${c.trayRange}` : `Session ${c.s.n} · ${c.s.trays.length} trays`;
  return `<div class="op-glow"></div><div class="op-top">${ui.sess()}<span>${where}</span></div>
    <div class="op-body">${body}</div>
    <div class="op-dock"><div class="dock-h">${dockH}</div>${c.p || c.step.k === "tray" ? ui.chips(c) : ui.trayChips(c)}${ui.nav(c)}${ui.progress(c)}</div>`;
}
let lastKey = "";
function opRender() {
  const c = opCtx();
  store.set("op4", op);
  opScene(c.step, c.s);
  const el = $("#op");
  const key = `${op.sess}:${op.at}`;
  el.innerHTML = panelHTML(c);
  el.dataset.bin = c.p ? c.p.command : "";
  if (key !== lastKey && !STILL) { el.classList.remove("swap"); void el.offsetWidth; el.classList.add("swap"); } // content flows in when the step changes
  lastKey = key;
  $("#op-go").onclick = opNext;
  $("#op-back").onclick = opBack;
  $("#op-sess").onchange = (e) => { op.sess = +e.target.value; op.at = 0; opRender(); };
  $("#op-parts")?.addEventListener("click", () => openDrawer("parts"));
  $$("#op [data-at]").forEach((b) => (b.onclick = () => { if (+b.dataset.at >= 0) { op.at = +b.dataset.at; opRender(); } }));
}
function opNext() {
  const { steps } = opCur();
  if (op.at < steps.length - 1) op.at++;
  else if (op.sess < REC.sess.length) { op.sess++; op.at = 0; }
  else { go(CHAPTERS.findIndex((c) => c.id === "loop")); return; }
  opRender();
}
function opBack() {
  if (op.at > 0) { op.at--; opRender(); }
}

/* ---------------------------------------------------------------- iteration: data flow */
const ITERS = ["v0", "v1", "v2", "v3"];
const trainCmd = (v) => `cd ~/www/so101-flywheel/python
uv run lerobot-train \\
  --dataset.repo_id=${REC.R.repo} --dataset.video_backend=pyav \\
  --policy.type=act --policy.device=mps \\
  --policy.repo_id=${REC.R.repo}-act-${v} --policy.private=true \\
  --output_dir=outputs/train/act-${v} --job_name=act-${v} \\
  --wandb.enable=false`;
const loopSel = { v: 0, open: null };
// each node: who does it (you / automatic), state (done / cur / todo), an honest "not yet" if it isn't ready
function loopSteps(v) {
  const n = REC.done.size, rec = n >= N_EP;
  if (v === 0) return [
    { t: "Episodes", who: "you", d: `${N_EP} episodes via teleop: ${REC.sess.length} sessions × 5 trays × 6 parts. Each part once into the correct bin, once into the opposite one.`, prog: `${n}/${N_EP}`, state: rec ? "done" : "cur", mx: true, act: ["Go to the recording screen", "recording"] },
    { t: "Dataset", who: "automatic", d: `Uploaded as private at the end of each session: ${REC.R.repo}. LeRobot v3, 30 fps, top + wrist camera, joint angles and the [ok, reject] command.`, prog: `${n}/${N_EP}`, state: rec ? "done" : n ? "cur" : "todo", link: `https://huggingface.co/datasets/${REC.R.repo}` },
    { t: "Training", who: "one command", d: "ACT policy. Image + joints + command → a 100-step action chunk. The model is private too: " + `${REC.R.repo}-act-v0.`, state: rec ? "cur" : "todo", cmd: trainCmd("v0") },
    { t: "Evaluation", who: "automatic", d: "16 never-seen parts × 2 commands = 32 trials. Same plan every round; success, command following and grasping are counted separately.", state: "todo", note: "no eval script yet" },
    { t: "Failures", who: "you + verifier", d: "Failed trials are split by type: grasp, wrong bin, ignoring the command. Correction episodes are planned for each type.", state: "todo", note: "not yet" },
  ];
  return [
    { t: "Correction episodes", who: "you", d: `New episodes for the cases v${v - 1} got wrong; same recording screen.`, state: "todo" },
    { t: `Dataset v${v}`, who: "automatic", d: "Old + new episodes, same Hugging Face repo.", state: "todo" },
    { t: "Training", who: "one command", d: `act-v${v}.`, state: "todo" },
    { t: "Evaluation", who: "automatic", d: `Same 32 trials. If it doesn't beat v${v - 1}, the old model stays.`, state: "todo" },
    { t: "Failures", who: "you + verifier", d: v < 3 ? `Remaining failures define v${v + 1}'s correction episodes.` : "Remaining failures go into the final report.", state: "todo" },
  ];
}
function loopRender() {
  const res = Object.fromEntries((P.iterations || []).map((r) => [r.v, r]));
  const steps = loopSteps(loopSel.v);
  if (loopSel.open === null) loopSel.open = Math.max(0, steps.findIndex((s) => s.state === "cur"));
  const atV = ITERS.findIndex((v) => !res[v]?.success);
  const c = opCtx();
  $("#loop").innerHTML = `<div class="op-glow"></div><div class="op-top"><span>Iteration</span><span>goal · +20 points by v3</span></div>
    <div class="op-body"><div class="iters">${ITERS.map((v, i) => `<button data-i="${i}" class="${i === loopSel.v ? "on" : ""} ${i < atV ? "d" : i === atV ? "c" : ""}"><b>${v}</b><span>${res[v]?.success != null ? `${res[v].success}%` : "—"}</span></button>`).join("")}</div>
    <ol class="flow">${steps.map((s, i) => `<li class="st-${s.state}${i === loopSel.open ? " open" : ""}" data-i="${i}">
      <div class="fl-h"><i></i><b>${s.t}</b><span class="who">${s.who}</span>${s.prog ? `<em>${s.prog}</em>` : ""}${s.note ? `<span class="tag">${s.note}</span>` : ""}</div>
      <div class="fl-b"><p>${s.d}</p>${s.mx ? ui.progress(c) : ""}${s.cmd ? cmdBlock(["Command", s.cmd]) : ""}${s.link ? `<a class="link" href="${s.link}" target="_blank" rel="noopener">Open on Hugging Face ↗</a>` : ""}${s.act ? `<button class="btn" data-go="${s.act[1]}">${s.act[0]} →</button>` : ""}</div></li>`).join("")}
      <li class="fl-back"><button data-next="${Math.min(3, loopSel.v + 1)}">↺ ${loopSel.v < 3 ? `v${loopSel.v + 1}: new episodes from failures` : "final report"}</button></li></ol></div>`;
  $$("#loop .iters button").forEach((b) => (b.onclick = () => { loopSel.v = +b.dataset.i; loopSel.open = null; loopRender(); }));
  $$("#loop .fl-h").forEach((h) => (h.onclick = () => { const i = +h.parentNode.dataset.i; loopSel.open = loopSel.open === i ? -1 : i; loopRender(); }));
  $$("#loop [data-go]").forEach((b) => (b.onclick = () => go(CHAPTERS.findIndex((x) => x.id === b.dataset.go))));
  $$("#loop [data-next]").forEach((b) => (b.onclick = () => { loopSel.v = +b.dataset.next; loopSel.open = null; loopRender(); }));
}

/* ---------------------------------------------------------------- intro: what the project is, what it maps to in real life, what it solves */
const NPARTS = D.parts.length, NHELD = D.parts.filter((p) => p.heldout).length;
const RW = (sim, real) => `${sim}<em>→ ${real}</em>`;
// as each section comes into view, the camera and labels turn to it
const INTRO = [
  { k: "What", h: "End-of-line quality control, on a desk",
    body: `<p>Every part coming off a press, casting or injection molding line is checked before shipping: is the hole drilled, is an edge broken, is it the variant on the order? This project builds that station with an open-source SO-101 arm, two cameras and 3D-printed parts; it teaches the robot the task by showing it, not by writing code.</p>
      <div class="in-stats"><div><b>4</b><span>part types</span></div><div><b>2</b><span>color variants</span></div><div><b>2</b><span>defect types</span></div><div><b>${NPARTS}</b><span>parts · ${NHELD} held-out test</span></div></div>`,
    pos: W(-1.0, -0.95, 0.9), target: W(0.12, 0.02, -0.07), hots: [] },
  { k: "In real life", h: "Everything on the desk has a factory counterpart",
    body: `<div class="in-map">${[
      ["Tray, 6 pockets", "End-of-line fixture or conveyor outfeed"],
      ["Rim, gear, brake pad, bracket", "Cast rim, transmission gear, brake pad, sheet-metal bracket"],
      ["Missing hole, broken corner", "Skipped drilling operation, crack or break"],
      ["Work order: white / orange", "Variant from the MES: finish, color, customer"],
      ["OK bin", "Shipping bin"],
      ["REJECT bin", "Scrap or rework"],
      ["Top + wrist camera", "Inspection station cameras"],
      ["SO-101 + leader arm", "Cobot and the operator showing it the motion"],
    ].map(([a, b]) => `<div><span>${a}</span><i></i><b>${b}</b></div>`).join("")}</div>`,
    pos: W(-0.62, -0.62, 0.66), target: W(0.12, 0.02, -0.02),
    hots: [
      { label: RW("Tray", "end-of-line fixture"), pos: [0.168, 0, 0.03], cls: "rw" },
      { label: RW("OK", "shipping"), pos: [0.063, -0.147, 0.06], cls: "rw" },
      { label: RW("REJECT", "scrap / rework"), pos: [0.063, 0.147, 0.06], cls: "rw" },
      { label: RW("Top camera", "inspection station"), pos: [0.3, 0, 0.67], cls: "rw" },
      { label: RW("SO-101", "cobot"), pos: [0.0, 0.0, 0.2], cls: "rw" },
    ] },
  { k: "What it solves", h: "Four old problems, four answers",
    body: `<div class="in-pairs">${[
      ["Visual inspection is tiring; by the end of a shift, defects slip through.", "Every part gets the same attention; every decision is recorded on video and can be reviewed later."],
      ["Programming a robot takes weeks and an expert.", "The operator shows the motion with the leader arm; the model learns from the demos."],
      ["When the variant changes, the station has to be reconfigured.", "The OK/REJECT decision is a rule tied to the work order; a color change doesn't mean retraining the model."],
      ["A model is trained once and its errors pile up.", "Failed trials become new data; every round is measured with the same test."],
    ].map(([a, b]) => `<div><span>${a}</span><b>${b}</b></div>`).join("")}</div>`,
    pos: W(-0.46, -0.52, 0.44), target: W(0.13, 0.02, 0.05), hots: [{ label: "The part is shown to the camera", pos: [0.13, 0, 0.17], cls: "lbl" }] },
  { k: "How it loops", h: "A loop that learns from its own mistakes",
    body: `<div class="in-loop">${[["Show", "operator records 120 episodes"], ["Record", "dataset to Hugging Face as private"], ["Train", "ACT policy"], ["Measure", "32 trials on never-seen parts"], ["Fix", "failures become new episodes"]]
      .map(([a, b], i) => `<div><i>${i + 1}</i><b>${a}</b><span>${b}</span></div>`).join("")}<div class="back">↺ next round</div></div>
      <p>Goal: <b>+20 points</b> of success rate in three rounds, with under half an hour of human effort per round. Today the simulation and recording infrastructure are ready; the first recordings on the real arm are starting.</p>`,
    pos: W(-1.05, -1.1, 1.2), target: W(0.14, 0.02, -0.06), hots: [] },
];
let introSec = -1;
function introRender() {
  $("#intro").innerHTML = `<div class="op-glow"></div><div class="op-top"><span>Project</span><span>SO-101 · QC flywheel</span></div>
    <div class="op-body">${INTRO.map((s, i) => `<section class="in-s" data-i="${i}"><span class="in-k">${String(i + 1).padStart(2, "0")} · ${s.k}</span><h3>${s.h}</h3>${s.body}</section>`).join("")}<div class="in-end"></div></div>
    <div class="op-dock"><div class="in-dots">${INTRO.map((s, i) => `<button data-i="${i}" aria-label="${s.k}"></button>`).join("")}</div><button class="btn main" id="intro-go"><span>Tour the cell</span><kbd>→</kbd></button></div>`;
  const body = $("#intro .op-body");
  const io = new IntersectionObserver((es) => es.forEach((e) => e.isIntersecting && introAt(+e.target.dataset.i)), { root: body, rootMargin: "-35% 0px -55% 0px" });
  $$("#intro .in-s").forEach((s) => io.observe(s));
  $$("#intro .in-dots button").forEach((b) => (b.onclick = () => $(`#intro .in-s[data-i="${b.dataset.i}"]`).scrollIntoView({ behavior: "smooth", block: "start" })));
  $("#intro-go").onclick = () => go(1);
  introSec = -1;
  const sec = +new URLSearchParams(location.search).get("sec") || 0; // test: ?sec=2
  if (sec) $(`#intro .in-s[data-i="${sec}"]`).scrollIntoView({ block: "start" });
  introAt(sec);
}
function introAt(i) {
  if (i === introSec || CHAPTERS[cur].id !== "project") return;
  introSec = i;
  const s = INTRO[i];
  flyTo(s.pos, s.target, 2.4);
  setHots(s.hots);
  $$("#intro .in-s").forEach((x, k) => x.classList.toggle("on", k === i));
  $("#cap").classList.toggle("compact", i > 0); // once reading starts, the left title shrinks and the scene opens up
  $$("#intro .in-dots button").forEach((x, k) => x.classList.toggle("on", k === i));
}

/* ---------------------------------------------------------------- drawer: commands */
const CAL = "/Users/furkanportakal/www/so101-flywheel/hardware/calibration", FP = "/dev/cu.usbmodem5AB01794851", LP = "/dev/cu.usbmodem5A7C1232561";
const CMDS = [
  ["Recording", "Session by session: each command is 30 episodes, private Hugging Face push when done.", [
    ["Session status", "cd ~/www/so101-flywheel/python\nuv run flywheel-robot sessions"],
    ["Session 1", "uv run flywheel-robot record --session 1          # top camera 0, wrist 1"],
    ["Trial (not pushed to the Hub)", "uv run flywheel-robot record --session 1 --no-push --repo-id fport/so101-qc-test"],
    ["No hardware", "uv run flywheel-robot record --session 1 --fake"],
    ["Camera indices", "lerobot-find-cameras opencv"],
  ]],
  ["Training", "After recording, ACT v0. mps on the Mac; --policy.device=cuda on Colab/RunPod.", [
    ["ACT v0", trainCmd("v0")],
  ]],
  ["Calibration and teleop", "c + ENTER at the first prompt. Wrists at the same angle on both arms.", [
    ["Both at once", "uv run flywheel-robot calibrate both"],
    ["Teleop", "uv run flywheel-robot teleop --top 0 --wrist 1"],
    ["Raw: follower", `lerobot-calibrate \\\n  --robot.type=so101_follower --robot.port=${FP} \\\n  --robot.id=follower --robot.calibration_dir=${CAL}`],
    ["Raw: leader", `lerobot-calibrate \\\n  --teleop.type=so101_leader --teleop.port=${LP} \\\n  --teleop.id=leader --teleop.calibration_dir=${CAL}`],
  ]],
  ["Simulation", "", [
    ["Scene", "uv run flywheel-sim build\nuv run flywheel-sim view"],
    ["Episode and plan", "uv run flywheel-sim episode --part J-T-06\nuv run flywheel-sim batch --order W"],
  ]],
  ["Setup", "", [
    ["Environment", "cd ~/www/so101-flywheel/python\nuv sync --extra robot\nuv run pytest"],
    ["This site", "cd ~/www/so101-flywheel\npython/.venv/bin/python site/export_scene.py\npython/.venv/bin/python site/build.py\ncd site && npx wrangler deploy"],
  ]],
];
const cmdBlock = ([t, c]) => `<div class="blk"><div class="blk-h"><span>${esc(t)}</span><button class="copy" data-copy="${esc(c)}">Copy</button></div><pre>${esc(c).split("\n").map((l) => l.replace(/(\s)(#\s.*)$/, '$1<span class="c">$2</span>')).join("\n")}</pre></div>`;
function drawerBody() {
  const { s } = opCur();
  const ids = [...new Set(s.picks.map((q) => q.part))];
  const got = new Set(store.get(`got3:${op.sess}`, []));
  const byType = ["rim", "gear", "pad", "bracket"].map((t) => [t, ids.filter((i) => partById[i].type === t).sort()]);
  const cf = D.calibration.follower || {}, cl = D.calibration.leader || {};
  return `<div class="dsec" id="d-parts"><h3>Session ${op.sess} · ${ids.length} parts</h3><p>Click each one as you take it out of the box.</p>
      ${byType.map(([t, a]) => `<div class="ptype">${TYPE_NAME[t]} · ${a.length}</div><div class="plist">${a.map((i) => `<div data-id="${i}" class="${got.has(i) ? "got" : ""}">${i}<i class="${partById[i].variant === "T" ? "t" : ""}"></i></div>`).join("")}</div>`).join("")}</div>
    ${CMDS.map(([h, p, cs]) => `<div class="dsec"><h3>${h}</h3>${p ? `<p>${p}</p>` : ""}${cs.map(cmdBlock).join("")}</div>`).join("")}
    <div class="dsec"><h3>Versions</h3><table class="kv">${Object.entries(D.versions).map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("")}</table></div>
    <div class="dsec"><h3>Calibration</h3><table class="kv">${Object.keys(cf).map((j) => `<tr><td>${j}</td><td>${cf[j].homing_offset} · ${cl[j] ? cl[j].homing_offset : ""}</td></tr>`).join("")}</table></div>
    <div class="dsec"><h3>Log</h3><table class="kv">${P.log.map((l) => `<tr><td>${esc(l.text)}</td><td>${l.date.slice(5)}</td></tr>`).join("")}</table></div>`;
}
function openDrawer(section) {
  $("#drawer-body").innerHTML = drawerBody();
  $("#drawer").classList.add("open");
  $("#drawer").setAttribute("aria-hidden", "false");
  if (section) $(`#d-${section}`)?.scrollIntoView();
  $$(".plist div").forEach((d) => (d.onclick = () => {
    const got = new Set(store.get(`got3:${op.sess}`, []));
    got.has(d.dataset.id) ? got.delete(d.dataset.id) : got.add(d.dataset.id);
    store.set(`got3:${op.sess}`, [...got]);
    d.classList.toggle("got");
  }));
}
const closeDrawer = () => { $("#drawer").classList.remove("open"); $("#drawer").setAttribute("aria-hidden", "true"); };
$("#open-drawer").onclick = () => openDrawer();
$("#close-drawer").onclick = closeDrawer;
$("#drawer").addEventListener("click", (e) => { if (e.target.id === "drawer") closeDrawer(); });
addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });
document.addEventListener("click", async (e) => {
  const b = e.target.closest(".copy"); if (!b) return;
  try { await navigator.clipboard.writeText(b.dataset.copy); } catch (err) {}
  b.textContent = "Copied"; b.classList.add("ok");
  setTimeout(() => { b.textContent = "Copy"; b.classList.remove("ok"); }, 1300);
});
$("#now").textContent = `Now · ${P.now_text || "Place the bins and tags"}`;
$("#now").onclick = () => go(1);

/* ---------------------------------------------------------------- render loop */
function resize() {
  renderer.setSize(innerWidth, innerHeight, false);
  cam.aspect = innerWidth / innerHeight;
  cam.fov = innerWidth < 900 ? 44 : 32;
  cam.updateProjectionMatrix();
}
addEventListener("resize", resize);
resize();

let last = performance.now();
const T_OFF = +new URLSearchParams(location.search).get("t") || 0; // ?t=12: fast-forward playback (test)
function loop(now) {
  const dt = Math.max(0, Math.min(0.05, (now - last) / 1000)); // rAF time can be earlier than performance.now() on the first frame
  last = now;
  const ch = CHAPTERS[cur];
  if (st.playing) { st.t = (st.t + dt * S.fps) % NF; setFrame(st.t); if (ch.hud) updateSimHud(); }
  for (const s of showItems) s.holder.rotation.z = now / 1000 * 0.5 + s.phase;
  if (ring.visible) { const k = 1 + 0.12 * Math.sin(now / 260); ring.scale.set(k, k, 1); ringMat.opacity = 0.65 + 0.3 * Math.sin(now / 260); }
  if (fly) {
    const k = STILL ? 1 : Math.min(1, (now - fly.born) / fly.dur);
    fly.mesh.position.copy(fly.curve.getPoint(ease(k)));
    if (fly.hover && k >= 1) {  // rotates slowly about world z while shown to the camera
      fly.mesh.position.z += 0.004 * Math.sin(now / 400);
      fly.mesh.quaternion.setFromAxisAngle(Z_UP, (now - fly.born - fly.dur) / 1600).multiply(fly.base);
    }
  }
  if (arrow) { const k = Math.min(1, (now - arrow.userData.born) / 900); arrow.geometry.setDrawRange(0, Math.floor(ease(k) * arrow.geometry.index.count / 3) * 3); }
  const glow = 0.35 + 0.35 * Math.sin(now / 320);
  for (const k of ["ok", "reject"]) for (const m of binMats[k]) m.emissiveIntensity = m.userData.glow ? glow : 0;
  const ghost = ch.op ? 0.14 : 1;  // in the recording assistant the arm must not hide the tray
  st.ghost += (ghost - st.ghost) * (STILL ? 1 : Math.min(1, dt * 3));
  for (const m of armMats) { m.transparent = st.ghost < 0.99; m.opacity = st.ghost; m.depthWrite = st.ghost > 0.5; }
  updateCamera(dt);
  updateHots();

  renderer.setViewport(0, 0, innerWidth, innerHeight);
  renderer.setScissorTest(false);
  renderer.render(scene, cam);
  if (ch.pip) {
    const rects = pipRects();
    renderer.setScissorTest(true);
    rects.forEach((r, i) => {
      const y = innerHeight - r.top - r.h;
      renderer.setViewport(r.left, y, r.w, r.h);
      renderer.setScissor(r.left, y, r.w, r.h);
      renderer.render(scene, simCams[i]);
      const el = $$(".pipf")[i];
      Object.assign(el.style, { left: r.left + "px", top: r.top + "px", width: r.w + "px", height: r.h + "px" });
    });
    renderer.setScissorTest(false);
  }
  requestAnimationFrame(loop);
}

// start: glide in from a distance
view.pos.copy(W(-1.8, -1.6, 1.4));
view.target.copy(W(0.1, 0, 0.1));
go(cur, false);
if (T_OFF) { st.t = T_OFF * S.fps; setFrame(st.t); updateSimHud(); }
$("#loader").classList.add("done");
requestAnimationFrame(loop);
