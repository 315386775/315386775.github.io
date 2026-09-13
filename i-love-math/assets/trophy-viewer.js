import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const TAU = Math.PI * 2;
const palette = {
  navy: 0x173b62, blue: 0x3d78c7, cyan: 0x63d7dc, steel: 0x9cb7c9,
  gold: 0xffc857, orange: 0xff784e, green: 0x5cad75, dark: 0x172638,
  pink: 0xf28cbb, violet: 0x8f78e7, white: 0xf4f8ff, glass: 0x77d9ff,
};

let active = null;

function material(color, options = {}) {
  return new THREE.MeshPhysicalMaterial({
    color,
    roughness: options.roughness ?? .38,
    metalness: options.metalness ?? .45,
    clearcoat: options.clearcoat ?? .35,
    clearcoatRoughness: .25,
    transparent: options.transparent ?? false,
    opacity: options.opacity ?? 1,
    transmission: options.transmission ?? 0,
    emissive: options.emissive ?? 0x000000,
    emissiveIntensity: options.emissiveIntensity ?? 0,
    side: options.side ?? THREE.FrontSide,
  });
}

function roundedGeometry(w, h, d, radius = .16) {
  const r = Math.min(radius, w / 3, h / 3);
  const s = new THREE.Shape();
  s.moveTo(-w / 2 + r, -h / 2);
  s.lineTo(w / 2 - r, -h / 2); s.quadraticCurveTo(w / 2, -h / 2, w / 2, -h / 2 + r);
  s.lineTo(w / 2, h / 2 - r); s.quadraticCurveTo(w / 2, h / 2, w / 2 - r, h / 2);
  s.lineTo(-w / 2 + r, h / 2); s.quadraticCurveTo(-w / 2, h / 2, -w / 2, h / 2 - r);
  s.lineTo(-w / 2, -h / 2 + r); s.quadraticCurveTo(-w / 2, -h / 2, -w / 2 + r, -h / 2);
  const g = new THREE.ExtrudeGeometry(s, { depth: d, bevelEnabled: true, bevelSegments: 3, steps: 1, bevelSize: Math.min(.08, r / 2), bevelThickness: Math.min(.08, d / 3), curveSegments: 5 });
  g.center();
  return g;
}

function polygonGeometry(points, depth = .12, bevel = .05) {
  const s = new THREE.Shape();
  points.forEach(([x, y], i) => i ? s.lineTo(x, y) : s.moveTo(x, y));
  s.closePath();
  const g = new THREE.ExtrudeGeometry(s, { depth, bevelEnabled: true, bevelSegments: 2, bevelSize: bevel, bevelThickness: bevel, curveSegments: 4 });
  g.center();
  return g;
}

function createBuilder() {
  const root = new THREE.Group(), exterior = [], interior = [], animated = [];
  const add = (geometry, mat, pos = [0, 0, 0], rot = [0, 0, 0], scale = [1, 1, 1], parent = root, kind = 'exterior') => {
    const mesh = new THREE.Mesh(geometry, mat);
    mesh.position.set(...pos); mesh.rotation.set(...rot); mesh.scale.set(...scale);
    mesh.castShadow = true; mesh.receiveShadow = true; parent.add(mesh);
    (kind === 'interior' ? interior : exterior).push(mesh);
    return mesh;
  };
  const rounded = (size, color, pos, rot, kind, options) => add(roundedGeometry(...size), material(color, options), pos, rot, [1, 1, 1], root, kind);
  const box = (size, color, pos, rot, kind, options) => add(new THREE.BoxGeometry(...size, 2, 2, 2), material(color, options), pos, rot, [1, 1, 1], root, kind);
  const cylinder = (radius, length, color, pos, rot = [0, 0, 0], kind = 'exterior', options = {}, radial = 28) => add(new THREE.CylinderGeometry(radius, radius, length, radial, 2), material(color, options), pos, rot, [1, 1, 1], root, kind);
  const cone = (radius, length, color, pos, rot = [0, 0, 0], kind = 'exterior', options = {}, radial = 28) => add(new THREE.ConeGeometry(radius, length, radial, 2), material(color, options), pos, rot, [1, 1, 1], root, kind);
  const sphere = (radius, color, pos, scale = [1, 1, 1], kind = 'exterior', options = {}) => add(new THREE.SphereGeometry(radius, 40, 24), material(color, options), pos, [0, 0, 0], scale, root, kind);
  return { root, exterior, interior, animated, add, rounded, box, cylinder, cone, sphere };
}

function addGlow(builder, pos, color = palette.orange, size = .22, kind = 'interior') {
  const mesh = builder.sphere(size, color, pos, [1, 1, 1], kind, { metalness: .1, roughness: .15, emissive: color, emissiveIntensity: 2.4 });
  const light = new THREE.PointLight(color, 2.2, 4); light.position.set(...pos); builder.root.add(light);
  builder.animated.push((t) => { const s = 1 + Math.sin(t * 3.5) * .12; mesh.scale.setScalar(s); light.intensity = 1.8 + Math.sin(t * 3.5) * .55; });
  return mesh;
}

function addJet(b, scale = 1, pos = [0, 0, 0]) {
  const g = new THREE.Group(); g.position.set(...pos); g.scale.setScalar(scale); b.root.add(g);
  const fuselage = b.add(new THREE.CapsuleGeometry(.22, 1.45, 5, 12), material(palette.steel, { metalness: .8 }), [0, 0, 0], [0, 0, Math.PI / 2], [1, 1, 1], g);
  b.add(polygonGeometry([[-.65, 0], [.25, -.8], [.55, -.18], [.62, 0], [.55, .18], [.25, .8]], .08, .025), material(palette.blue, { metalness: .72 }), [-.05, -.02, 0], [Math.PI / 2, 0, 0], [1, 1, 1], g);
  b.add(polygonGeometry([[-.45, 0], [.2, -.36], [.35, 0], [.2, .36]], .07, .02), material(palette.navy), [-.55, .08, 0], [Math.PI / 2, 0, 0], [1, 1, 1], g);
  b.add(new THREE.SphereGeometry(.2, 24, 14), material(palette.glass, { transparent: true, opacity: .8, metalness: .15, transmission: .15 }), [.38, .17, 0], [0, 0, 0], [1.35, .65, .7], g);
  return fuselage;
}

function buildJ20(b) {
  b.add(new THREE.CapsuleGeometry(.48, 5.1, 8, 24), material(palette.steel, { metalness: .8 }), [0, .15, 0], [0, 0, Math.PI / 2], [1, .8, 1]);
  b.add(polygonGeometry([[-2.4, 0], [-.55, -2.3], [.85, -1.4], [1.6, -.38], [2.5, 0], [1.6, .38], [.85, 1.4], [-.55, 2.3]], .22, .08), material(palette.blue, { metalness: .78 }), [-.15, 0, 0], [Math.PI / 2, 0, 0]);
  b.add(polygonGeometry([[-.65, 0], [.55, -.95], [.92, 0], [.55, .95]], .16, .04), material(palette.navy), [-2.15, .28, 0], [Math.PI / 2, 0, 0]);
  [-.58, .58].forEach(z => {
    b.cylinder(.28, 1.5, palette.dark, [-1.3, -.2, z], [0, 0, Math.PI / 2]);
    const flame = b.cone(.25, 1.05, palette.orange, [-2.35, -.2, z], [0, 0, Math.PI / 2], 'interior', { emissive: palette.orange, emissiveIntensity: 2, transparent: true, opacity: .85 });
    b.animated.push(t => { flame.scale.x = .8 + Math.sin(t * 8 + z) * .24; });
  });
  b.sphere(.46, palette.glass, [1.28, .48, 0], [1.7, .65, .75], 'exterior', { transparent: true, opacity: .72, transmission: .2, metalness: .15 });
  b.rounded([1.25, .28, .72, .1], palette.cyan, [.8, .18, 0], [0, 0, 0], 'interior', { emissive: palette.cyan, emissiveIntensity: .45 });
  addGlow(b, [-.55, .1, 0], palette.gold, .26);
  return ['座舱与航电', '双发动力舱', '机翼与隐身外形'];
}

function buildCarrier(b) {
  b.add(polygonGeometry([[-4.8, 0], [-3.7, -1.25], [3.7, -1.5], [4.8, -.72], [4.8, .72], [3.7, 1.5], [-3.7, 1.25]], 1.05, .12), material(palette.navy, { metalness: .72 }), [0, -.55, 0], [Math.PI / 2, 0, 0]);
  b.add(polygonGeometry([[-4.65, 0], [-3.6, -1.4], [4.45, -1.45], [4.8, -.72], [4.8, .72], [4.45, 1.45], [-3.6, 1.4]], .2, .04), material(0x536d82, { metalness: .68, roughness: .45 }), [0, .12, 0], [Math.PI / 2, 0, 0]);
  b.box([7.4, .03, .07], palette.white, [.1, .25, 0]);
  b.box([2.7, .04, .045], palette.gold, [1.6, .26, -.72], [0, .24, 0]);
  b.rounded([1.2, 1.25, .85, .12], palette.steel, [1.35, .78, .65]);
  b.rounded([.78, .32, .72, .08], palette.glass, [1.25, 1.15, .62], [0, 0, 0], 'exterior', { transparent: true, opacity: .65, transmission: .2, metalness: .1 });
  const radar = b.box([1.05, .1, .12], palette.cyan, [1.35, 1.72, .65]);
  b.cylinder(.06, .8, palette.steel, [1.35, 1.42, .65]);
  b.animated.push(t => { radar.rotation.y = t * 1.8; });
  [[-2.4, -.8], [-1.5, .72], [-.4, -.75], [.4, .78]].forEach(([x, z], i) => addJet(b, .34, [x, .42, z]));
  b.rounded([5.5, .55, 2.05, .12], palette.orange, [-.3, -.3, 0], [0, 0, 0], 'interior', { emissive: 0x3d210d, emissiveIntensity: .22 });
  for (let i = 0; i < 5; i++) b.box([.68, .12, .38], palette.gold, [-2 + i * .85, -.28, 0], [0, 0, 0], 'interior');
  addGlow(b, [2.45, -.35, 0], palette.cyan, .28);
  return ['飞行甲板与起降线', '舰载机机库', '舰岛指挥中心', '动力舱（示意）'];
}

function buildTank(b) {
  b.rounded([5.2, 1.15, 2.65, .3], palette.green, [0, .25, 0], [0, 0, 0]);
  [-1.42, 1.42].forEach(z => {
    b.rounded([4.8, .78, .48, .2], palette.dark, [-.25, -.35, z]);
    for (let i = 0; i < 6; i++) b.cylinder(.34, .16, palette.steel, [-2 + i * .77, -.35, z], [Math.PI / 2, 0, 0]);
  });
  b.cylinder(1.05, .66, palette.green, [-.25, 1.1, 0], [0, 0, 0]);
  b.rounded([2.2, .68, 1.8, .22], palette.green, [-.15, 1.28, 0]);
  b.cylinder(.15, 4.4, palette.dark, [2.5, 1.35, 0], [0, 0, Math.PI / 2]);
  b.cylinder(.23, .55, palette.steel, [4.65, 1.35, 0], [0, 0, Math.PI / 2]);
  b.rounded([1.25, .58, 1.45, .12], palette.orange, [-1.25, .25, 0], [0, 0, 0], 'interior');
  b.rounded([1.35, .58, 1.45, .12], palette.gold, [1.1, .25, 0], [0, 0, 0], 'interior');
  addGlow(b, [0, 1.15, 0], palette.cyan, .22);
  return ['履带与负重轮', '可旋转炮塔', '乘员舱（示意）', '动力舱（示意）'];
}

function buildRocket(b) {
  b.cylinder(1.05, 5.2, palette.white, [0, 0, 0]);
  b.cone(1.06, 2.3, palette.orange, [0, 3.75, 0]);
  [1.85, -.4, -2.05].forEach(y => b.cylinder(1.11, .16, palette.blue, [0, y, 0]));
  for (let i = 0; i < 4; i++) {
    const a = i * Math.PI / 2;
    const fin = b.add(polygonGeometry([[-.85, 0], [.8, 0], [-.25, 1.1]], .16, .04), material(palette.blue), [Math.cos(a) * .88, -2.15, Math.sin(a) * .88], [Math.PI / 2, 0, -a]);
    fin.rotation.y = -a;
    const nozzle = b.cone(.3, .85, palette.dark, [Math.cos(a) * .48, -3.02, Math.sin(a) * .48], [Math.PI, 0, 0]);
    const flame = b.cone(.24, 1.25, palette.orange, [Math.cos(a) * .48, -4.0, Math.sin(a) * .48], [Math.PI, 0, 0], 'interior', { emissive: palette.orange, emissiveIntensity: 2.2, transparent: true, opacity: .82 });
    b.animated.push(t => { flame.scale.y = .8 + Math.sin(t * 8 + i) * .22; });
  }
  b.cylinder(.73, 1.4, palette.cyan, [0, 1.15, 0], [0, 0, 0], 'interior', { emissive: 0x16434a, emissiveIntensity: .25 });
  b.cylinder(.73, 1.65, palette.gold, [0, -.95, 0], [0, 0, 0], 'interior');
  addGlow(b, [0, 2.35, 0], palette.cyan, .22);
  return ['整流罩与航天器', '上下层燃料舱', '分离连接环', '发动机喷口'];
}

function buildSaturn(b) {
  const planet = b.sphere(2.15, palette.gold, [0, 0, 0], [1, .94, 1], 'exterior', { roughness: .6, metalness: .05 });
  const bands = [2.8, 3.35, 3.9, 4.45];
  bands.forEach((r, i) => b.add(new THREE.TorusGeometry(r, .18 - i * .018, 10, 120), material(i % 2 ? 0xd49b5e : 0xe8d29b, { roughness: .7, metalness: .12, transparent: true, opacity: .92 }), [0, 0, 0], [Math.PI / 2.45, .08, 0]));
  for (let i = 0; i < 7; i++) b.add(new THREE.TorusGeometry(2.18 + i * .014, .022, 5, 90), material(i % 2 ? 0x9b6a3f : 0xf4d685, { metalness: .1 }), [0, -.82 + i * .27, 0], [Math.PI / 2, 0, 0]);
  b.sphere(1.5, palette.cyan, [0, 0, 0], [1, 1, 1], 'interior', { transparent: true, opacity: .72, emissive: 0x123e45, emissiveIntensity: .35 });
  b.sphere(.85, palette.orange, [0, 0, 0], [1, 1, 1], 'interior', { emissive: 0x52200a, emissiveIntensity: .45 });
  addGlow(b, [0, 0, 0], palette.gold, .3);
  b.animated.push(t => { planet.rotation.y = t * .18; });
  return ['气体外层与云带', '壮观冰尘环', '金属氢层（示意）', '岩石核心（示意）'];
}

function buildMoonbase(b) {
  b.cylinder(5.2, .25, 0x657482, [0, -.85, 0], [0, 0, 0], 'exterior', { roughness: .9, metalness: .1 }, 48);
  [[-2, 0, 0], [1.4, .1, .7], [.3, 0, -1.7]].forEach(([x, y, z], i) => {
    b.sphere(1.15 - i * .12, i ? palette.steel : palette.white, [x, y, z], [1, .58, 1], 'exterior', { roughness: .3, metalness: .25 });
    b.cylinder(.07, 1.15, palette.dark, [x, y - .25, z + .72], [Math.PI / 2, 0, 0]);
  });
  [[-3.3, -1.8], [3.0, -1.55]].forEach(([x, z]) => {
    for (let row = 0; row < 2; row++) for (let col = 0; col < 4; col++) b.box([.62, .05, .44], 0x285da5, [x + col * .66, -.25, z + row * .48], [0, 0, 0], 'exterior', { metalness: .6, roughness: .28 });
  });
  const dish = b.add(new THREE.SphereGeometry(.72, 32, 14, 0, TAU, 0, Math.PI / 2), material(palette.white, { side: THREE.DoubleSide }), [2.45, .35, 1.8], [0, 0, -.45]);
  b.animated.push(t => { dish.rotation.y = Math.sin(t * .45) * .65; });
  b.rounded([1.05, .5, 1.05, .14], palette.orange, [-2, -.15, 0], [0, 0, 0], 'interior');
  addGlow(b, [1.4, 0, .7], palette.cyan, .3);
  return ['生活舱与连通管', '太阳能电池阵列', '通信天线', '能源中心（示意）'];
}

function buildCastle(b) {
  b.cylinder(4.7, .3, palette.green, [0, -1.1, 0], [0, 0, 0], 'exterior', { roughness: .85, metalness: .02 }, 48);
  b.rounded([4.1, 2.8, 2.6, .22], palette.white, [0, .1, 0]);
  [[-2, 0, -1.15], [2, 0, -1.15], [-2, 0, 1.15], [2, 0, 1.15]].forEach(([x, y, z], i) => {
    b.cylinder(.78, 3.5, i % 2 ? palette.violet : palette.pink, [x, .35, z], [0, 0, 0], 'exterior', { roughness: .45, metalness: .12 });
    b.cone(1.0, 1.7, palette.gold, [x, 2.9, z]);
    for (let k = 0; k < 6; k++) { const a = k * TAU / 6; b.box([.24, .38, .24], palette.white, [x + Math.cos(a) * .63, 2.0, z + Math.sin(a) * .63]); }
  });
  b.add(polygonGeometry([[-.6, -.8], [-.6, .25], [0, .9], [.6, .25], [.6, -.8]], .16, .05), material(palette.violet), [0, -.2, 1.38], [0, 0, 0]);
  b.rounded([2.2, 1.5, 1.55, .18], palette.orange, [0, .05, 0], [0, 0, 0], 'interior');
  addGlow(b, [0, .25, 0], palette.gold, .36);
  return ['四座星光塔', '中央城堡大厅', '拱形城门', '魔法能量核心'];
}

function capsuleBetween(b, radius, length, color, pos, rot, kind = 'exterior', options = {}) {
  return b.add(new THREE.CapsuleGeometry(radius, length, 6, 18), material(color, options), pos, rot, [1, 1, 1], b.root, kind);
}

function buildUnicorn(b) {
  b.sphere(1.55, palette.white, [0, .35, 0], [1.65, .85, .8]);
  b.sphere(.82, palette.white, [2.05, 1.05, 0], [1.0, .9, .78]);
  capsuleBetween(b, .42, 1.0, palette.white, [1.28, .78, 0], [0, 0, -.75]);
  [[-.9, -.55], [.65, -.55], [-.9, .55], [.65, .55]].forEach(([x, z]) => {
    capsuleBetween(b, .22, 2.1, palette.white, [x, -.95, z], [0, 0, 0]);
    b.cylinder(.25, .28, palette.gold, [x, -2.18, z]);
  });
  b.cone(.18, 1.45, palette.gold, [2.65, 2.1, 0], [0, 0, -.38], 'exterior', { metalness: .7, roughness: .2 });
  [-.35, .05, .45].forEach((y, i) => b.add(new THREE.TorusGeometry(1.05 - i * .13, .16, 8, 32, Math.PI * 1.3), material([palette.pink, palette.violet, palette.cyan][i]), [-1.25, 1.1 + y, -.3], [0, .4, -.75]));
  b.add(new THREE.TorusGeometry(1.6, .17, 8, 36, Math.PI * 1.35), material(palette.pink), [-1.6, .6, -.1], [0, .5, -1.1]);
  b.sphere(.36, palette.cyan, [0, .5, 0], [1, 1, 1], 'interior', { transparent: true, opacity: .9, emissive: palette.cyan, emissiveIntensity: 1.5 });
  addGlow(b, [2.0, 1.08, .45], palette.violet, .13);
  return ['星光独角', '彩虹鬃毛与尾巴', '灵巧四肢', '童话能量核心'];
}

function buildOcean(b) {
  b.cylinder(5.0, .3, 0x2679a1, [0, -1.15, 0], [0, 0, 0], 'exterior', { roughness: .35, metalness: .15, transparent: true, opacity: .82 }, 48);
  b.sphere(3.4, palette.glass, [0, .15, 0], [1, .72, 1], 'exterior', { transparent: true, opacity: .23, transmission: .35, roughness: .12, metalness: .05, side: THREE.DoubleSide });
  [[-1.65, 0, -.7], [1.55, 0, -.5], [0, 0, 1.0]].forEach(([x, y, z], i) => {
    b.cylinder(.62, 2.65 + i * .25, i === 2 ? palette.pink : palette.violet, [x, .05, z]);
    b.cone(.84, 1.35, palette.gold, [x, 2.05 + i * .13, z]);
  });
  for (let i = 0; i < 9; i++) {
    const a = i * TAU / 9, coral = b.cylinder(.08 + i % 2 * .04, .8 + i % 3 * .22, i % 2 ? palette.orange : palette.pink, [Math.cos(a) * 2.8, -.45, Math.sin(a) * 2.8], [0, 0, (i % 3 - 1) * .25]);
    b.animated.push(t => { coral.rotation.z = Math.sin(t * 1.2 + i) * .08; });
  }
  b.rounded([2.25, 1.25, 1.45, .16], palette.orange, [0, -.1, .15], [0, 0, 0], 'interior');
  addGlow(b, [0, -.1, .15], palette.gold, .34);
  return ['透明海洋穹顶', '珊瑚花园', '三座珍珠塔', '海洋生态舱（示意）'];
}

const builders = { j20: buildJ20, carrier: buildCarrier, tank: buildTank, rocket: buildRocket, saturn: buildSaturn, moonbase: buildMoonbase, castle: buildCastle, unicorn: buildUnicorn, ocean: buildOcean };

function buildModel(id) {
  const b = createBuilder();
  const labels = (builders[id] || buildCarrier)(b);
  const box = new THREE.Box3().setFromObject(b.root), center = box.getCenter(new THREE.Vector3());
  b.root.position.sub(center);
  return { ...b, labels };
}

function stars(scene) {
  const count = 650, positions = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    const r = 22 + Math.random() * 35, a = Math.random() * TAU, u = Math.acos(2 * Math.random() - 1);
    positions[i * 3] = Math.sin(u) * Math.cos(a) * r;
    positions[i * 3 + 1] = Math.cos(u) * r;
    positions[i * 3 + 2] = Math.sin(u) * Math.sin(a) * r;
  }
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  scene.add(new THREE.Points(g, new THREE.PointsMaterial({ color: 0xaedcff, size: .08, transparent: true, opacity: .75 })));
}

function disposeActive() {
  if (!active) return;
  cancelAnimationFrame(active.frame); active.observer?.disconnect(); active.controls.dispose(); active.renderer.dispose();
  active.scene.traverse(o => { if (o.geometry) o.geometry.dispose(); if (o.material) (Array.isArray(o.material) ? o.material : [o.material]).forEach(m => m.dispose()); });
  active.container.replaceChildren(); active = null;
}

function mount(container, { id = 'carrier', stage = '3d' } = {}) {
  disposeActive();
  const scene = new THREE.Scene(); scene.background = new THREE.Color(stage === '4d' ? 0x080b22 : 0x08182a); scene.fog = new THREE.FogExp2(scene.background, .018);
  const camera = new THREE.PerspectiveCamera(38, 1, .1, 120); const wide = window.innerWidth > 760; camera.position.set(wide ? 8.5 : 11, wide ? 5.4 : 7, wide ? 10 : 13);
  let renderer;
  try { renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' }); }
  catch (error) { container.innerHTML = '<div class="viewer-error">这台设备暂时无法开启3D加速，请在新版浏览器中再试试。</div>'; return null; }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2)); renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap; renderer.outputColorSpace = THREE.SRGBColorSpace; renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.18;
  container.replaceChildren(renderer.domElement);
  const controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = true; controls.dampingFactor = .06; controls.enablePan = false; controls.minDistance = 6.5; controls.maxDistance = 28; controls.autoRotate = stage === '4d'; controls.autoRotateSpeed = 1.2; controls.target.set(0, .25, 0);
  scene.add(new THREE.HemisphereLight(0xa9ddff, 0x142235, 2.4));
  const key = new THREE.DirectionalLight(0xffffff, 4.5); key.position.set(7, 11, 8); key.castShadow = true; key.shadow.mapSize.set(1024, 1024); scene.add(key);
  const rim = new THREE.DirectionalLight(0x5aa8ff, 3.0); rim.position.set(-8, 4, -7); scene.add(rim);
  const warm = new THREE.PointLight(0xffa85a, 3.0, 20); warm.position.set(0, -1, 5); scene.add(warm);
  const model = buildModel(id); scene.add(model.root);
  model.interior.forEach(mesh => { mesh.visible = false; });
  const platform = new THREE.Mesh(new THREE.CylinderGeometry(5.4, 5.9, .34, 64), material(0x142a44, { metalness: .82, roughness: .28 })); platform.position.y = -3.25; platform.receiveShadow = true; scene.add(platform);
  const ring = new THREE.Mesh(new THREE.TorusGeometry(4.7, .035, 8, 128), material(stage === '4d' ? palette.violet : palette.cyan, { emissive: stage === '4d' ? palette.violet : palette.cyan, emissiveIntensity: 2.6 })); ring.position.y = -3.05; ring.rotation.x = Math.PI / 2; scene.add(ring);
  const grid = new THREE.GridHelper(40, 40, 0x2b6aa1, 0x153553); grid.position.y = -3.04; scene.add(grid); stars(scene);
  const fit = () => { const rect = container.getBoundingClientRect(); if (!rect.width || !rect.height) return; camera.aspect = rect.width / rect.height; camera.updateProjectionMatrix(); renderer.setSize(rect.width, rect.height, false); };
  const observer = new ResizeObserver(fit); observer.observe(container); fit();
  const clock = new THREE.Clock();
  const tick = () => {
    const t = clock.getElapsedTime(); controls.update(); model.animated.forEach(fn => fn(t));
    if (stage === '4d') { ring.rotation.z = t * .22; warm.intensity = 2.7 + Math.sin(t * 1.8) * .5; }
    renderer.render(scene, camera); active.frame = requestAnimationFrame(tick);
  };
  active = { container, scene, camera, renderer, controls, model, observer, frame: 0, stage };
  tick();
  return {
    labels: model.labels,
    setCutaway(enabled) {
      model.exterior.forEach(mesh => {
        const mat = mesh.material;
        if (!mat.userData.viewerOriginal) mat.userData.viewerOriginal = { transparent: mat.transparent, opacity: mat.opacity, depthWrite: mat.depthWrite, wireframe: mat.wireframe };
        if (enabled) { mat.transparent = true; mat.opacity = .12; mat.depthWrite = false; mat.wireframe = true; }
        else Object.assign(mat, mat.userData.viewerOriginal);
        mat.needsUpdate = true;
      });
      model.interior.forEach(mesh => { mesh.visible = enabled || stage === '4d'; });
      model.interior.forEach(mesh => { if (!enabled && stage !== '4d') mesh.visible = false; });
    },
    setAutoRotate(enabled) { controls.autoRotate = enabled; },
    reset() { camera.position.set(wide ? 8.5 : 11, wide ? 5.4 : 7, wide ? 10 : 13); controls.target.set(0, .25, 0); controls.update(); },
    zoom(delta) { camera.position.multiplyScalar(delta > 0 ? .85 : 1.15); controls.update(); },
    dispose: disposeActive,
  };
}

window.Trophy3D = { mount, dispose: disposeActive };
window.dispatchEvent(new Event('trophy3d-ready'));
