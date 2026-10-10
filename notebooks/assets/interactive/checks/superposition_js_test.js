// node superposition_js_test.js : runs the figure's <script id="data"> and <script id="core"> headless.
const fs = require('fs');
const html = fs.readFileSync(__dirname + '/../superposition.html', 'utf8');
const grab = id => html.split(`<script id="${id}">`)[1].split('</script>')[0];
const mod = { exports: {} };
new Function('module', grab('data') + '\n' + grab('core') + '\nmodule.exports.TABLE = TABLE;')(mod);
const C = mod.exports;
console.log('== formula values (compare with Python (a)) ==');
console.log('antipodal p=0.1 r=1 W=[1,-1] b=-0.1:', C.antiLoss([0, 0, -0.1, -0.1], 0.1, 1).toFixed(6), '(py 0.005674)');
console.log('antipodal p=0.3 r=0.5:', C.antiLoss([0.1, -0.2, 0.05, -0.2], 0.3, 0.5).toFixed(6), '(py 0.042160)');
console.log('antipodal p=0.05 r=2:', C.antiLoss([-0.1, 0.05, -0.02, 0.03], 0.05, 2).toFixed(6), '(py 0.004212)');
console.log('E3(1,1,0) =', C.E3(1, 1, 0), '(exact 1/6)');
console.log('== live solver vs Python table (sample cells) ==');
for (const [i, j, py] of [[0, 40, 1.755444e-03], [20, 60, 2.047767e-02], [40, 70, 9.145951e-02]]) {
  const p = Math.pow(10, -2 + 2 * j / 80), r = Math.pow(10, -1 + i / 40);
  const R = C.solvePoint(p, r);
  console.log(`r=${r.toFixed(3)} p=${p.toFixed(4)}: JS L_C=${R.L.C.toExponential(6)} py ${py.toExponential(6)} rel ${(R.L.C / py - 1).toExponential(1)}`, R.sol);
}
// full-table agreement of the live solver at off-grid points
let maxRel = 0, cnt = 0, phaseMis = 0;
for (let i = 0; i <= 80; i += 4) for (let j = 0; j <= 80; j += 4) {
  const t = C.tableLC(i, j); const p = Math.pow(10, -2 + 2 * j / 80), r = Math.pow(10, -1 + 2 * i / 80);
  const R = C.solvePoint(p, r);
  if (t && R.L.C) { maxRel = Math.max(maxRel, Math.abs(R.L.C / t.LC - 1)); cnt++; }
  if (R.best !== C.cellPhase(i, j)) phaseMis++;
}
console.log(`live NM vs table: ${cnt} cells, max rel diff ${maxRel.toExponential(2)}; phase mismatches ${phaseMis}`);
const tr = C.TABLE.trained; console.log('trained agreement', tr.filter(t => t[2] === t[3]).length + '/' + tr.length);
console.log('== panel 2 training in JS (compare Python (e): S=0 loss .1829, S=0.8 ~.060-.068 square, S=0.95 .0073 pentagon, S=0.99 .0008) ==');
for (const S of [0, 0.8, 0.95, 0.99]) for (const seed of [1, 2]) {
  const t0 = Date.now(); const M = C.trainAll(5, S, 0.9, seed);
  const n = 5, on = []; const nr = [];
  for (let i = 0; i < n; i++) { const L = Math.hypot(M.W[i], M.W[n + i]); nr.push(L.toFixed(2)); if (L > 0.5) on.push(Math.atan2(M.W[n + i], M.W[i]) * 180 / Math.PI); }
  on.sort((a, b) => a - b); const gaps = on.map((a, k) => ((k + 1 < on.length ? on[k + 1] : on[0] + 360) - a).toFixed(0));
  // MC loss with fresh data
  let L = 0; const N = 200000; let a = 987654321 + seed; const rnd = () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  for (let k = 0; k < N; k++) { const x = []; for (let i = 0; i < n; i++) x.push(rnd() < 1 - S ? rnd() : 0);
    const h = [0, 1].map(q => x.reduce((a, v, j) => a + M.W[q * n + j] * v, 0));
    for (let i = 0; i < n; i++) { const y = Math.max(0, M.b[i] + M.W[i] * h[0] + M.W[n + i] * h[1]); L += Math.pow(0.9, i) * (y - x[i]) ** 2; } }
  console.log(`S=${S} seed${seed}: loss ${(L / N).toFixed(5)}, norms [${nr}], gaps [${gaps}], ${Date.now() - t0} ms`);
}
