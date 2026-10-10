// node checks/spectral_filters_js_check.js : runs the page's math block against Python reference values
const fs = require("fs"), path = require("path");
const html = fs.readFileSync(path.join(__dirname, "..", "spectral_filters.html"), "utf8");
const src = html.match(/<script id="mathlib">([\s\S]*?)<\/script>/)[1];
const m = { exports: {} }; new Function("module", src)(m); const L = m.exports;
const ref = JSON.parse(fs.readFileSync(path.join(__dirname, "spectral_filters_mc.json"))).js_ref;
let worst = 0;
for (const key in ref) {
  const [a, ap, s2, n] = key.split("_").map(Number);
  const S = L.makeSpectrum(a, ap);
  const de = s2 === 0 ? L.ridgeless(S, n, 0) : L.optimalRidge(S, n, s2).E;
  const he = L.heuristic(S, n, s2).E;
  const r1 = Math.abs(de / ref[key].de - 1), r2 = Math.abs(he / ref[key].heur - 1);
  worst = Math.max(worst, r1, r2);
  console.log(key.padEnd(18), "DE js", de.toExponential(4), "py", ref[key].de.toExponential(4), "rel", r1.toExponential(1),
              "| heur js", he.toExponential(4), "py", ref[key].heur.toExponential(4), "rel", r2.toExponential(1));
}
console.log("worst rel err", worst.toExponential(2));
// optimal ridge at sigma2=0 should be <= ridgeless
const S = L.makeSpectrum(1.5, 2.0);
console.log("sigma2=0, n=1000: optimal", L.optimalRidge(S, 1000, 0).E, "ridgeless", L.ridgeless(S, 1000, 0));
// dof checks vs brute sums
for (const [a, lam, t] of [[1.5, 1e-3, 693], [1.1, 1e-4, 1e4]]) {
  const S2 = L.makeSpectrum(a, 2.0); let bR = 0, bG = 0; const P = 2e7;
  for (let k = 1; k <= P; k++) { const x = Math.pow(k, -a); bR += x / (x + lam); bG += -Math.expm1(-x * t); }
  bR += Math.pow(P, 1 - a) / (a - 1) / lam; bG += t * Math.pow(P, 1 - a) / (a - 1);
  console.log(`dof a=${a}: ridge ${L.dofRidge(S2, lam).toFixed(3)} brute ${bR.toFixed(3)}; flow ${L.dofGF(S2, t).toFixed(3)} brute ${bG.toFixed(3)}`);
}
// slopes
for (const [a, ap, s2] of [[1.5, 2, 0], [1.5, 3, 0], [1.5, 5, 0], [2, 3, 0.01]]) {
  const S3 = L.makeSpectrum(a, ap);
  const f = n => (s2 ? L.optimalRidge(S3, n, s2).E : L.optimalRidge(S3, n, 0).E);
  console.log(`slope a=${a} a'=${ap} s2=${s2}: ${(-Math.log10(f(1e5) / f(1e4))).toFixed(3)} predicted ${L.predictedExponent(a, ap, s2).toFixed(3)}`);
}
