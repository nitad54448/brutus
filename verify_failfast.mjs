// The impurity-path fail-fast is only safe if it can never reject a candidate
// the full computation would have accepted. This runs both paths over random
// error vectors and asserts they agree on every accept/reject decision.

function fullScore(errors, nPeaks, maxImp) {          // what the sort computes
  const e = errors.slice(0, nPeaks).sort((a,b)=>a-b);
  const count = nPeaks - maxImp;
  let s = 0;
  for (let i = 0; i < count; i++) s += e[i];
  return s / count;
}

function withFailFast(errors, nPeaks, maxImp, thresh) {
  const count = nPeaks - maxImp;
  const budget = thresh * count;
  const useBound = maxImp <= 4;
  const top = [0,0,0,0];
  let topSum = 0, sumAll = 0;
  for (let i = 0; i < nPeaks; i++) {
    const norm = errors[i];
    if (useBound) {
      let minI = 0, minV = top[0];
      for (let t = 1; t < maxImp; t++) if (top[t] < minV) { minV = top[t]; minI = t; }
      if (norm > minV) { topSum = topSum - minV + norm; top[minI] = norm; }
      sumAll += norm;
      if ((sumAll - topSum) > budget) return 999;     // early reject
    }
  }
  const avg = fullScore(errors, nPeaks, maxImp);
  return avg > thresh ? 999 : avg;
}

let seed = 987654321;
const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;

let cases = 0, disagree = 0, earlyRejects = 0, monotoneViolations = 0;
for (let trial = 0; trial < 400000; trial++) {
  const nPeaks = 5 + Math.floor(rnd() * 27);          // 5..31
  const maxImp = 1 + Math.floor(rnd() * Math.min(5, nPeaks - 1));
  const thresh = 0.2 + rnd() * 1.5;
  const scale  = [0.05, 0.5, 1.0, 5.0][Math.floor(rnd() * 4)];
  const errors = Array.from({length: nPeaks}, () => rnd() * scale);
  // occasional huge outliers, which is exactly the impurity case
  if (rnd() < 0.4) for (let k = 0; k < maxImp; k++) errors[Math.floor(rnd()*nPeaks)] = rnd() * 200;

  const full = fullScore(errors, nPeaks, maxImp);
  const fullVerdict = full <= thresh;                  // would be ACCEPTED
  const ff = withFailFast(errors, nPeaks, maxImp, thresh);
  const ffVerdict = ff !== 999 && ff <= thresh;
  cases++;
  if (fullVerdict !== ffVerdict) disagree++;
  if (ff === 999 && fullVerdict) {                     // the fatal case
    console.log("FALSE REJECT", {nPeaks, maxImp, thresh, full, errors});
  }
  if (ff === 999 && !fullVerdict) earlyRejects++;

  // the bound itself must be non-decreasing in i
  if (maxImp <= 4) {
    const top = [0,0,0,0]; let topSum = 0, sumAll = 0, prev = -Infinity;
    for (let i = 0; i < nPeaks; i++) {
      const norm = errors[i];
      let minI = 0, minV = top[0];
      for (let t = 1; t < maxImp; t++) if (top[t] < minV) { minV = top[t]; minI = t; }
      if (norm > minV) { topSum = topSum - minV + norm; top[minI] = norm; }
      sumAll += norm;
      const bound = sumAll - topSum;
      if (bound < prev - 1e-9) monotoneViolations++;
      prev = bound;
    }
    // and it must never exceed the true final sum
    const e = errors.slice().sort((a,b)=>a-b);
    let trueSum = 0; for (let i = 0; i < nPeaks - maxImp; i++) trueSum += e[i];
    if (prev > trueSum + 1e-9) { console.log("BOUND EXCEEDS TRUTH", {prev, trueSum}); disagree++; }
  }
}

console.log(`cases tested            : ${cases}`);
console.log(`accept/reject disagreed : ${disagree}`);
console.log(`bound non-monotone      : ${monotoneViolations}`);
console.log(`rejected early (a win)  : ${earlyRejects}  (${(100*earlyRejects/cases).toFixed(1)}% of cases skip the sort)`);
console.log(disagree === 0 && monotoneViolations === 0
  ? "\nPASS: the fail-fast is decision-identical to the full computation."
  : "\nFAIL");
