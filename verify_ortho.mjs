// Exact reachability test for the orthorhombic transpose fix.
//
// For a basis triple T with rows r_i = (h_i^2, k_i^2, l_i^2) and true params
// p = (A,B,C):
//   NEW code solves M_T p = q, so it reaches the peak triple whose q's are the
//        q's of T's own three reflections.
//   OLD code solved M_T^T x = q, and (M_T^T p)_0 = h0^2*A + h1^2*B + h2^2*C,
//        i.e. the q of the TRANSPOSED reflection (h0,h1,h2). So it reaches the
//        peak triple made of the transposed reflections instead.
// Enumerating every basis triple and asking which observed peak triples each
// version can reach is therefore exact -- no FoM, no sampling.

function orthoList() {
  const h = [];
  for (let a = 0; a <= 12; a++) for (let b = 0; b <= 12; b++) for (let c = 0; c <= 12; c++)
    if (!(a === 0 && b === 0 && c === 0)) h.push([a, b, c]);
  h.sort((p, q) => (p[0]**2+p[1]**2+p[2]**2) - (q[0]**2+q[1]**2+q[2]**2));
  return h;
}
function splitSpecial(list) {
  const s = [], r = [];
  for (const x of list) {
    const [h,k,l] = x;
    if ((k===0&&l===0)||(h===0&&l===0)||(h===0&&k===0)) s.push(x); else r.push(x);
  }
  return [...s, ...r];
}

function reach(cell, nBasis, nPeaks, dropFraction = 0) {
  const [a,b,c] = cell, A=1/(a*a), B=1/(b*b), C=1/(c*c);
  const basis = splitSpecial(orthoList()).slice(0, nBasis);
  const sq = basis.map(([h,k,l]) => [h*h,k*k,l*l]);
  const qOf = (s) => s[0]*A + s[1]*B + s[2]*C;

  // Observed peaks: the lowest-q distinct lines of this cell. dropFraction
  // deletes some of them, standing in for weak/absent reflections.
  let lines = [...new Set(basis.map(s => qOf([s[0]*s[0], s[1]*s[1], s[2]*s[2]])))];
  lines = [...new Set(sq.map(qOf))].sort((x,y)=>x-y);
  let obs = lines.slice(0, Math.round(nPeaks / (1 - dropFraction)));
  if (dropFraction > 0) obs = obs.filter((_, i) => (i * 7919) % 100 >= dropFraction * 100);
  obs = obs.slice(0, nPeaks);

  const TOL = 1e-9;
  const qIndex = new Map();
  obs.forEach((q, i) => qIndex.set(Math.round(q / TOL), i));
  const lookup = (q) => {
    const k = Math.round(q / TOL);
    for (let d = -2; d <= 2; d++) if (qIndex.has(k + d)) return qIndex.get(k + d);
    return -1;
  };

  const key = (i,j,k) => { const s=[i,j,k].sort((x,y)=>x-y); return s[0]*10000+s[1]*100+s[2]; };
  const reachNew = new Set(), reachOld = new Set();
  const n = basis.length;

  // An orthorhombic cell with its axes permuted is the SAME cell -- the
  // refinement stage sorts a/b/c, so recovering (B,A,C) counts as a hit.
  // Both versions are therefore scored against all 6 permutations of p.
  const PERMS = [[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]];
  const ps = PERMS.map(pm => [[A,B,C][pm[0]], [A,B,C][pm[1]], [A,B,C][pm[2]]]);
  const qWith = (s, p) => s[0]*p[0] + s[1]*p[1] + s[2]*p[2];

  for (let x = 0; x < n; x++) {
    for (let y = x+1; y < n; y++) {
      for (let z = y+1; z < n; z++) {
        const t0 = [sq[x][0], sq[y][0], sq[z][0]];
        const t1 = [sq[x][1], sq[y][1], sq[z][1]];
        const t2 = [sq[x][2], sq[y][2], sq[z][2]];
        for (const p of ps) {
          // NEW: solving M_T x = q reaches the peak triple made of T's own
          // three reflections.
          const i0 = lookup(qWith(sq[x],p)), i1 = lookup(qWith(sq[y],p)), i2 = lookup(qWith(sq[z],p));
          if (i0 >= 0 && i1 >= 0 && i2 >= 0 && i0!==i1 && i1!==i2 && i0!==i2) reachNew.add(key(i0,i1,i2));
          // OLD: solving M_T^T x = q reaches the peak triple made of the
          // TRANSPOSED reflections instead.
          const j0 = lookup(qWith(t0,p)), j1 = lookup(qWith(t1,p)), j2 = lookup(qWith(t2,p));
          if (j0 >= 0 && j1 >= 0 && j2 >= 0 && j0!==j1 && j1!==j2 && j0!==j2) reachOld.add(key(j0,j1,j2));
        }
      }
    }
  }
  const total = (obs.length*(obs.length-1)*(obs.length-2))/6;
  return { total, oldN: reachOld.size, newN: reachNew.size };
}

console.log("Orthorhombic: how many of the observed peak triples can each version");
console.log("use to recover the true cell? (exact enumeration, no FoM)\n");
console.log("  cell            basis peaks   OLD        NEW");
const cases = [
  [[5.43, 7.11, 11.02], 60,  10, 0],
  [[5.43, 7.11, 11.02], 120, 12, 0],
  [[5.43, 7.11, 11.02], 200, 12, 0],
  [[8.79, 12.31, 15.04], 120, 12, 0],
  [[8.79, 12.31, 15.04], 120, 12, 0.35],   // 35% of the low lines missing
];
for (const [cell, nb, np, drop] of cases) {
  const r = reach(cell, nb, np, drop);
  const tag = drop ? ` (${Math.round(drop*100)}% lines missing)` : "";
  console.log(`  ${cell.join(" x ").padEnd(22)} ${String(nb).padStart(4)} ${String(np).padStart(5)}   ` +
              `${String(r.oldN+"/"+r.total).padStart(9)}  ${String(r.newN+"/"+r.total).padStart(9)}${tag}`);
}
