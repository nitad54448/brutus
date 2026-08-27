// Triclinic factorisation: old (no pivoting) vs new (partial pivoting),
// transliterated from the WGSL, run against the real 123-reflection basis.
// (The orthorhombic comparison lives in verify_ortho.mjs.)

function triList() {
  const h = [];
  for (let a = -5; a <= 5; a++) for (let b = -5; b <= 5; b++) for (let c = 0; c <= 5; c++) {
    if (a === 0 && b === 0 && c === 0) continue;
    if (c === 0 && b < 0) continue;
    if (c === 0 && b === 0 && a <= 0) continue;
    h.push([a, b, c]);
  }
  h.sort((p, q) => (p[0]**2+p[1]**2+p[2]**2) - (q[0]**2+q[1]**2+q[2]**2));
  return h;
}

// ---------- triclinic factorisation: old (no pivot) vs new (partial pivot) ---
function factorOld(M) {
  const U = M.slice(), n = 6, facs = [];
  for (let i=0;i<n;i++) {
    const piv = U[i*n+i];
    if (Math.abs(piv) < 1e-10) return null;
    for (let r=i+1;r<n;r++) {
      const f = U[r*n+i]/piv; facs.push(f);
      for (let c=i;c<n;c++) U[r*n+c] -= f*U[i*n+c];
    }
  }
  return { U, facs, piv: [0,1,2,3,4,5] };
}
function factorNew(M) {
  const U = M.slice(), n = 6, facs = [], piv = new Array(6).fill(0);
  for (let i=0;i<n;i++) {
    let br=i, bv=Math.abs(U[i*n+i]);
    for (let r=i+1;r<n;r++) { const v=Math.abs(U[r*n+i]); if (v>bv) { bv=v; br=r; } }
    if (bv < 1e-10) return null;
    piv[i]=br;
    if (br!==i) for (let c=0;c<n;c++) { const t=U[i*n+c]; U[i*n+c]=U[br*n+c]; U[br*n+c]=t; }
    const p=U[i*n+i];
    for (let r=i+1;r<n;r++) {
      const f=U[r*n+i]/p; facs.push(f);
      for (let c=i;c<n;c++) U[r*n+c] -= f*U[i*n+c];
    }
  }
  return { U, facs, piv };
}
function substitute({U,facs,piv}, b) {
  const v=b.slice(), n=6; let fi=0;
  for (let i=0;i<n;i++) {
    const p=piv[i]; if (p!==i) { const t=v[i]; v[i]=v[p]; v[p]=t; }
    for (let r=i+1;r<n;r++) { v[r]-=facs[fi]*v[i]; fi++; }
  }
  const x=new Array(6).fill(0);
  for (let i=5;i>=0;i--) { let s=v[i]; for (let j=i+1;j<n;j++) s-=U[i*n+j]*x[j]; x[i]=s/U[i*n+i]; }
  return x;
}

function triCoverage(factor, nBasis, samples) {
  const basis = triList().slice(0, nBasis);
  const row = ([h,k,l]) => [h*h,k*k,l*l,k*l,h*l,h*k];
  let nonsing=0, accepted=0, maxres=0;
  let seed = 12345;
  const rnd = () => (seed = (seed*1103515245+12345) & 0x7fffffff) / 0x7fffffff;
  for (let s=0;s<samples;s++) {
    const idx=new Set(); while (idx.size<6) idx.add(Math.floor(rnd()*nBasis));
    const ix=[...idx].sort((a,b)=>a-b);
    const M=[]; for (const i of ix) M.push(...row(basis[i]));
    // reference determinant via full pivoting LU in doubles
    const R=M.slice(); let det=1;
    for (let i=0;i<6;i++){
      let br=i,bv=Math.abs(R[i*6+i]);
      for(let r=i+1;r<6;r++){const v=Math.abs(R[r*6+i]); if(v>bv){bv=v;br=r;}}
      if(bv<1e-9){det=0;break;}
      if(br!==i){for(let c=0;c<6;c++){const t=R[i*6+c];R[i*6+c]=R[br*6+c];R[br*6+c]=t;} det=-det;}
      det*=R[i*6+i];
      for(let r=i+1;r<6;r++){const f=R[r*6+i]/R[i*6+i]; for(let c=i;c<6;c++) R[r*6+c]-=f*R[i*6+c];}
    }
    if (Math.abs(det)<1e-9) continue;
    nonsing++;
    const F=factor(M);
    if (!F) continue;
    accepted++;
    const b=[rnd(),rnd(),rnd(),rnd(),rnd(),rnd()];
    const x=substitute(F,b);
    for (let r=0;r<6;r++){ let acc=0; for(let c=0;c<6;c++) acc+=M[r*6+c]*x[c]; maxres=Math.max(maxres,Math.abs(acc-b[r])); }
  }
  return { nonsing, accepted, pct: (100*accepted/nonsing).toFixed(1), maxres };
}

// ============================== run ==========================================
console.log("TRICLINIC: how much of the search space survives factorisation?");
console.log("  123-reflection basis, 40000 random 6-combinations\n");
for (const [name, fn] of [["OLD (no pivoting)", factorOld], ["NEW (partial pivoting)", factorNew]]) {
  const r = triCoverage(fn, 123, 40000);
  console.log(`  ${name.padEnd(24)} accepted ${r.accepted}/${r.nonsing} non-singular (${r.pct}%), worst residual ${r.maxres.toExponential(2)}`);
}
