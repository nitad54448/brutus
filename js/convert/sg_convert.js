/* A cell carried to another space group by a change of basis (P, p), as in International Tables A, 1.5:
 *
 *   (a', b', c') = (a, b, c) P        p: the new origin in the old basis
 *   G' = P^T G P
 *   (W', w') = (P^-1 W P,  P^-1 (W p + w - p))
 *
 * P is row-major, its COLUMNS are the new basis vectors in the old basis ("a+b,-a+b,c": columns (1,1,0), (-1,1,0), (0,0,1)).
 * Brutus has a cell and no atoms, so only the basis, the operators and the metric are handled here: finding the changes of
 * basis that relate two groups, and the cell they give. */
(function (root) {
  'use strict';
  var I3 = [1, 0, 0, 0, 1, 0, 0, 0, 1];

  function det3(m) { return m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]); }
  function inv3(m) {
    var d = det3(m);
    if (Math.abs(d) < 1e-12) return null;
    return [(m[4] * m[8] - m[5] * m[7]) / d, (m[2] * m[7] - m[1] * m[8]) / d, (m[1] * m[5] - m[2] * m[4]) / d,
            (m[5] * m[6] - m[3] * m[8]) / d, (m[0] * m[8] - m[2] * m[6]) / d, (m[2] * m[3] - m[0] * m[5]) / d,
            (m[3] * m[7] - m[4] * m[6]) / d, (m[1] * m[6] - m[0] * m[7]) / d, (m[0] * m[4] - m[1] * m[3]) / d];
  }
  function mul3(a, b) { var o = new Array(9); for (var i = 0; i < 3; i++) for (var j = 0; j < 3; j++) o[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j]; return o; }
  function mulv(m, v) { return [m[0] * v[0] + m[1] * v[1] + m[2] * v[2], m[3] * v[0] + m[4] * v[1] + m[5] * v[2], m[6] * v[0] + m[7] * v[1] + m[8] * v[2]]; }
  function transpose(m) { return [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]]; }
  function clean(v) { var r = Math.round(v); return Math.abs(v - r) < 1e-9 ? r : v; }
  function wrap(v) { v -= Math.floor(v); if (v > 1 - 1e-9) v = 0; return Math.abs(v) < 1e-12 ? 0 : v; }
  function isInt(v) { return Math.abs(v - Math.round(v)) < 1e-6; }

  // ---------------------------------------------------------------- metric
  function metric(c) {
    var r = Math.PI / 180, ca = Math.cos(c.alpha * r), cb = Math.cos(c.beta * r), cg = Math.cos(c.gamma * r);
    return [c.a * c.a, c.a * c.b * cg, c.a * c.c * cb, c.a * c.b * cg, c.b * c.b, c.b * c.c * ca, c.a * c.c * cb, c.b * c.c * ca, c.c * c.c];
  }
  function cellFromMetric(G) {
    var a = Math.sqrt(G[0]), b = Math.sqrt(G[4]), c = Math.sqrt(G[8]), d = 180 / Math.PI;
    return { a: a, b: b, c: c, alpha: Math.acos(Math.max(-1, Math.min(1, G[5] / (b * c)))) * d, beta: Math.acos(Math.max(-1, Math.min(1, G[2] / (a * c)))) * d, gamma: Math.acos(Math.max(-1, Math.min(1, G[1] / (a * b)))) * d };
  }
  function transformCell(cell, P) { return cellFromMetric(mul3(mul3(transpose(P), metric(cell)), P)); }

  function fracText(v) {
    var neg = v < 0, a = Math.abs(v);
    for (var d = 1; d <= 24; d++) { var n = Math.round(a * d); if (Math.abs(a * d - n) < 1e-6) return (neg ? '-' : '') + (d === 1 ? String(n) : n + '/' + d); }
    return String(+v.toFixed(6));
  }
  function formatBasis(P) {
    return [0, 1, 2].map(function (j) {
      var s = '';
      for (var i = 0; i < 3; i++) {
        var c = P[i * 3 + j];
        if (Math.abs(c) < 1e-9) continue;
        var t = Math.abs(Math.abs(c) - 1) < 1e-9 ? '' : fracText(Math.abs(c));
        s += (c < 0 ? '-' : s ? '+' : '') + t + 'abc'.charAt(i);
      }
      return s || '0';
    }).join(',');
  }
  function formatShift(p) { return p.map(fracText).join(','); }

  // ---------------------------------------------------------------- operators
  function opToNew(op, P, p, Pi) {
    Pi = Pi || inv3(P);
    var W = op.r, Wn = mul3(mul3(Pi, W), P).map(clean), Wp = mulv(W, p), w = op.t;
    var t = mulv(Pi, [Wp[0] + w[0] - p[0], Wp[1] + w[1] - p[1], Wp[2] + w[2] - p[2]]).map(wrap);
    return { r: Wn, t: t };
  }
  function opToOld(op, P, p, Pi) {
    Pi = Pi || inv3(P);
    var Wo = mul3(mul3(P, op.r), Pi).map(clean), Pw = mulv(P, op.t), Wp = mulv(Wo, p);
    return { r: Wo, t: [Pw[0] + p[0] - Wp[0], Pw[1] + p[1] - Wp[1], Pw[2] + p[2] - Wp[2]].map(wrap) };
  }
  function rotKey(r) { return r.map(function (v) { return Math.round(v); }).join(','); }
  function intRot(r) { return r.every(isInt); }
  // the translations of a group by rotation, for membership tests
  function index(ops) {
    var m = new Map();
    ops.forEach(function (o) { var k = rotKey(o.r); if (!m.has(k)) m.set(k, []); m.get(k).push(o.t); });
    return m;
  }
  function member(op, idx) {
    if (!intRot(op.r)) return false;
    var ts = idx.get(rotKey(op.r));
    if (!ts) return false;
    return ts.some(function (t) { return isInt(op.t[0] - t[0]) && isInt(op.t[1] - t[1]) && isInt(op.t[2] - t[2]); });
  }

  // ---------------------------------------------------------------- finding a transformation
  var unimodular = null;
  function candidates() {
    if (unimodular) return unimodular;
    unimodular = [];
    var v = [-1, 0, 1], m = new Array(9);
    (function rec(i) {
      if (i === 9) { if (Math.abs(det3(m) - 1) < 1e-9) unimodular.push(m.slice()); return; }
      for (var k = 0; k < 3; k++) { m[i] = v[k]; rec(i + 1); }
    })(0);
    unimodular.sort(function (a, b) { return offDiagonal(a) - offDiagonal(b); });
    return unimodular;
  }
  function offDiagonal(P) { var n = 0; for (var i = 0; i < 9; i++) if (Math.abs(P[i] - I3[i]) > 1e-9) n++; return n; }
  function metricMisfit(cell, normalise) {
    if (!normalise) return 0;
    var n = normalise(cell);
    if (!n) return Infinity;
    return Math.max(Math.abs(n.a - cell.a) / cell.a, Math.abs(n.b - cell.b) / cell.b, Math.abs(n.c - cell.c) / cell.c,
                    Math.abs(n.alpha - cell.alpha) / 90, Math.abs(n.beta - cell.beta) / 90, Math.abs(n.gamma - cell.gamma) / 90);
  }
  var shiftCache = null;
  function shiftList() {
    if (shiftCache) return shiftCache;
    shiftCache = [];
    // eighths and twelfths, which the index-3 klassengleiche subgroups need
    var seen = {};
    [8, 12].forEach(function (n) {
      for (var i = 0; i < n; i++) for (var j = 0; j < n; j++) for (var k = 0; k < n; k++) {
        var key = [i / n, j / n, k / n].map(function (v) { return Math.round(v * 24); }).join(',');
        if (!seen[key]) { seen[key] = 1; shiftCache.push([i / n, j / n, k / n]); }
      }
    });
    shiftCache.sort(function (a, b) { return shiftSize(a) - shiftSize(b); });
    return shiftCache;
  }
  function findTransforms(fromOps, toOps, cell, opts) {
    opts = opts || {};
    var maxMisfit = opts.maxMisfit != null ? opts.maxMisfit : 0.03, limit = opts.limit || 8, out = [];
    var oldIdx = index(fromOps), newIdx = index(toOps), oldRot = new Set(fromOps.map(function (o) { return rotKey(o.r); })), newRot = new Set(toOps.map(function (o) { return rotKey(o.r); }));
    var shifts = shiftList();
    (opts.matrices || candidates()).forEach(function (P) {
      if (opts.firstHit && out.length) return;
      var Pi = inv3(P);
      if (!Pi) return;
      Pi = Pi.map(clean);
      var sub = toOps.every(function (o) { var r = mul3(mul3(P, o.r), Pi).map(clean); return intRot(r) && oldRot.has(rotKey(r)); });
      var sup = fromOps.every(function (o) { var r = mul3(mul3(Pi, o.r), P).map(clean); return intRot(r) && newRot.has(rotKey(r)); });
      if (!sub && !sup) return;
      var newCell = transformCell(cell, P), misfit = metricMisfit(newCell, opts.normalise);
      if (!(misfit <= maxMisfit)) return;
      for (var s = 0; s < shifts.length; s++) {
        var p = shifts[s];
        var okSub = sub && toOps.every(function (o) { return member(opToOld(o, P, p, Pi), oldIdx); });
        var okSup = sup && fromOps.every(function (o) { return member(opToNew(o, P, p, Pi), newIdx); });
        if (okSub || okSup) {
          out.push({ P: P, p: p.map(function (v) { return v > 0.5 + 1e-9 ? v - 1 : v; }), relation: okSub && okSup ? 'same' : okSub ? 'subgroup' : 'supergroup', misfit: misfit, changed: offDiagonal(P), shift: shiftSize(p), compact: compactness(newCell), cell: newCell });
          break;
        }
      }
    });
    var rank = { same: 0, subgroup: 1, supergroup: 2 };
    out.sort(function (a, b) { return (Math.round(a.misfit * 1e4) - Math.round(b.misfit * 1e4)) || ((a.changed > 0) - (b.changed > 0)) || (Math.round(a.compact * 1e3) - Math.round(b.compact * 1e3)) || (acute(a.cell) - acute(b.cell)) || (a.changed - b.changed) || (a.shift - b.shift) || (rank[a.relation] - rank[b.relation]); });
    return out.slice(0, limit);
  }
  function shiftSize(p) { return p.reduce(function (s, v) { var w = v - Math.floor(v); return s + Math.min(w, 1 - w); }, 0); }
  function acute(c) { return (c.alpha < 89.999) + (c.beta < 89.999) + (c.gamma < 89.999); }
  function compactness(c) { return (c.a + c.b + c.c) / Math.cbrt(Math.max(1e-9, c.a * c.b * c.c)) + (Math.abs(c.alpha - 90) + Math.abs(c.beta - 90) + Math.abs(c.gamma - 90)) / 90; }

  var api = { inv3: inv3, mul3: mul3, mulv: mulv, transformCell: transformCell, formatBasis: formatBasis, formatShift: formatShift, findTransforms: findTransforms, candidates: candidates };
  root.SGConvert = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
