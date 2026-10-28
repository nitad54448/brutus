// js/dialogs/convert.js
// Solution -> Convert: carry the selected cell to a subgroup or supergroup of its
// space group. Subgroups lists the maximal subgroups (translationengleiche
// and klassengleiche) from the operators; SGConvert relates two groups by a change of
// basis (P, p).
//
// A subgroup of index > 1 in the lattice has a larger cell; its lines are denser than
// the parent's, so the M(20) carried over from the parent is NOT recomputed. The
// converted solution says so. Atoms are not involved: only the cell and the group.
//
// Classic script, loaded after js/chart/manual-space-group.js; shares the global scope.

const convertDlg = document.getElementById('convert-dialog');
const convertEl = id => document.getElementById(id);
const convertState = { sol: null, from: null, mode: 'sub', rows: [], picked: -1, token: 0 };
const CONVERT_SYSTEMS = { cubic: ['cubic'], tetragonal: ['tetragonal'], hexagonal: ['hexagonal', 'trigonal'],
    orthorhombic: ['orthorhombic'], monoclinic: ['monoclinic'], triclinic: ['triclinic'] };
const CENTRING_MULTIPLICITY = { P: 1, A: 2, B: 2, C: 2, I: 2, F: 4, R: 3 };

const convertFlat = m => [m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2]];

// The operators of a database setting as {r: [9], t: [3]}; the list already holds the centring translations.
const convertSettingOps = (setting) => {
    const den = Number(setting.t_den) || 12, rots = spaceGroupData.rotations;
    return setting.ops.map(o => ({ r: rots[o[0]].slice(), t: [o[1] / den, o[2] / den, o[3] / den] }));
};
const convertCell = (sol) => ({ a: sol.a, b: sol.b ?? sol.a, c: sol.c ?? sol.a, alpha: sol.alpha ?? 90, beta: sol.beta ?? 90,
    gamma: sol.gamma ?? (sol.system === 'hexagonal' ? 120 : 90) });
// The crystal system a cell has, to a tolerance: a converted cell carries no label of its own.
const convertClassify = (c) => {
    const L = (x, y) => Math.abs(x - y) / Math.max(x, y) < BRUTUS_LIMITS.conversion.axisRelativeTolerance, A = (x, y) => Math.abs(x - y) < BRUTUS_LIMITS.conversion.angleToleranceDeg;
    const r90 = [c.alpha, c.beta, c.gamma].map(v => A(v, 90));
    if (L(c.a, c.b) && L(c.b, c.c) && r90.every(Boolean)) return 'cubic';
    if (L(c.a, c.b) && r90.every(Boolean)) return 'tetragonal';
    if (L(c.a, c.b) && A(c.alpha, 90) && A(c.beta, 90) && A(c.gamma, 120)) return 'hexagonal';
    if (r90.every(Boolean)) return 'orthorhombic';
    if (A(c.alpha, 90) && A(c.gamma, 90)) return 'monoclinic';
    return 'triclinic';
};
const convertCellText = (c) => `${c.a.toFixed(4)}, ${c.b.toFixed(4)}, ${c.c.toFixed(4)} Å; ${c.alpha.toFixed(3)}, ${c.beta.toFixed(3)}, ${c.gamma.toFixed(3)}°`;
// Integer matrices that express the conventional axes of a centred cell in its primitive axes (the inverse of
// the primitive transform of niggli.js), and that transform itself: what relates groups of different centring.
const convertCentringMatrices = () => {
    const out = [];
    for (const letter of ['A', 'B', 'C', 'I', 'F', 'R']) {
        const prim = convertFlat(primitiveTransformByCentering[letter]);
        const inv = SGConvert.inv3(prim);
        if (inv) { out.push(inv.map(v => Math.round(v * 1e6) / 1e6)); out.push(prim); }
    }
    return out;
};
const convertMatrixCache = new Map();
const convertMatrixList = (extra) => {
    const key = JSON.stringify(extra);
    if (convertMatrixCache.has(key)) return convertMatrixCache.get(key);
    const U = SGConvert.candidates(), list = U.slice();
    for (const K of extra) for (const u of U.slice(0, BRUTUS_LIMITS.conversion.matrixCandidates)) list.push(SGConvert.mul3(u, K));
    convertMatrixCache.set(key, list);
    return list;
};

// ---- pickers --------------------------------------------------------------
const convertFillPicker = (select, entries, filter, keep) => {
    const q = filter.trim().toLowerCase().replace(/\s/g, '');
    const shown = entries.filter(e => !q || (e.label + e.setting.hall).toLowerCase().replace(/\s/g, '').includes(q) || e.setting.setting_id === keep);
    select.replaceChildren(...shown.map(e => new Option(e.label, e.setting.setting_id)));
    if (keep && shown.some(e => e.setting.setting_id === keep)) select.value = keep;
};
const convertEntry = id => manualSG.entries.find(e => e.setting.setting_id === id) || null;
const convertCompatible = () => {
    const allowed = CONVERT_SYSTEMS[convertState.sol.system] || [];
    return manualSG.entries.filter(e => allowed.includes(e.group.crystal_system));
};

function openConvertDialog() {
    if (isIndexing) return;
    const sol = selectedSolution;
    if (!sol) { showStatus('Select a solution first.'); return; }
    if (!spaceGroupData || !manualSG.entries.length) { showStatus('The space-group database is not loaded.', 'error'); return; }
    Object.assign(convertState, { sol, mode: 'sub', rows: [], picked: -1, token: convertState.token + 1 });
    const keep = defaultSimSetting(sol);
    convertEl('convert-from-filter').value = ''; convertEl('convert-to-filter').value = '';
    convertFillPicker(convertEl('convert-from'), convertCompatible(), '', keep);
    convertFillPicker(convertEl('convert-to'), manualSG.entries, '', null);
    convertEl('convert-intro').textContent = `Cell ${convertCellText(convertCell(sol))}. Choose the space group this cell is expressed in; ` +
        `its axes are taken as those of the chosen setting.`;
    convertSetMode('sub');
    convertDlg.showModal();
}
function convertSetMode(mode) {
    convertState.mode = mode;
    document.querySelectorAll('[data-convert-mode]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.convertMode === mode)));
    convertEl('convert-other').hidden = mode !== 'other';
    convertRun();
}

// ---- computation ----------------------------------------------------------
function convertRender() {
    const rows = convertState.rows, out = convertEl('convert-results');
    if (!rows.length) { out.replaceChildren(); return; }
    const head = '<thead><tr><th></th><th>Relation</th><th>Space group</th><th>New axes</th><th>Cell</th><th>V (Å³)</th></tr></thead>';
    out.innerHTML = `<table>${head}<tbody>${rows.map((r, i) => `<tr data-row="${i}" class="${i === convertState.picked ? 'picked' : ''}">` +
        `<td><input type="radio" name="convert-pick" ${i === convertState.picked ? 'checked' : ''} aria-label="Choose"></td>` +
        `<td>${r.relation}${r.copies > 1 ? ` (×${r.copies} equivalent)` : ''}</td><td>${r.name || '<i>not named</i>'}</td><td>${r.basis}</td><td>${convertCellText(r.cell)}</td><td>${getVolumeTriclinic(r.cell).toFixed(1)}</td></tr>`).join('')}</tbody></table>`;
    convertEl('convert-apply').disabled = convertState.picked < 0;
}
async function convertRun() {
    const token = ++convertState.token;
    convertState.rows = []; convertState.picked = -1;
    convertEl('convert-apply').disabled = true; convertRender();
    const from = convertEntry(convertEl('convert-from').value);
    convertState.from = from;
    const status = convertEl('convert-status');
    if (!from) { status.textContent = 'Choose the current space group.'; return; }
    const cell = convertCell(convertState.sol), fromOps = convertSettingOps(from.setting);
    if (convertState.mode === 'sub') await convertSubgroups(token, from, fromOps, cell);
    else await convertOther(token, from, fromOps, cell);
}
const convertYield = () => new Promise(r => setTimeout(r, 0));

// Conjugate subgroups with one name, one index and one cell are listed once, with their count.
function convertMerge() {
    const seen = new Map(), out = [];
    for (const r of convertState.rows) {
        const key = `${r.kind}|${r.index}|${r.entry ? r.entry.setting.hall : '?' + r.cand.nrep}|${[r.cell.a, r.cell.b, r.cell.c, r.cell.alpha, r.cell.beta, r.cell.gamma].map(v => v.toFixed(3)).join(',')}`;
        const first = seen.get(key);
        if (first) { first.copies = (first.copies || 1) + 1; continue; }
        seen.set(key, r); out.push(r);
    }
    convertState.rows = out;
}
async function convertSubgroups(token, from, fromOps, cell) {
    const status = convertEl('convert-status');
    const cands = Subgroups.enumerate(fromOps), seen = new Set();
    if (!cands.length) { status.textContent = `${from.label} has no proper subgroup (it is P1).`; return; }
    const base = [];
    for (const c of cands) {
        const nc = SGConvert.transformCell(cell, c.B), key = `${c.kind}|${c.index}|${[nc.a, nc.b, nc.c, nc.alpha, nc.beta, nc.gamma].map(v => v.toFixed(3)).join(',')}|` +
            c.ops.map(o => o.r.map(Math.round).join('') + o.t.map(v => v.toFixed(3)).join(',')).sort().join(';');
        if (seen.has(key)) continue; seen.add(key);
        base.push({ cand: c, kind: c.kind, index: c.index, relation: `${c.kind === 't' ? 'translationengleiche' : 'klassengleiche'}, index ${c.index}`,
            P: c.B, p: [0, 0, 0], basis: SGConvert.formatBasis(c.B), cell: nc, name: '', entry: null });
    }
    base.sort((x, y) => (x.kind === 't' ? 0 : 1) - (y.kind === 't' ? 0 : 1) || x.index - y.index);
    convertState.rows = base; convertRender();
    const t0 = performance.now();
    const centring = convertCentringMatrices();
    const rotCount = (ops) => new Set(ops.map(o => o.r.join(','))).size;
    for (let i = 0; i < base.length; i++) {
        if (token !== convertState.token) return;
        const row = base[i], c = row.cand;
        status.textContent = `Naming subgroup ${i + 1} of ${base.length}…`;
        for (const T of manualSG.entries) {
            if (token !== convertState.token) return;
            const mT = CENTRING_MULTIPLICITY[String(T.setting.centering || 'P').charAt(0)] || 1;
            if (T.setting.ops.length !== c.nrep * mT) continue;
            const tOps = convertSettingOps(T.setting);
            if (rotCount(tOps) !== c.nrep) continue;
            const letter = String(T.setting.centering || 'P').charAt(0);
            const extra = letter === 'P' ? [] : centring.filter((K, k) => ['A', 'B', 'C', 'I', 'F', 'R'][k >> 1] === letter && k % 2 === 0);
            const found = SGConvert.findTransforms(c.ops, tOps, SGConvert.transformCell(cell, c.B),
                { matrices: convertMatrixList(extra), limit: 1, firstHit: true, maxMisfit: Infinity });
            const hit = found.find(f => f.relation === 'same' || f.relation === 'subgroup');
            if (hit) {
                row.entry = T; row.name = `${T.group.number} ${T.setting.hm || T.setting.symbol}`;
                row.P = SGConvert.mul3(c.B, hit.P); row.p = SGConvert.mulv(c.B, hit.p);
                row.basis = SGConvert.formatBasis(row.P); row.cell = SGConvert.transformCell(cell, row.P);
                break;
            }
            await convertYield();
            if (performance.now() - t0 > BRUTUS_LIMITS.conversion.namingTimeMs) { status.textContent = `Naming stopped after ${BRUTUS_LIMITS.conversion.namingTimeMs / 1000} s; the remaining subgroups are listed without a name.`; convertMerge(); convertRender(); return; }
        }
        convertRender();
    }
    if (token !== convertState.token) return;
    convertMerge(); convertRender();
    status.textContent = `${convertState.rows.length} maximal subgroups of ${from.label}. Choose one and press Add as solution.`;
}

async function convertOther(token, from, fromOps, cell) {
    const status = convertEl('convert-status');
    const to = convertEntry(convertEl('convert-to').value);
    if (!to) { status.textContent = 'Choose the target space group.'; return; }
    status.textContent = 'Searching for a change of axes…';
    await convertYield();
    if (token !== convertState.token) return;
    const found = SGConvert.findTransforms(fromOps, convertSettingOps(to.setting), cell,
        { matrices: convertMatrixList(convertCentringMatrices()), limit: BRUTUS_LIMITS.conversion.resultLimit, maxMisfit: Infinity });
    if (token !== convertState.token) return;
    convertState.rows = found.map(f => ({ relation: f.relation, P: f.P, p: f.p, basis: `${SGConvert.formatBasis(f.P)}; origin ${SGConvert.formatShift(f.p)}`,
        cell: SGConvert.transformCell(cell, f.P), name: `${to.group.number} ${to.setting.hm || to.setting.symbol}`, entry: to }));
    status.textContent = found.length ? `${found.length} change(s) of axes relate ${from.label} and ${to.label}.`
        : `${to.label} is not a subgroup or supergroup of ${from.label} by a change of axes (centring changes and index-2 to 4 sublattices are in the subgroup list).`;
    convertState.picked = found.length ? 0 : -1;
    convertRender();
}

// ---- adding the result ----------------------------------------------------
function convertApply() {
    const row = convertState.rows[convertState.picked], parent = convertState.sol, from = convertState.from;
    if (!row || !parent) return;
    const cell = row.cell, system = convertClassify(cell), child = { ...parent };
    for (const key of ['sgClass', 'sgMembers', 'sgHall', 'sgSettingId', 'sgConditions', 'sgEvidence', 'sgMargin', 'analysis', 'mcPolished', 'mcFrom', 'lattice',
        '_niggli', '_symAlt', '_symAltOf', 'spaceGroup', 'spaceGroups', 'simSG', 'errors']) delete child[key];
    Object.assign(child, { system, a: cell.a, b: cell.b, c: cell.c, alpha: cell.alpha, beta: cell.beta, gamma: cell.gamma,
        _fitContext: `convert:${Date.now()}`,
        convertedFrom: { system: parent.system, a: parent.a, b: parent.b, c: parent.c, alpha: parent.alpha, beta: parent.beta, gamma: parent.gamma,
            group: from.label, relation: row.relation, basis: row.basis } });
    if (system === 'cubic') {
        delete child.b; delete child.c; delete child.alpha; delete child.beta; delete child.gamma;
    } else if (system === 'tetragonal') {
        child.a = 0.5 * (cell.a + cell.b);
        delete child.b; delete child.alpha; delete child.beta; delete child.gamma;
    } else if (system === 'hexagonal') {
        child.a = 0.5 * (cell.a + cell.b);
        delete child.b; delete child.alpha; delete child.beta; child.gamma = 120;
    } else if (system === 'orthorhombic') {
        delete child.alpha; delete child.beta; delete child.gamma;
    } else if (system === 'monoclinic') {
        delete child.alpha; delete child.gamma;
    }
    child.volume = getVolume(child);
    if (row.entry) {
        addSolutionSG(child, { label: row.entry.label, symbol: row.entry.setting.hm || row.entry.setting.symbol, number: Number(row.entry.group.number),
            hall: row.entry.setting.hall, setting_id: row.entry.setting.setting_id, system: row.entry.group.crystal_system,
            source: `converted from ${from.setting.hm || from.setting.symbol}, ${row.relation}` });
    } else {
        addSolutionSG(child, { label: `subgroup (${row.relation}) of ${from.label}`, symbol: '', number: null, hall: '', setting_id: '',
            system, source: 'converted' });
    }
    const at = solutions.indexOf(parent);
    if (at >= 0) solutions.splice(at, 0, child); else solutions.push(child);
    displayedSolutions = solutions.slice(0, MAX_DISPLAYED_SOLUTIONS);
    applySolutionSelection(child);
    updateSolutionsTable();
    document.querySelector('.tab-btn[data-tab="solutions"]').click();
    const opened = ui.solutionsTableBody.querySelector('details.sol-acc[open]');
    if (opened) opened.scrollIntoView({ block: 'nearest' });
    appendWorkspaceLog(`Converted ${from.label} to ${solutionSGs(child)[0].label} (${row.relation}); M(20) is the parent's.`);
    showStatus(`Added the converted cell (${system}). Its M(20) is the parent's: it is not recomputed.`, 'info', 7000);
    convertDlg.close();
}

convertEl('convert-from').addEventListener('change', convertRun);
convertEl('convert-to').addEventListener('change', convertRun);
convertEl('convert-from-filter').addEventListener('input', () => {
    convertFillPicker(convertEl('convert-from'), convertCompatible(), convertEl('convert-from-filter').value, convertEl('convert-from').value);
    convertRun();
});
let convertToTimer = null;
convertEl('convert-to-filter').addEventListener('input', () => {
    convertFillPicker(convertEl('convert-to'), manualSG.entries, convertEl('convert-to-filter').value, convertEl('convert-to').value);
    clearTimeout(convertToTimer);
    convertToTimer = setTimeout(() => { if (convertState.mode === 'other') convertRun(); }, 350);
});
document.querySelectorAll('[data-convert-mode]').forEach(b => b.addEventListener('click', () => convertSetMode(b.dataset.convertMode)));
convertEl('convert-results').addEventListener('click', e => {
    const tr = e.target.closest('tr[data-row]'); if (!tr) return;
    convertState.picked = Number(tr.dataset.row); convertRender();
});
convertEl('convert-apply').addEventListener('click', convertApply);
convertEl('convert-cancel').addEventListener('click', () => { convertState.token++; convertDlg.close(); });
convertDlg.addEventListener('close', () => { convertState.token++; });
