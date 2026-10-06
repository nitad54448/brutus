// js/dialogs/refine-mc.js
// Refine MC dialog: Monte-Carlo polishing of one cell.
//
// Classic script, loaded in order by brutus.html (see the list there); its
// top-level names are shared with the other app scripts.

// 3c. Action: Refine MC
// Least squares converges to the nearest minimum of the RESIDUAL, but M20 is
// not the residual: a cell a few tenths of a percent away can index more
// lines and score considerably higher. This runs a symmetry-constrained
// stochastic search around the selected cell (and around the next best ones,
// if the user asks for more than one) and keeps only cells that beat their
// parent's M20. Zero is a search dimension whenever "Refine Zero" is on.
let mcRunning = false;
const mcOverlay = document.getElementById('mc-overlay');
const mcMsg = document.getElementById('mc-msg');
const mcApplyBtn = document.getElementById('mc-apply');
const closeMcModal = () => {
    if (mcRunning) return;               // don't close mid-run
    mcOverlay.classList.remove('open');
};
document.getElementById('ctx-mc').addEventListener('click', () => {
    const parent = ctxTargetSolution();
    if (!parent) return;
    if (typeof monteCarloRefineCell !== 'function') {
        showStatus('Monte-Carlo refinement is unavailable (crystallography scripts not loaded).', 'error', 5000);
        return;
    }
    if (!parent.system || typeof MC_NPAR === 'undefined' || !MC_NPAR[parent.system]) {
        showStatus('This solution has no crystal system the MC can constrain.', 'error', 4000);
        return;
    }

    document.getElementById('mc-title').textContent =
        `Refine MC - ${parent.system}, a=${parent.a.toFixed(4)}` +
        (parent.b ? `, b=${parent.b.toFixed(4)}` : '') +
        (parent.c ? `, c=${parent.c.toFixed(4)}` : '');
    mcMsg.textContent = '';
    mcMsg.classList.remove('info');
    ['mc-iterations', 'mc-restarts']
        .forEach(id => document.getElementById(id).classList.remove('invalid'));
    mcApplyBtn.disabled = false;
    mcApplyBtn.textContent = 'Apply';
    mcOverlay.classList.add('open');
    contextMenu.style.display = 'none';
});
// Clear the error highlight as soon as the user starts fixing a field.
['mc-iterations', 'mc-restarts'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('input', () => el.classList.remove('invalid'));
});
document.getElementById('mc-cancel').addEventListener('click', closeMcModal);
mcOverlay.addEventListener('click', (e) => { if (e.target === mcOverlay) closeMcModal(); });
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && mcOverlay.classList.contains('open')) closeMcModal();
});
mcApplyBtn.addEventListener('click', () => {
    if (mcRunning) return;
    const parent = ctxTargetSolution();
    if (!parent) { closeMcModal(); return; }

    // --- read and validate the three parameters -------------------------
    const mcInputIds = ['mc-iterations', 'mc-restarts'];
    mcInputIds.forEach(id => document.getElementById(id).classList.remove('invalid'));
    const readInt = (id, lo, hi, label) => {
        const el = document.getElementById(id);
        const raw = el.value.trim();
        const v = parseInt(raw, 10);
        let error = null;
        if (raw === '' || !isFinite(v)) error = `${label} must be a number.`;
        else if (v < lo || v > hi) error = `${label} must be between ${lo} and ${hi}.`;
        if (error) { el.classList.add('invalid'); return { error }; }
        return { value: v };
    };
    const rIt  = readInt('mc-iterations', 50, 20000, 'Iterations');
    const rRes = readInt('mc-restarts', 1, 20, 'Restarts');
    for (const r of [rIt, rRes]) {
        if (r.error) { mcMsg.classList.remove('info'); mcMsg.textContent = r.error; return; }
    }
    const nIterations = rIt.value, nRestarts = rRes.value;

    const wl = getWavelength();
    const te = getTthError();
    const tMin = parseFloat(ui.tthMinSlider.value);
    const tMax = parseFloat(ui.tthMaxSlider.value);
    const imp = getImpurityPeaks();
    const rz = !!(ui.refineZeroCheckbox && ui.refineZeroCheckbox.checked);
    const pk = pickedPeaks.filter(p => p.tth >= tMin && p.tth <= tMax);
    const minReq = { cubic: 4, tetragonal: 5, hexagonal: 5, orthorhombic: 6, monoclinic: 7, triclinic: 7 }[parent.system] || 6;
    if (pk.length < minReq) {
        mcMsg.classList.remove('info');
        mcMsg.textContent = `Not enough peaks in the 2-theta range (need at least ${minReq} for ${parent.system}).`;
        return;
    }
    if (!isFinite(wl) || wl <= 0) {
        mcMsg.classList.remove('info');
        mcMsg.textContent = 'Invalid wavelength.';
        return;
    }

    // --- build the state object the MC expects --------------------------
    // Same shape refineAndTestSolution uses in the worker, so the tolerances
    // and figures of merit come out identical to the rest of the program.
    let mcState;
    try {
        const sorted = getSortedPeaks(pk, wl);
        const maxTth = Math.max(...pk.map(p => p.tth));
        const d_min = wl / (2 * Math.sin(maxTth * Math.PI / 360));
        if (!isFinite(d_min) || d_min <= 0) throw new Error('bad d_min');
        mcState = {
            q_obs: sorted.q_obs,
            original_indices: sorted.original_indices,
            tth_obs_rad: sorted.tth_obs_rad,
            peaks_sorted_by_q: sorted.peaks_sorted_by_q,
            N_FOR_M20: 20,
            q_max: 1 / (d_min * d_min),
            d_min: d_min,
            foundSolutions: [],
            foundSolutionMap: new Map()
        };
    } catch (err) {
        mcMsg.classList.remove('info');
        mcMsg.textContent = 'Could not prepare peak list: ' + (err.message || 'unknown');
        return;
    }

    const mcData = {
        peaks: pk, wavelength: wl, tth_error: te,
        max_volume: getMaxVolume(),
        impurity_peaks: imp, refineZero: rz
    };

    // --- run --------------------------------------------------------------
    // Operates on the clicked solution only. The call is synchronous and can
    // take a couple of seconds, so the button is disabled and the dialog is
    // held open with a message; one setTimeout yield lets the browser paint
    // that message before the walk blocks the thread.
    mcRunning = true;
    mcApplyBtn.disabled = true;
    mcApplyBtn.textContent = 'Running...';
    mcMsg.classList.add('info');
    mcMsg.textContent = `Refining ${nRestarts} run(s) of ${nIterations} iterations...`;

    setTimeout(() => {
        let res = null, failure = null;
        try {
            res = monteCarloRefineCell(parent, mcData, mcState, {
                iterations: nIterations,
                restarts: nRestarts
            });
        } catch (err) {
            failure = err && err.message ? err.message : 'unknown error';
            console.warn('MC refinement failed:', err);
        }

        mcRunning = false;
        mcApplyBtn.disabled = false;
        mcApplyBtn.textContent = 'Apply';

        if (failure) {
            mcMsg.classList.remove('info');
            mcMsg.textContent = 'Refinement failed: ' + failure;
            return;
        }
        if (!res || !isFinite(res.m20) || res.m20 <= (parent.m20 || 0)) {
            mcMsg.classList.remove('info');
            mcMsg.textContent =
                `No improvement on M20 ${(parent.m20 || 0).toFixed(2)}. ` +
                `Try more iterations or restarts.`;
            return;
        }

        try {
            res.analysis = analyzeSystematicAbsences(res, pk, spaceGroupData, wl, te, tMax, imp, tMin);
        } catch (err) { res.analysis = null; }

        // Place the refined cell immediately above its parent rather than at
        // the end of the list. The two are meant to be read together -- the
        // whole point is comparing the child against the cell it came from --
        // and appending would strand it wherever the ledger happens to end.
        // A higher M20 also means the default ranking would put it above the
        // parent anyway, so this matches where a re-sort would land it.
        const parentIdx = solutions.indexOf(parent);
        if (parentIdx >= 0) solutions.splice(parentIdx, 0, res);
        else solutions.push(res);
        displayedSolutions = [...solutions];
        updateSolutionsTable();
        updateAllMarkers();
        mcOverlay.classList.remove('open');
        showStatus(
            `MC refined: M20 ${(parent.m20 || 0).toFixed(2)} -> ${res.m20.toFixed(2)}`,
            'success', 8000);
    }, 0);
});
