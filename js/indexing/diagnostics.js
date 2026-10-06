// js/indexing/diagnostics.js
// Explaining an empty result from the search diagnostics.
//
// Classic script, loaded in order by brutus.html (see the list there); its
// top-level names are shared with the other app scripts.

// ---- GPU run diagnostics -------------------------------------------------
// A run that finds nothing used to say only that. These turn the three
// counters the shaders now keep (see the @binding(4) comment there) into a
// sentence naming the setting to change.
//
// Collected per GPU task and cleared at the start of each indexing run.
let gpuDiagnostics = [];
// Same, for the CPU worker systems (cubic / tetragonal / hexagonal), which
// produced no diagnostics at all before -- so selecting only CPU systems
// guaranteed the bare, unexplained "no solutions" message.
let cpuDiagnostics = [];
// One-line technical summary, for the console.
const describeGpuDiagnostics = (d) => {
    const vol = (d.volMin === null)
        ? 'no candidate passed the axis test'
        : `candidate volumes ${Math.round(d.volMin)}-${Math.round(d.volMax)} A^3 ` +
          `(limits ${Math.round(d.minVolume)}-${Math.round(d.maxVolume)})`;
    return `diagnostics: ${vol}; best candidate held ${d.peaksInBudget}/${d.nPeaksScored} ` +
           `peaks inside the FoM budget (threshold ${d.fomThreshold})`;
};
// The reason a run produced nothing, phrased as something to act on.
//
// This function must ALWAYS return at least one sentence. The previous
// version returned null whenever the diagnostics did not cleanly implicate
// a single setting, and the caller then printed a bare "No valid solutions
// were found." That silence was the single worst failure mode in the app:
// the help promises a reason, and every path that could have produced one
// (no GPU diagnostics at all, a CPU-only run, a diagnostic that matched
// none of the branches below) fell through to nothing. A weaker guess with
// the numbers attached beats no message, because the numbers let the user
// check the guess.
const explainNoSolutions = (gpuDiags, cpuDiags, ctx) => {
    const reasons = [];

    for (const d of (gpuDiags || [])) {
        const label = d.system ? `${d.system}: ` : '';
        if (d.volMin === null || d.volMin === undefined) {
            reasons.push(`${label}no candidate cell had all axes in range — check the peak list and 2θ range`);
        } else if (d.volMin > d.maxVolume) {
            reasons.push(`${label}every candidate cell was larger than Max Volume ` +
                         `(${Math.round(d.maxVolume)} Å³); they ranged ` +
                         `${Math.round(d.volMin)}–${Math.round(d.volMax)} Å³ — raise Max Volume`);
        } else if (d.volMax < d.minVolume) {
            reasons.push(`${label}every candidate cell was smaller than the ${Math.round(d.minVolume)} Å³ floor — ` +
                         `the peaks may be indexing on a subcell; try more peaks`);
        } else if (d.nPeaksScored > 0 && d.peaksInBudget < d.nPeaksScored) {
            reasons.push(`${label}cells of plausible size were found, but the best one matched only ` +
                         `${d.peaksInBudget} of ${d.nPeaksScored} peaks within the error budget — ` +
                         `raise 2θ Error, or raise the GPU FoM threshold`);
        } else {
            // Reached only when cells of plausible size passed the FoM gate
            // yet nothing survived refinement on the CPU side.
            reasons.push(`${label}candidate cells passed the GPU filter but none survived refinement — ` +
                         `raise 2θ Error slightly, or add more peaks`);
        }
    }

    for (const d of (cpuDiags || [])) {
        const label = d.system ? `${d.system}: ` : '';
        const trials = d.trials || 0;
        if (trials === 0) {
            reasons.push(`${label}no trial cells were generated at all — check that peaks are present in the 2θ range`);
        } else if (!d.volPassed) {
            if (d.volumeTooLarge && d.volOverMin) {
                reasons.push(`${label}all ${trials.toLocaleString()} trial cells exceeded Max Volume ` +
                             `(${Math.round(d.max_volume)} Å³; smallest was ${Math.round(d.volOverMin)} Å³) — raise Max Volume`);
            } else {
                reasons.push(`${label}none of the ${trials.toLocaleString()} trial cells had a physically ` +
                             `plausible volume — check the wavelength and the peak list`);
            }
        } else if (!d.refined) {
            reasons.push(`${label}${d.volPassed.toLocaleString()} cells were the right size, but the best one ` +
                         `indexed only ${d.bestIndexed || 0} peaks — raise 2θ Error, or remove spurious peaks`);
        } else if (d.bestM20 !== null && d.bestM20 !== undefined) {
            reasons.push(`${label}the best refined cell reached M(20) = ${d.bestM20.toFixed(2)}, below the ` +
                         `${d.min_m20} threshold — the peak list probably contains impurity or spurious peaks`);
        } else {
            reasons.push(`${label}${trials.toLocaleString()} cells were tested and none passed refinement — ` +
                         `raise 2θ Error, or add more peaks`);
        }
    }

    // Last resort: no diagnostics arrived from anywhere (a crashed worker,
    // an aborted GPU task). Say what the run was actually given, since that
    // is still enough for the user to spot an obviously wrong setting.
    if (!reasons.length) {
        reasons.push(`no diagnostics were returned by the search. Run settings: ` +
                     `${ctx.nPeaks} peaks over ${ctx.tthMin.toFixed(2)}–${ctx.tthMax.toFixed(2)}° 2θ, ` +
                     `λ = ${ctx.lambda}, 2θ error ${ctx.tthError}°, Max Volume ${Math.round(ctx.maxVolume)} Å³, ` +
                     `systems: ${ctx.systems.join(', ') || 'none'}`);
    }

    // Cheap, always-true sanity checks appended as hints. These catch the
    // settings that make a search hopeless before it starts.
    const hints = [];
    if (ctx.nPeaks < 8) {
        hints.push(`only ${ctx.nPeaks} peaks are in the 2θ window — indexing is unreliable below about 15`);
    }
    if (ctx.tthError >= 0.15) {
        hints.push(`2θ Error is ${ctx.tthError}°, wide enough to let false cells outscore the true one`);
    } else if (ctx.tthError <= 0.01) {
        hints.push(`2θ Error is ${ctx.tthError}°, tight enough to reject the true cell on zero-shift alone`);
    }
    if (hints.length) reasons.push('Also: ' + hints.join('; '));

    return reasons;
};
