// js/report/pdcif-export.js
// pdCIF export of a solution.
//
// Classic script, loaded in order by brutus.html (see the list there); its
// top-level names are shared with the other app scripts.

// 4. Action: Export pdCIF
document.getElementById('ctx-pdcif').addEventListener('click', () => {
    const target = ctxTargetSolution();
    if (target) {
        generatePdCIF(target);
    }
});
const generatePdCIF = (sol) => {
    // Fetch original wavelength based on preset to match the raw data export
    let wl = getWavelength();
    let isDoublet = false;
    let wlKa1, wlKa2, wlRatio;
    
    const presetVal = ui.wavelengthPreset.value;
    if (presetVal !== 'custom') {
        const [element, type] = presetVal.split('_');
        const data = WAVELENGTH_PRESETS[element];
        if (data) {
            if (type === 'avg') {
                isDoublet = true;
                wlKa1 = data.ka1;
                wlKa2 = data.ka2;
                wlRatio = data.ratio;
            } else {
                wl = data.ka1;
            }
        }
    }
    
    // Brutus: 2th_corr = 2th_obs - Z. pdCIF: 2th_calc = 2th_meas + offset.
    const pdCifOffset = -(sol.zero_correction || 0);

    let cif = `data_brutus_solution\n\n`;
    cif += `_audit_creation_method 'Brutus'\n`;
    cif += `_cell_length_a       ${sol.a.toFixed(5)}\n`;
    cif += `_cell_length_b       ${(sol.b || sol.a).toFixed(5)}\n`;
    cif += `_cell_length_c       ${(sol.c || sol.a).toFixed(5)}\n`;
    cif += `_cell_angle_alpha    ${(sol.alpha || 90).toFixed(3)}\n`;
    cif += `_cell_angle_beta     ${(sol.beta || 90).toFixed(3)}\n`;
    cif += `_cell_angle_gamma    ${(sol.gamma || 90).toFixed(3)}\n`;
    cif += `_cell_volume         ${sol.volume.toFixed(3)}\n\n`;
    
    // --- Space group -------------------------------------------------
    //
    // The symmetry tags are written ONLY when a Space Group MC run actually
    // produced a determination. sgClass and sgMembers are attached by the
    // "Add cell refined under ..." handler and by nothing else, so a plain
    // indexed solution has neither and exports with '?' -- which is what
    // CIF means by "not determined". Asserting a space group nobody
    // established would be worse than leaving it blank: the reader has no
    // way to tell a guess from a result.
    //
    // Two further rules, both about not overclaiming:
    //   * _space_group_name_H-M_alt means a real Hermann-Mauguin symbol. An
    //     extinction-class label (P--c, I4(1)--) is not one -- it names a
    //     set of groups powder data cannot separate -- so a class-only
    //     result also gets '?', with the class recorded in a comment.
    //   * Values are quoted and forced to ASCII. CIF 1.1 is an ASCII
    //     format and the class labels carry subscript digits, which is
    //     enough on its own for some parsers to reject the file.
    const toAsciiSg = (t) => String(t == null ? '' : t)
        .replace(/[\u2080-\u2089]/g, d => String(d.charCodeAt(0) - 0x2080))
        .replace(/[\u2212\u2013\u2014]/g, '-')
        .replace(/'/g, '');

    const mcRan = !!(sol.sgClass || (sol.sgMembers && sol.sgMembers.length));
    const decisiveNats = (typeof SG_DECISIVE_NATS === 'number') ? SG_DECISIVE_NATS : 2.3;

    let sgName = null;          // a definite H-M symbol, or null
    let sgNumber = null;
    const notes = [];

    if (!mcRan) {
        notes.push('No space group analysis was run for this cell, so the symmetry');
        notes.push('is left undetermined. The cell parameters above are complete and');
        notes.push('usable on their own; assume P 1 unless you know otherwise.');
    } else {
        if (sol.sgMembers && sol.sgMembers.length > 0) {
            const parts = String(sol.sgMembers[0]).split(' (');
            sgName = toAsciiSg(parts[0]).trim() || null;
            if (parts.length > 1) {
                const n = parseInt(parts[1].replace(')', ''), 10);
                if (Number.isFinite(n) && n >= 1 && n <= 230) sgNumber = n;
            }
        }
        if (sol.sgClass) {
            notes.push(`Space Group MC: extinction class ${toAsciiSg(sol.sgClass)}` +
                       (sol.sgMembers && sol.sgMembers.length > 1
                           ? `, ${sol.sgMembers.length} groups share these absences: ` +
                             sol.sgMembers.map(toAsciiSg).join(', ')
                           : ''));
        }
        if (Number.isFinite(sol.sgMargin)) {
            notes.push(sol.sgMargin >= decisiveNats
                ? `Margin over the runner-up: ${sol.sgMargin.toFixed(1)} nats (decisive).`
                : `NOT DECISIVE: only ${sol.sgMargin.toFixed(1)} nats over the runner-up ` +
                  `(${decisiveNats.toFixed(1)} needed). Treat the symbol as one of several.`);
        }
        if (!sgName) {
            notes.push('Only an extinction class was determined, not a single space');
            notes.push('group, so the symmetry tags are left undetermined.');
        }
    }

    // The Hall symbol, when there is one. This is the tag that names a
    // SETTING rather than a group: IT number 62 covers Pnma, Pmnb, Pbnm,
    // Pcmn, Pmcn and Pnam, and they forbid different reflections in the
    // same cell. Without it a reader has to infer the setting from the H-M
    // string, which fails on whitespace and on symbols that collide across
    // settings. powder5 reads this tag in preference to the H-M string for
    // exactly that reason.
    const sgHall = toAsciiSg(sol.sgHall).trim() || null;

    for (const line of notes) cif += `# ${line}\n`;
    cif += `_space_group_name_H-M_alt       ${sgName ? `'${sgName}'` : '?'}\n`;
    cif += `_space_group_IT_number          ${sgNumber != null ? sgNumber : '?'}\n`;
    if (sgHall) cif += `_space_group_name_Hall          '${sgHall}'\n`;
    cif += `\n`;

    cif += `_pd_calib_2theta_offset         ${pdCifOffset.toFixed(5)}\n\n`;

    if (isDoublet) {
        cif += `loop_\n`;
        cif += `    _diffrn_radiation_wavelength_id\n`;
        cif += `    _diffrn_radiation_wavelength\n`;
        cif += `    _diffrn_radiation_wavelength_wt\n`;
        cif += `  1  ${wlKa1.toFixed(6)}  1.0\n`;
        cif += `  2  ${wlKa2.toFixed(6)}  ${wlRatio.toFixed(5)}\n\n`;
    } else {
        cif += `_diffrn_radiation_wavelength    ${wl.toFixed(6)}\n\n`;
    }
    
    cif += `_diffrn_radiation_probe         x-ray\n\n`;

    cif += `loop_\n`;
    cif += `_pd_meas_2theta_scan\n`;
    cif += `_pd_meas_intensity_total\n`;
    
    // Write raw data (limited to the indexing range)
    const tMin = parseFloat(ui.tthMinSlider.value);
    const tMax = parseFloat(ui.tthMaxSlider.value);
    const tth = fullExperimentalData.tth;
    const int = fullExperimentalData.intensity;
    for (let i = 0; i < tth.length; i++) {
        if (tth[i] >= tMin && tth[i] <= tMax) {
            cif += `${tth[i].toFixed(5)} ${int[i]}\n`;
        }
    }

    // Trigger download
    const blob = new Blob([cif], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${baseNameNoExt(loadedFileName) || 'pattern'}_sol.CIF`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 1000);

    // Say what the file carries. A pdCIF with no symmetry is perfectly
    // valid -- the cell is complete -- but some refinement programs build
    // their lattice inputs FROM the space group, so an absent one can leave
    // them nowhere to put the cell. Better heard here than puzzled over
    // somewhere else.
    if (!mcRan) {
        showStatus("pdCIF exported with no space group: none was determined for this cell. " +
                   "Run Space Group MC first if you need one. The cell is complete — if the " +
                   "receiving program requires a space group, start from P 1.", 'info', 9000);
    } else if (!sgName) {
        showStatus(`pdCIF exported with no space group: only the extinction class ` +
                   `${toAsciiSg(sol.sgClass)} was resolved, which is not a single space group. ` +
                   `The cell is complete; the candidate groups are listed in a comment.`,
                   'info', 9000);
    } else if (Number.isFinite(sol.sgMargin) && sol.sgMargin < decisiveNats) {
        showStatus(`pdCIF exported with space group ${sgName}, but the Space Group MC was ` +
                   `NOT decisive (${sol.sgMargin.toFixed(1)} nats over the runner-up). The file ` +
                   `records that; treat the symbol as one of several.`, 'info', 9000);
    }
};
