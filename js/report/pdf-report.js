// js/report/pdf-report.js
// PDF report.
//
// Classic script, loaded in order by brutus.html (see the list there); its
// top-level names are shared with the other app scripts.

// 5. Action: Single Report
document.getElementById('ctx-report').addEventListener('click', () => {
    const target = ctxTargetSolution();
    if (target) {
        // Call report function with the specific solution
        generatePDFReport(target);
    }
});
//main report function, on peut optimiser...
const generatePDFReport = async (singleSolution = null) => {
const solutionsToReport = singleSolution ? [singleSolution] : displayedSolutions;    

if (displayedSolutions.length === 0) {
    showStatus("No solutions found to generate a report.", 'info');
    return;
}

const tthMinVal = parseFloat(ui.tthMinSlider.value);
const tthMaxVal = parseFloat(ui.tthMaxSlider.value);
const reportPeaks = pickedPeaks.filter(p => p.tth >= tthMinVal && p.tth <= tthMaxVal);

if (reportPeaks.length === 0) {
     showStatus("No peaks selected in the current 2-theta range for the report.", 'info');
     return;
}

// Safeguard: Ensure every solution being reported has undergone space group analysis
if (spaceGroupData) {
    solutionsToReport.forEach(sol => {
        if (!sol.analysis) {
            sol.analysis = analyzeSystematicAbsences(
                sol,
                reportPeaks,
                spaceGroupData,
                getWavelength(),
                getTthError(),
                tthMaxVal,
                getImpurityPeaks(),
                tthMinVal
            );
        }
    });
}

ui.reportButton.textContent = 'Generating...';
ui.reportButton.disabled = true;
document.body.style.cursor = 'wait';

try {
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF({
        orientation: 'p',
        unit: 'mm',
        format: 'a4'
    });

    const margin = 15;
    let yPos = 20;
    const pdfWidth = doc.internal.pageSize.getWidth();
    const lambda = getWavelength();
    const _activeKa2Preset = (typeof getActiveKa2Preset === 'function') ? getActiveKa2Preset() : null;
    const lambdaKa1 = _activeKa2Preset ? _activeKa2Preset.ka1 : null;
    const lambdaKa2 = _activeKa2Preset ? _activeKa2Preset.ka2 : null;
    const tthError = getTthError();
        
    const FONT = {
        TITLE: 'helvetica',
        LABEL: 'helvetica',
        DATA: 'courier'
    };
    const SIZE = {
        TITLE: 18,
        H1: 14,
        H2: 12,
        BODY: 9,
        TABLE_HEADER: 8,
        TABLE_BODY: 8,
        SMALL: 7
    };

    // header
    const now = new Date();
    const timestamp = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')} ${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}:${String(now.getSeconds()).padStart(2, '0')}`;
    const versionInfo = document.getElementById('app-footer')?.textContent || 'Brutus, 22 april 2026';
    const programURL = window.location.href;
        
    doc.setFont(FONT.TITLE, 'bold').setFontSize(SIZE.TITLE).text('Brutus - Powder Indexing Report', pdfWidth / 2, yPos, { align: 'center' });
    yPos += 10;
        
    doc.setFont(FONT.LABEL, 'normal').setFontSize(SIZE.BODY);
    doc.text(`Generated:`, margin, yPos);
    doc.setFont(FONT.DATA, 'normal').text(timestamp, margin + 25, yPos);
    yPos += 5;

    doc.setFont(FONT.LABEL, 'normal').setFontSize(SIZE.BODY);
    doc.text(`URL:`, margin, yPos);
    doc.setFont(FONT.DATA, 'normal').text(programURL, margin + 25, yPos);
    yPos += 5;

    doc.setFont(FONT.LABEL, 'normal').setFontSize(SIZE.BODY);
    doc.text(`Version:`, margin, yPos);
    doc.setFont(FONT.DATA, 'normal').text(versionInfo, margin + 25, yPos);
    yPos += 5;
        
    doc.setFont(FONT.LABEL, 'normal').text(`Data File:`, margin, yPos);
    // Read the variable, not the DOM. The filename now lives in a chip with
    // sibling nodes (size, clear button), so textContent would drag "248 KB"
    // into the report's Data file line.
    doc.setFont(FONT.DATA, 'normal').text(loadedFileName || '', margin + 25, yPos);
    yPos += 10;
      
    const imgData = xrdChart.toBase64Image('image/png', 1.0);      
    const imgProps = doc.getImageProperties(imgData);
    const availableWidth = pdfWidth - 2 * margin;
    const pdfHeight = doc.internal.pageSize.getHeight();
    const availableHeight = pdfHeight - yPos - margin - 5; 
    const scale = Math.min(availableWidth / imgProps.width, availableHeight / imgProps.height);
    const drawWidth = imgProps.width * scale;
    const drawHeight = imgProps.height * scale;
    const drawX = margin + (availableWidth - drawWidth) / 2; 
    doc.addImage(imgData, 'PNG', drawX, yPos, drawWidth, drawHeight);

    // Parameters
    doc.addPage();
    yPos = 20;

    doc.setFont(FONT.LABEL, 'bold').setFontSize(SIZE.H1).text('Indexing Parameters', margin, yPos);
    yPos += 8;

    const presetText = ui.wavelengthPreset.options[ui.wavelengthPreset.selectedIndex].text;
    const paramData = [
        { label: 'Radiation:', value: presetText },
        { label: 'Max Volume (A^3):', value: String(getMaxVolume()) },
        { label: 'Wavelength (A):', value: getWavelength().toFixed(5) },
        { label: 'Tolerance (2theta):', value: String(getTthError()) },
        { label: 'Ka2 Identified:',
          value: getActiveKa2Preset()
                     ? (ui.stripKa2Checkbox.checked ? 'True' : 'False')
                     : 'N/A' },
        { label: 'Impurity Peaks:', value: ui.impurityPeaksInput.value },
        { label: 'Min Peak (%):', value: ui.peakThresholdValue.textContent },
        { label: 'Prominence (%):', value: ui.peakProminenceValue ? ui.peakProminenceValue.textContent : '0' },
        { label: 'Refine Zero:', value: ui.refineZeroCheckbox.checked ? 'True' : 'False' },
        { label: '2theta Min (deg):', value: tthMinVal.toFixed(2) },
        { label: '2theta Max (deg):', value: tthMaxVal.toFixed(2) },
    ];
    // The GPU search settings the last run used (captured when it started),
    // on rows of their own. They used to be printed only in the summary line.
    if (lastGpuRunSettings) {
        if (paramData.length % 2) paramData.push({ label: '', value: '' });
        paramData.push(
            { label: 'HKL (%/unknown):', value: String(lastGpuRunSettings.perUnknown) },
            { label: 'Depth:', value: String(lastGpuRunSettings.depth) },
            { label: 'FoM Tolerance:', value: String(lastGpuRunSettings.fom) },
            { label: 'Candidates:', value: Number(lastGpuRunSettings.candidates).toLocaleString('en-US') });
    }

    const col1X = margin;
    const col2X = margin + 85;
    const labelWidth = 35;

    paramData.forEach((item, index) => {
        const isCol1 = index % 2 === 0;
        const x = isCol1 ? col1X : col2X;
        if (isCol1) yPos += 5;
        
        doc.setFont(FONT.LABEL, 'normal').setFontSize(SIZE.BODY).text(item.label, x, yPos);
        doc.setFont(FONT.DATA, 'normal').setFontSize(SIZE.BODY).text(String(item.value), x + labelWidth, yPos);
    });
    yPos += 7;
    // (No "Systems Searched" line: the per-system table below lists them.)

    doc.setDrawColor(200); doc.line(margin, yPos, pdfWidth - margin, yPos); yPos += 8;

    doc.setFont(FONT.LABEL, 'bold').setFontSize(SIZE.H1).text('Indexing Solutions Summary', margin, yPos); yPos += 8;
          
    // Per-system breakdown of the last run: basis, peaks, trials searched /
    // planned, candidates refined, time, and a note (truncated, skipped,
    // stopped...). Fixed-width rows in the data font (Courier), so the
    // columns line up without a table layout. The run totals follow it; the
    // search settings are in the parameter block above, so the old one-line
    // summary that repeated all of this is gone.
    const hasSystemRows = Array.isArray(lastSystemSearchStats) && lastSystemSearchStats.length;
    if (hasSystemRows) {
        const rows = formatSystemSearchStats(lastSystemSearchStats);
        doc.setFont(FONT.DATA, 'normal').setFontSize(SIZE.SMALL);
        rows.forEach((row, i) => {
            if (yPos > 280) { doc.addPage(); yPos = 20; }
            doc.setFont(FONT.DATA, i === 0 ? 'bold' : 'normal');
            doc.text(row, margin, yPos);
            yPos += 3.5;
        });
    }
    if (lastRunTotals) {
        const n = (v) => Math.round(v).toLocaleString('en-US');
        const t = lastRunTotals;
        let line = (t.total !== null && t.total > 0)
            ? `Trials: ${n(t.done)} / ${n(t.total)} (${fmtSearchedPercent(Math.min(1, t.done / t.total))})`
            : `Trials: ${n(t.done)}${t.total === null ? ' (CPU)' : ''}`;
        line += `    Time: ${t.time}`;
        if (t.failures && t.failures.length) line += `    INCOMPLETE: ${t.failures.join(' ')}`;
        if (hasSystemRows) yPos += 1.5;
        doc.setFont(FONT.DATA, 'bold').setFontSize(SIZE.SMALL);
        // Wrapped: an INCOMPLETE message can be longer than the page.
        doc.splitTextToSize(line, pdfWidth - 2 * margin).forEach(l => {
            if (yPos > 280) { doc.addPage(); yPos = 20; }
            doc.text(l, margin, yPos);
            yPos += 3.5;
        });
    }
    if (hasSystemRows || lastRunTotals) yPos += 4;
          
    doc.setFont(FONT.LABEL, 'bold').setFontSize(SIZE.TABLE_HEADER);
    doc.text('Sys', margin, yPos);
    const first_sol_n_20 = (displayedSolutions.length > 0 && displayedSolutions[0].n_20) ? displayedSolutions[0].n_20 : Math.min(20, reportPeaks.length);
    doc.text(`M(${first_sol_n_20})`, margin + 15, yPos);
    doc.text(`F(${first_sol_n_20})`, margin + 30, yPos);
    doc.text('Volume(A^3)', margin + 45, yPos);
    doc.text('Parameters', margin + 75, yPos);
    yPos += 5;

    doc.setFont(FONT.DATA, 'normal').setFontSize(SIZE.TABLE_BODY);
    solutionsToReport.slice(0, 30).forEach(sol => {
        if (yPos > 280) { doc.addPage(); yPos = 20; }
        let paramStr = '';
        const p = sol.errors || {};
        switch (sol.system) {
             case 'cubic': paramStr = `a=${formatWithError(sol.a, p.s_a)}`; break;
             case 'tetragonal': paramStr = `a=${formatWithError(sol.a, p.s_a)}, c=${formatWithError(sol.c, p.s_c)}`; break;
             case 'hexagonal': paramStr = `a=${formatWithError(sol.a, p.s_a)}, c=${formatWithError(sol.c, p.s_c)}`; break;
             case 'orthorhombic': paramStr = `a=${formatWithError(sol.a, p.s_a)}, b=${formatWithError(sol.b, p.s_b)}, c=${formatWithError(sol.c, p.s_c)}`; break;
             case 'monoclinic': paramStr = `a=${formatWithError(sol.a, p.s_a)}, b=${formatWithError(sol.b, p.s_b)}, c=${formatWithError(sol.c, p.s_c)}, beta=${formatWithError(sol.beta, p.s_beta)}`; break;
             case 'triclinic': 
                paramStr = `a=${formatWithError(sol.a, p.s_a)}, b=${formatWithError(sol.b, p.s_b)}, c=${formatWithError(sol.c, p.s_c)}`;
                doc.text(sol.system.substring(0,4) + (sol.lattice === 'R' ? ' R' : ''), margin, yPos);
                doc.text(sol.m20.toFixed(2), margin + 15, yPos);
                doc.text((sol.fN_20 || 0).toFixed(2), margin + 30, yPos);
                doc.text(sol.volume.toFixed(2), margin + 45, yPos);
                doc.text(paramStr, margin + 75, yPos);
                yPos += 5; 
                paramStr = `al=${formatWithError(sol.alpha, p.s_alpha)}, be=${formatWithError(sol.beta, p.s_beta)}, ga=${formatWithError(sol.gamma, p.s_gamma)}`;
                doc.text(paramStr, margin + 75, yPos); 
                yPos += 5;
                return; 
        }
        doc.text(sol.system.substring(0,4) + (sol.lattice === 'R' ? ' R' : ''), margin, yPos);
        doc.text(sol.m20.toFixed(2), margin + 15, yPos);
        doc.text((sol.fN_20 || 0).toFixed(2), margin + 30, yPos);
        doc.text(sol.volume.toFixed(2), margin + 45, yPos);
        doc.text(paramStr, margin + 75, yPos);
        yPos += 5;
    });

    // ------------------------------------------------------------------
    // SPACE-GROUP DETERMINATION BLOCK
    //
    // Reproduces the Space Group MC's verdict from the fields it stamped on
    // the solution. The report deliberately performs NO ranking of its own:
    // the two used to disagree because they answered different questions
    // (likelihood ratio over merged extinction CLASSES, each with its own
    // refit, versus a violation tally over individual SETTINGS judged
    // against one extinction-blind cell) on different data (the MC excludes
    // Ka2-suspect peaks; the report's analysis does not) from different
    // candidate pools (the report's is pre-filtered by the centering test,
    // the MC's is not, on purpose).
    //
    // Nothing here recomputes: if a value was not produced by an MC run it
    // is not shown, and the absence of a determination is stated plainly.
    const writeSgVerdict = (sol) => {
        const wrap = (text, x, size, style) => {
            doc.setFont(FONT.DATA, style || 'normal').setFontSize(size);
            doc.splitTextToSize(text, pdfWidth - x - margin).forEach(l => {
                if (yPos > 280) { doc.addPage(); yPos = 20; }
                doc.text(l, x, yPos);
                yPos += (size <= SIZE.SMALL) ? 3.5 : 4;
            });
        };

        if (yPos > 258) { doc.addPage(); yPos = 20; }
        doc.setFont(FONT.LABEL, 'normal').setFontSize(SIZE.H2)
           .text('Space Group Determination:', margin, yPos);
        yPos += 6;

        if (!sol.sgClass) {
            wrap('No space-group determination was performed for this cell. The list below states ' +
                 'which settings the observed absences are compatible with; it does not choose ' +
                 'between them. Run "Space Group MC" on this solution to compare the extinction ' +
                 'classes as hypotheses, then add the result as a solution to have it appear here.',
                 margin + 5, SIZE.BODY, 'italic');
            yPos += 3;
            return;
        }

        wrap(`Class: ${sol.sgClass}`, margin + 5, SIZE.BODY, 'bold');
        if (sol.sgMembers && sol.sgMembers.length) {
            wrap(`Space groups in this class: ${sol.sgMembers.join(', ')}`, margin + 5, SIZE.BODY);
        }
        if (sol.sgConditions && sol.sgConditions.length) {
            wrap(`Reflection conditions: ${sol.sgConditions.join(' ; ')}`, margin + 5, SIZE.BODY);
        }

        // The margin, in the MC's own units, with the decisiveness verdict
        // spelled out. A cell taken from a row that was in a tie is not a
        // determination, and the report must not let the bold class name
        // above imply that it was.
        const marg = sol.sgMargin;
        const decisiveNats = (typeof SG_DECISIVE_NATS === 'number') ? SG_DECISIVE_NATS : 2.3;
        if (isFinite(marg)) {
            const odds = Math.exp(Math.min(30, marg));
            if (marg >= decisiveNats) {
                wrap(`Margin: ${marg.toFixed(1)} nats ahead of the runner-up ` +
                     `(about ${odds.toPrecision(2)}:1).`, margin + 5, SIZE.BODY);
            } else {
                doc.setTextColor(150, 60, 0);
                wrap(`NOT DECISIVE: only ${marg.toFixed(1)} nats ahead of the runner-up ` +
                     `(about ${odds.toPrecision(2)}:1), below the ${decisiveNats.toFixed(1)}-nat ` +
                     `threshold. The absences in this pattern do not separate this class from the ` +
                     `next one; the cell below is refined under this hypothesis, not proof of it.`,
                     margin + 5, SIZE.BODY);
                doc.setTextColor(0, 0, 0);
            }
        } else if (marg === Infinity) {
            wrap('Margin: the only class the data do not contradict.', margin + 5, SIZE.BODY);
        }

        const ev = sol.sgEvidence;
        if (ev) {
            const bits = [];
            if (isFinite(ev.clean) && isFinite(ev.informative)) {
                bits.push(`${ev.clean}/${ev.informative} forbidden lines clean`);
            }
            if (isFinite(ev.hardViolations)) {
                bits.push(`${ev.hardViolations} hard violation(s)` +
                          (ev.softViolations ? ` (+${ev.softViolations} soft)` : ''));
            }
            if (isFinite(ev.unindexed)) bits.push(`${ev.unindexed} unindexed peak(s)`);
            if (bits.length) wrap(`Evidence: ${bits.join(', ')}.`, margin + 5, SIZE.BODY);

            // How the number was arrived at matters as much as the number.
            const how = [];
            how.push(ev.wilson
                ? 'absences weighted per reflection (Wilson |E|^2)'
                : 'absences weighted uniformly (no intensity weighting)');
            if (isFinite(ev.pHat)) how.push(`p(line observed) = ${(ev.pHat * 100).toFixed(0)}%`);
            how.push(ev.mode === 'mc'
                ? 'cell refined by Monte-Carlo under this hypothesis'
                : 'cell refined by least squares only (stage 1; not directly comparable with fully refined classes)');
            wrap(`Method: ${how.join('; ')}.`, margin + 5, SIZE.SMALL, 'italic');
        }
        if (isFinite(sol.sgScore)) {
            wrap(`Log-odds score: ${sol.sgScore.toFixed(2)} nats (relative; only differences are meaningful).`,
                 margin + 5, SIZE.SMALL, 'italic');
        }
        yPos += 3;
        doc.setFont(FONT.DATA, 'normal').setFontSize(SIZE.BODY);
    };

    // Detailed Solution 
    solutionsToReport.forEach((sol, solIndex) => {
        doc.addPage(); yPos = 20;
        
        doc.setFont(FONT.LABEL, 'bold').setFontSize(SIZE.H1); 
        doc.text(`Details for Solution #${solIndex + 1}: ${sol.system}${sol.lattice === 'R' ? ' (R lattice, hexagonal axes)' : ''}`, margin, yPos); 
        yPos += 8; 
        
        doc.setFont(FONT.DATA, 'normal').setFontSize(SIZE.BODY); 
        const p = sol.errors || {};
        const paramLines = [];
        
        switch (sol.system) {
            case 'cubic':
                paramLines.push({ label: 'a', value: `= ${formatWithError(sol.a, p.s_a)} A` });
                break;
            case 'tetragonal':
                paramLines.push({ label: 'a', value: `= ${formatWithError(sol.a, p.s_a)} A` });
                paramLines.push({ label: 'c', value: `= ${formatWithError(sol.c, p.s_c)} A` });
                break;
            case 'hexagonal':
                paramLines.push({ label: 'a', value: `= ${formatWithError(sol.a, p.s_a)} A` });
                paramLines.push({ label: 'c', value: `= ${formatWithError(sol.c, p.s_c)} A` });
                break;
            case 'orthorhombic':
                paramLines.push({ label: 'a', value: `= ${formatWithError(sol.a, p.s_a)} A` });
                paramLines.push({ label: 'b', value: `= ${formatWithError(sol.b, p.s_b)} A` });
                paramLines.push({ label: 'c', value: `= ${formatWithError(sol.c, p.s_c)} A` });
                break;
            case 'monoclinic':
                paramLines.push({ label: 'a', value: `= ${formatWithError(sol.a, p.s_a)} A` });
                paramLines.push({ label: 'b', value: `= ${formatWithError(sol.b, p.s_b)} A` });
                paramLines.push({ label: 'c', value: `= ${formatWithError(sol.c, p.s_c)} A` });
                paramLines.push({ label: 'beta', value: `= ${formatWithError(sol.beta, p.s_beta)} deg` });
                break;
            case 'triclinic':
                paramLines.push({ label: 'a', value: `= ${formatWithError(sol.a, p.s_a)} A` });
                paramLines.push({ label: 'b', value: `= ${formatWithError(sol.b, p.s_b)} A` });
                paramLines.push({ label: 'c', value: `= ${formatWithError(sol.c, p.s_c)} A` });
                paramLines.push({ label: 'alpha', value: `= ${formatWithError(sol.alpha, p.s_alpha)} deg` });
                paramLines.push({ label: 'beta', value: `= ${formatWithError(sol.beta, p.s_beta)} deg` });
                paramLines.push({ label: 'gamma', value: `= ${formatWithError(sol.gamma, p.s_gamma)} deg` });
                break;
        }
        
        paramLines.push({ label: 'Volume', value: `= ${sol.volume.toFixed(2)} A^3` });
    
        if (sol.zero_correction !== undefined) { 
            paramLines.push({ label: 'Zero Error (2theta)', value: `= ${formatWithError(sol.zero_correction, p.s_zero)} deg` });
        }
        
        const n_20_pdf = sol.n_20 || Math.min(20, reportPeaks.length);
        paramLines.push({ label: `M(${n_20_pdf})`, value: `= ${sol.m20.toFixed(2)}` });
        paramLines.push({ label: `F(${n_20_pdf})`, value: `= ${(sol.fN_20 || 0).toFixed(2)}` });

        const n_all_pdf = sol.n_all || reportPeaks.length;
        paramLines.push({ label: `M(${n_all_pdf})`, value: `= ${(sol.m_all || 0).toFixed(2)}` });
        paramLines.push({ label: `F(${n_all_pdf})`, value: `= ${(sol.fN_all || 0).toFixed(2)}` });
                 
        const longestLabelWidth = Math.max(...paramLines.map(line => doc.getTextWidth(line.label)));
        const labelEndX = margin + longestLabelWidth;
        const dataStartX = labelEndX + 2; 
    
        paramLines.forEach(line => {
            doc.text(line.label, labelEndX, yPos, { align: 'right' });
            doc.text(line.value, dataStartX, yPos);
            yPos += 4;
        });
    
        yPos += 3; 

        if (sol.analysis && sol.analysis.centering) {
            doc.setFont(FONT.LABEL, 'normal').setFontSize(SIZE.H2).text(`Lattice Centering:`, margin, yPos);
            doc.setFont(FONT.DATA, 'normal').setFontSize(SIZE.BODY).text(sol.analysis.centering, margin + 42, yPos);
            yPos += 7;
        }

        if (yPos > 225) { doc.addPage(); yPos = 20; }

        try {
            const r = Math.PI / 180.0;
            const cellVolume = (cl) => {
                const ca = Math.cos(cl.alpha * r), cb = Math.cos(cl.beta * r), cg = Math.cos(cl.gamma * r);
                const term = Math.max(0, 1 - ca*ca - cb*cb - cg*cg + 2*ca*cb*cg);
                return (term > 0) ? (cl.a * cl.b * cl.c * Math.sqrt(term)) : 0;
            };

            const col1LabelEndX = margin + 15;
            const col1DataStartX = col1LabelEndX + 2;
            const col2LabelEndX = margin + 78;
            const col2DataStartX = col2LabelEndX + 2;

            const drawCellBlock = (title, cl, subtitle) => {
                doc.setFont(FONT.LABEL, 'normal').setFontSize(SIZE.H2).text(title, margin, yPos);
                yPos += 5;
                if (subtitle) {
                    doc.setFont(FONT.LABEL, 'normal').setFontSize(SIZE.SMALL);
                    doc.setTextColor(90, 90, 90);
                    doc.text(subtitle, margin, yPos);
                    doc.setTextColor(0, 0, 0);
                    yPos += 5;
                }

                const d = [
                    { label: 'a',     value: `= ${cl.a.toFixed(4)} A` },
                    { label: 'b',     value: `= ${cl.b.toFixed(4)} A` },
                    { label: 'c',     value: `= ${cl.c.toFixed(4)} A` },
                    { label: 'alpha', value: `= ${cl.alpha.toFixed(3)} deg` },
                    { label: 'beta',  value: `= ${cl.beta.toFixed(3)} deg` },
                    { label: 'gamma', value: `= ${cl.gamma.toFixed(3)} deg` }
                ];

                doc.setFont(FONT.DATA, 'normal').setFontSize(SIZE.TABLE_BODY);

                for (let i = 0; i < 3; i++) {
                    doc.text(d[i].label, col1LabelEndX, yPos, { align: 'right' });
                    doc.text(d[i].value, col1DataStartX, yPos);
                    doc.text(d[i + 3].label, col2LabelEndX, yPos, { align: 'right' });
                    doc.text(d[i + 3].value, col2DataStartX, yPos);
                    yPos += 4;
                }

                doc.text("Volume", col1LabelEndX, yPos, { align: 'right' });
                doc.text(`= ${cellVolume(cl).toFixed(2)} A^3`, col1DataStartX, yPos);
                yPos += 6;
            };

            const niggliResult = reduceToNiggliCell(sol);
            const nCell = niggliResult.cell;
            const nVol = cellVolume(nCell);

            const centeringStr = String((sol.analysis && sol.analysis.centering) || '').toUpperCase();
            let centeringKey = null;   
            for (const cType of ['F', 'I', 'R', 'A', 'B', 'C', 'P']) {
                if (centeringStr.includes(`(${cType})`)) { centeringKey = cType; break; }
            }

            let subtitle1, title1;
            if (centeringKey === 'P') {
                title1 = 'Reduced (Niggli) Cell:';
                subtitle1 = 'Krivy-Gruber reduction of the solved lattice metric. Lattice is primitive, so this IS the reduced (Niggli) cell.';
            } else if (centeringKey === null) {
                title1 = 'Standardised Conventional Cell:';
                subtitle1 = 'Krivy-Gruber reduction of the solved lattice metric. Centering undetermined, so no primitive cell is derived.';
            } else {
                title1 = 'Standardised Conventional Cell (centering not applied):';
                subtitle1 = `Krivy-Gruber reduction of the conventional basis only; the detected ${centeringKey}-centering is NOT applied, so this is not the reduced cell of the lattice.`;
            }

            drawCellBlock(title1, nCell, subtitle1);

            if (centeringKey && centeringKey !== 'P') {
                try {
                    const primResult = reduceToNiggliCell(sol, { centering: centeringKey });
                    const pCell = primResult.cell;
                    const pVol = cellVolume(pCell);
                    const ratio = (pVol > 0) ? (nVol / pVol) : 0;

                    drawCellBlock(
                        'Reduced (Niggli) Cell - primitive:',
                        pCell,
                        `Reduced cell of the ${centeringKey}-centered lattice, centering applied`
                            + (ratio > 0 ? `; volume is 1/${ratio.toFixed(0)} of the cell above.` : '.')
                    );

                    if (primResult.converged === false) {
                        doc.setFont(FONT.LABEL, 'normal').setFontSize(SIZE.SMALL);
                        doc.setTextColor(180, 0, 0);
                        doc.text('Warning: reduction did not converge; cell above may not be fully reduced.', margin, yPos);
                        doc.setTextColor(0, 0, 0);
                        yPos += 5;
                    }
                } catch (ePrim) {
                    doc.setFont(FONT.LABEL, 'normal').setFontSize(SIZE.SMALL);
                    doc.setTextColor(180, 0, 0);
                    doc.text(`Reduced (Niggli) cell could not be computed for ${centeringKey}-centering.`, margin, yPos);
                    doc.setTextColor(0, 0, 0);
                    yPos += 6;
                }
            }

            const symmetryOrder = { 'cubic': 6, 'hexagonal': 5, 'tetragonal': 4, 'orthorhombic': 3, 'monoclinic': 2, 'triclinic': 1 };
            const currentOrder = symmetryOrder[sol.system] || 1;
            const niggliSym = getSymmetry(nCell.a, nCell.b, nCell.c, nCell.alpha, nCell.beta, nCell.gamma, 0.25);
            const detectedHigherSyms = [];
            
            if (symmetryOrder[niggliSym] > currentOrder) {
                detectedHigherSyms.push({
                    system: niggliSym,
                    note: `Metric: ${niggliSym}`,
                    cell: nCell
                });
            }
            
            const checkSystems = ['cubic', 'hexagonal', 'tetragonal', 'orthorhombic', 'monoclinic'];
            checkSystems.forEach(sys => {
                if (symmetryOrder[sys] > currentOrder) {
                    const equiv = generateEquivalentCells(nCell, 0, sys);
                    if (equiv && equiv.centeredCells) {
                        Object.values(equiv.centeredCells).forEach(cCell => {
                            if (symmetryOrder[cCell.system] > currentOrder) {
                                detectedHigherSyms.push({
                                    system: cCell.system,
                                    note: `${cCell.centering}-centered ${cCell.system}`,
                                    cell: cCell
                                });
                            }
                        });
                    }
                }
            });

            if (detectedHigherSyms.length > 0) {
                const uniqSyms = Array.from(new Map(detectedHigherSyms.map(item => [item.note, item])).values());
                
                if (yPos > 255) { doc.addPage(); yPos = 20; }
                
                doc.setFont(FONT.LABEL, 'bold').setFontSize(SIZE.BODY);
                doc.setTextColor(220, 38, 38); 
                doc.text("HIGHER SYMMETRY DETECTED IN CONVENTIONAL CELL METRIC", margin, yPos);
                doc.setTextColor(0, 0, 0); 
                yPos += 4.5;
                
                doc.setFont(FONT.DATA, 'normal').setFontSize(SIZE.SMALL);
                doc.text(`This ${sol.system} cell reduces to a metric tensor consistent with higher symmetry:`, margin + 2, yPos);
                yPos += 4;
                
                uniqSyms.forEach(sym => {
                    if (yPos > 275) { doc.addPage(); yPos = 20; }
                    const c = sym.cell;
                    let paramStr = `a=${c.a.toFixed(3)}, b=${c.b.toFixed(3)}, c=${c.c.toFixed(3)}`;
                    if (sym.system !== 'cubic' && sym.system !== 'orthorhombic' && sym.system !== 'tetragonal') {
                        paramStr += `, al=${c.alpha.toFixed(2)}, be=${c.beta.toFixed(2)}, ga=${c.gamma.toFixed(2)}`;
                    }
                    const labelStr = `* ${sym.note.toUpperCase()}:`;
                    doc.setFont(FONT.DATA, 'bold');
                    doc.text(labelStr, margin + 4, yPos);
                    doc.setFont(FONT.DATA, 'normal');
                    const labelW = doc.getTextWidth(labelStr);
                    const paramX = Math.max(margin + 50, margin + 4 + labelW + 3);
                    if (paramX + doc.getTextWidth(paramStr) > pdfWidth - margin) {
                        yPos += 3.5;
                        doc.text(paramStr, margin + 8, yPos);
                    } else {
                        doc.text(paramStr, paramX, yPos);
                    }
                    yPos += 4;
                });
                
                doc.setFont(FONT.DATA, 'italic').setFontSize(SIZE.SMALL);
                doc.text("Note: Consider refining in the higher symmetry system to reduce degrees of freedom.", margin + 2, yPos);
                doc.setFont(FONT.DATA, 'normal').setFontSize(SIZE.TABLE_BODY);
                yPos += 4;
            }

        } catch (niggliError) {
            console.error("Niggli reduction failed:", niggliError);
            doc.setFont(FONT.DATA, 'italic').setFontSize(SIZE.BODY);
            doc.text('Reduction failed. See console for error.', margin + 5, yPos);
            yPos += 5;
        }

        yPos += 4;

        if (sol.analysis) {
             doc.setFont(FONT.LABEL, 'normal').setFontSize(SIZE.H2).text('Possible Extinctions:', margin, yPos); 
             let extinctionList = sol.analysis.detectedExtinctions || [];
             let extinctionText = "";

             if (extinctionList.length > 0 && extinctionList[0] !== "None detected") {
                  extinctionText = extinctionList.join(', '); 
             } else {
                  extinctionText = "None clearly detected";
                  doc.setFont(FONT.DATA, 'italic'); 
             }

             const extinctionX = margin + 42;
             const extinctionMaxWidth = pdfWidth - extinctionX - margin; 
             const extinctionLines = doc.splitTextToSize(extinctionText, extinctionMaxWidth);

             doc.setFont(FONT.DATA, 'normal').setFontSize(SIZE.BODY); 
             extinctionLines.forEach(line => {
                  if (yPos > 280) { 
                       doc.addPage();
                       yPos = 20;
                  }
                  doc.text(line, extinctionX, yPos);
                  yPos += 4; 
             });
             yPos += 3; 

             // --- SPACE-GROUP DETERMINATION -------------------------------
             // The determination is the Space Group MC's, not this report's.
             // It is reproduced verbatim from the fields the MC stamped onto
             // the solution when the user pressed "Add as solution"; the
             // report does not re-derive, re-order or second-guess it. When
             // no MC was run the report says so, rather than substituting
             // the weaker violation-tally ordering that used to sit here and
             // silently disagree with what was on screen.
             writeSgVerdict(sol);

             const sgCompatible = sol.analysis.compatibleSettings ||
                                  sol.analysis.rankedSpaceGroups || [];
             if (sgCompatible.length > 0) {
                 if (yPos > 260) { doc.addPage(); yPos = 20; }
                 doc.setFont(FONT.LABEL, 'normal').setFontSize(SIZE.H2)
                    .text('Space groups compatible with the observed absences:', margin, yPos);
                 yPos += 6;

                 doc.setFont(FONT.DATA, 'italic').setFontSize(SIZE.SMALL);
                 const listNote = doc.splitTextToSize(
                     'Not a ranking. Settings are grouped by how many observed reflections contradict them ' +
                     'and listed by space-group number within each group. Ordering candidates by agreement ' +
                     'counts is unreliable, because a less-constrained setting can never score worse than a ' +
                     'more-constrained one it contains; use the Space Group MC to compare hypotheses.',
                     pdfWidth - 2 * margin - 5);
                 listNote.forEach(l => {
                     if (yPos > 280) { doc.addPage(); yPos = 20; }
                     doc.text(l, margin + 5, yPos);
                     yPos += 3.5;
                 });
                 yPos += 1.5;
                 doc.setFont(FONT.DATA, 'normal').setFontSize(SIZE.BODY);

                 if (sol.analysis.usedKa2SoftScoring) {
                     doc.setFont(FONT.DATA, 'italic').setFontSize(SIZE.SMALL);
                     const note = '(Hard = violations by real reflections. Soft = from Ka2-suspect/weak peaks only, shown for reference.)';
                     doc.text(note, margin + 5, yPos);
                     yPos += 4;
                     doc.setFont(FONT.DATA, 'normal').setFontSize(SIZE.BODY);
                 }

                 const sgList = sgCompatible;
                 const groupsByViolation = {};
                 sgList.forEach(sg => {
                     const v = sg.hardViolations || 0;
                     if (!groupsByViolation[v]) groupsByViolation[v] = [];
                     groupsByViolation[v].push(sg);
                 });

                 const violationBuckets = Object.keys(groupsByViolation).map(Number).sort((a, b) => a - b);

                 const printViolationList = (list, header, italic) => {
                     if (!list || list.length === 0) return;
                     if (yPos > 280) { doc.addPage(); yPos = 20; }
                     doc.setFont(FONT.DATA, italic ? 'italic' : 'normal').setFontSize(SIZE.SMALL);
                     doc.text(header, margin + 10, yPos);
                     yPos += 3.5;
                     list.forEach(viol => {
                         if (yPos > 280) { doc.addPage(); yPos = 20; }
                         const violationLines = doc.splitTextToSize(`- ${viol}`, pdfWidth - margin - margin - 15);
                         violationLines.forEach(vl => {
                             if (yPos > 280) { doc.addPage(); yPos = 20; }
                             doc.text(vl, margin + 12, yPos);
                             yPos += 3.5;
                         });
                     });
                 };

                 violationBuckets.forEach(v => {
                     if (yPos > 270) { doc.addPage(); yPos = 20; }
                     doc.setFont(FONT.LABEL, 'bold').setFontSize(SIZE.BODY).text(`[${v} hard violation${v !== 1 ? 's' : ''}]:`, margin, yPos);
                     yPos += 5;

                     groupsByViolation[v].forEach(sg => {
                         if (yPos > 280) { doc.addPage(); yPos = 20; }

                         const softTag = sg.softViolations > 0 ? `  [+${sg.softViolations} soft]` : '';
                         const extTag = (sg.extinctionsTotal > 0)
                             ? `  [explains ${sg.extinctionsExplained}/${sg.extinctionsTotal} absences]` : '';
                         const acentricTag = (sg.centrosymmetric === false) ? '  [acentric]' : '';
                         doc.setFont(FONT.DATA, 'bold').setFontSize(SIZE.TABLE_BODY);
                         doc.text(`${sg.number}: ${sg.symbol}${softTag}${extTag}${acentricTag}`, margin + 5, yPos);
                         yPos += 4;

                         const unexplained = sg.extinctionsUnexplained || [];
                         if (unexplained.length > 0) {
                             if (yPos > 280) { doc.addPage(); yPos = 20; }
                             doc.setFont(FONT.DATA, 'italic').setFontSize(SIZE.SMALL);
                             doc.setTextColor(150, 60, 0);
                             doc.text(`Does not explain: ${unexplained.join('; ')}`, margin + 10, yPos);
                             doc.setTextColor(0, 0, 0);
                             yPos += 4;
                         }

                         const hardList = sg.violatedReflectionsHard || [];
                         const softList = sg.violatedReflectionsSoft || [];

                         printViolationList(hardList, 'Hard violations (reflections observed on lines this setting forbids):', false);
                         printViolationList(softList, 'Soft violations (Ka2-suspect/weak; shown for reference only):', true);

                         doc.setFont(FONT.DATA, 'normal').setFontSize(SIZE.BODY);
                         yPos += 2;
                     });
                     yPos += 2;
                 });

                 // The list is capped. Say so explicitly: a truncated list
                 // read as complete is how "the correct group is not even a
                 // candidate" gets mistaken for "the correct group was ruled
                 // out".
                 const nTotal = sol.analysis.compatibleSettingsTotal;
                 if (isFinite(nTotal) && nTotal > sgList.length) {
                     if (yPos > 280) { doc.addPage(); yPos = 20; }
                     doc.setFont(FONT.DATA, 'italic').setFontSize(SIZE.SMALL);
                     doc.text(`Showing ${sgList.length} of ${nTotal} compatible settings ` +
                              `(highest space-group numbers first within each violation group).`,
                              margin + 5, yPos);
                     doc.setFont(FONT.DATA, 'normal').setFontSize(SIZE.BODY);
                     yPos += 4;
                 }
             }
             yPos += 3; 

             if (sol.analysis.centeringViolations && Object.keys(sol.analysis.centeringViolations).length > 0) {
                  if (yPos > 268) { doc.addPage(); yPos = 20; }
                  doc.setFont(FONT.LABEL, 'normal').setFontSize(SIZE.H2).text('Centering test violations:', margin, yPos); 
                  yPos += 5;
                  const ch = sol.analysis.centeringViolationsHard || {};
                  const cs = sol.analysis.centeringViolationsSoft || {};
                  const violText = Object.entries(sol.analysis.centeringViolations)
                      .sort(([,a], [,b]) => a - b)
                      .map(([key, val]) => {
                          const hardV = ch[key] != null ? ch[key] : val;
                          const softV = cs[key] != null ? cs[key] : 0;
                          return softV > 0 ? `${key}:${hardV}(+${softV} soft)` : `${key}:${hardV}`;
                      })
                      .join(', ');
                  doc.setFont(FONT.DATA, 'normal').setFontSize(SIZE.BODY).text(violText, margin, yPos);
                  yPos += 5;
                  if (sol.analysis.centeringViolations && sol.analysis.centeringViolationDetails) {
                       doc.setFont(FONT.DATA, 'italic').setFontSize(SIZE.SMALL); 
                       let detailsYOffset = 0; 

                       for (const type of ['I', 'F', 'A', 'B', 'C']) {
                           const count = sol.analysis.centeringViolations[type];
                           const details = sol.analysis.centeringViolationDetails[type];

                           if ((count === 1 || count === 2) && details && details.length > 0) {
                               let detailText = `${type} violation${count > 1 ? 's' : ''}: `;
                               detailText += details.map(d =>
                                   `(${d.h},${d.k},${d.l}) at ${d.tth.toFixed(3)}°`
                               ).join('; ');

                               if (yPos + detailsYOffset > 285) { 
                                   doc.addPage();
                                   yPos = 20;
                                   detailsYOffset = 0; 
                               }
                               doc.text(detailText, margin + 5, yPos + detailsYOffset); 
                               detailsYOffset += 3.5; 
                           }
                       }
                       yPos += detailsYOffset; 
                       doc.setFont(FONT.DATA, 'normal').setFontSize(SIZE.BODY); 
                  }
                  yPos += 5;
             }
        } else {
            // No absence analysis at all (it can throw, and the MC's
            // "Add as solution" path tolerates that). The MC verdict is
            // stored on the solution itself and does not depend on the
            // analysis, so it must still be reported here.
            writeSgVerdict(sol);
        }
        yPos += 4; 
        if (yPos > 255) { doc.addPage(); yPos = 20; }

        doc.setFont(FONT.DATA, 'bold').setFontSize(SIZE.TABLE_HEADER);
        const tableHeader = ' h  k  l  | 2th_exp 2th_cor 2th_cal diff(2t)|   d_corr   d_calc  diff(d)';
        doc.text(tableHeader, margin, yPos); yPos += 4;
        
        const hklList = sol.analysis?.hklList || generateHKL_for_analysis(sol, lambda, tthMaxVal);
        if (hklList.length === 0) {
             doc.setFont(FONT.DATA, 'italic').setFontSize(SIZE.BODY).text('Could not generate theoretical reflections for this cell.', margin, yPos);
             yPos += 5;
             return; 
        }
    
        const ambiguousHkls = (sol.analysis ? sol.analysis.ambiguousHkls : new Set()) || new Set();
        const corrected_tth_obs = reportPeaks.map(p => ({ ...p, tth_corr: p.tth - (sol.zero_correction || 0) }));
        const reportLines = []; 
        const assignedHkls = new Set();
    
        const manualByTth = new Map();
        (sol.manualSwaps || []).forEach(sw => {
             if (sw && Number.isFinite(sw.h) && Number.isFinite(sw.k) && Number.isFinite(sw.l)) {
                 manualByTth.set(Number(sw.tth).toFixed(4), sw);
             }
        });

        corrected_tth_obs.forEach((corr_peak) => {
             let bestMatchHkl = null; let minDiff = Infinity;
             let isManual = false;

             if (corr_peak.ka2Suspect && corr_peak.ka2ParentIdx != null && pickedPeaks[corr_peak.ka2ParentIdx]) {
                 const parentPeak = pickedPeaks[corr_peak.ka2ParentIdx];
                 const parentTthCorr = parentPeak.tth - (sol.zero_correction || 0);
                 
                 let parentHkl = null; let minParentDiff = Infinity;
                 hklList.forEach(hkl => {
                     const diff = Math.abs(hkl.tth - parentTthCorr);
                     if (diff < minParentDiff) { minParentDiff = diff; parentHkl = hkl; }
                 });

                 const manParent = manualByTth.get(Number(parentPeak.tth).toFixed(4));
                 if (manParent) {
                     const forced = hklList.find(x => x.h === manParent.h && x.k === manParent.k && x.l === manParent.l);
                     if (forced) parentHkl = forced;
                 }

                 if (parentHkl && lambdaKa2 && parentHkl.d > 0) {
                     const sinTheta2 = lambdaKa2 / (2 * parentHkl.d);
                     if (sinTheta2 < 1.0) {
                         const tthCalcKa2 = 2 * Math.asin(sinTheta2) * (180 / Math.PI);
                         bestMatchHkl = {
                             h: parentHkl.h, k: parentHkl.k, l: parentHkl.l,
                             tth: tthCalcKa2,
                             d: parentHkl.d
                         };
                         minDiff = Math.abs(bestMatchHkl.tth - corr_peak.tth_corr);
                     }
                 }
             } else {
                 hklList.forEach(hkl => { const diff = Math.abs(hkl.tth - corr_peak.tth_corr); if (diff < minDiff) { minDiff = diff; bestMatchHkl = hkl; } });

                 const _man = manualByTth.get(Number(corr_peak.tth).toFixed(4));
                 if (_man) {
                     const forced = hklList.find(x => x.h === _man.h && x.k === _man.k && x.l === _man.l);
                     if (forced) { bestMatchHkl = forced; minDiff = Math.abs(forced.tth - corr_peak.tth_corr); isManual = true; }
                 }
             }

             const indexWindow = tthError * 1.5;

             if (bestMatchHkl && (isManual || minDiff < indexWindow) && corr_peak.tth_corr >= tthMinVal && corr_peak.tth_corr <= tthMaxVal) {
                  reportLines.push({
                     h: bestMatchHkl.h, k: bestMatchHkl.k, l: bestMatchHkl.l,
                     tth_meas: corr_peak.tth, tth_corr: corr_peak.tth_corr,
                     tth_calc: bestMatchHkl.tth, d_calc: bestMatchHkl.d,
                     ka2Suspect: !!corr_peak.ka2Suspect,
                     hasKa2Child: !!corr_peak.hasKa2Child,
                     manual: isManual
                  });
                  assignedHkls.add(`${bestMatchHkl.h},${bestMatchHkl.k},${bestMatchHkl.l}`);
             }
        });
    
        hklList.forEach(hkl => {
             if (!assignedHkls.has(`${hkl.h},${hkl.k},${hkl.l}`) && hkl.tth >= tthMinVal && hkl.tth <= tthMaxVal) {
                  reportLines.push({ h: hkl.h, k: hkl.k, l: hkl.l, tth_meas: null, tth_corr: null, tth_calc: hkl.tth, d_calc: hkl.d });
             }
        });
        
        reportLines.sort((a, b) => a.tth_calc - b.tth_calc);
        
        doc.setFont(FONT.DATA, 'normal').setFontSize(SIZE.TABLE_BODY);
        
        reportLines.forEach(line => {
             if (yPos > 285) { doc.addPage(); yPos = 20; doc.setFont(FONT.DATA, 'bold').setFontSize(SIZE.TABLE_HEADER); doc.text(tableHeader, margin, yPos); yPos += 4; doc.setFont(FONT.DATA, 'normal').setFontSize(SIZE.TABLE_BODY); }
            
             const hkl_key = `${line.h},${line.k},${line.l}`;
             const isAmbiguous = ambiguousHkls.has(hkl_key);
             if (isAmbiguous) {
                 doc.setFont(FONT.DATA, 'italic');
             }

             let lambdaForD = lambda;
             if (line.hasKa2Child && lambdaKa1) lambdaForD = lambdaKa1;
             else if (line.ka2Suspect && lambdaKa2) lambdaForD = lambdaKa2;

             const d_corr = line.tth_corr ? lambdaForD / (2 * Math.sin(line.tth_corr * Math.PI / 360)) : null;

             const tth_m = line.tth_meas ? line.tth_meas.toFixed(3) : '-'; 
             const tth_c = line.tth_corr ? line.tth_corr.toFixed(3) : '-'; 
             const diff_2t = line.tth_corr ? (line.tth_corr - line.tth_calc).toFixed(3) : '-'; 

             const d_c_str = d_corr ? d_corr.toFixed(5) : '-'; 
             const diff_d = d_corr ? (d_corr - line.d_calc).toFixed(5) : '-';

             const _marker = line.hasKa2Child ? '*1' : (line.ka2Suspect ? '*2' : ''); 
             const hkl_nums = `${String(line.h).padStart(2)} ${String(line.k).padStart(2)} ${String(line.l).padStart(2)}`;
             const hkl_str = `${hkl_nums}${_marker}`.padEnd(10);
            
             let pdfLine = `${hkl_str}| ${tth_m.padStart(7)} ${tth_c.padStart(7)} ${line.tth_calc.toFixed(3).padStart(7)} ${diff_2t.padStart(8)}| ${d_c_str.padStart(8)} ${line.d_calc.toFixed(5).padStart(8)} ${diff_d.padStart(8)}`;
            
             if (isAmbiguous) {
                 pdfLine += ' *';
             }
             if (line.manual) {
                 pdfLine += '  (manual)';
             }

             doc.text(pdfLine, margin, yPos);
            
             doc.setFont(FONT.DATA, 'normal'); 
             yPos += 3.5;
        });

        if (reportLines.some(l => l.hasKa2Child || l.ka2Suspect)) {
            if (yPos > 276) { doc.addPage(); yPos = 20; }
            doc.setFont(FONT.DATA, 'italic').setFontSize(SIZE.SMALL);
            const note1 = `(*1) Ka1 parent line: d_corr computed with Ka1 = ${lambdaKa1 ? lambdaKa1.toFixed(5) : '-'} A.`;
            const note2 = `(*2) Ka2 companion line: d_corr computed with Ka2 = ${lambdaKa2 ? lambdaKa2.toFixed(5) : '-'} A (same d-spacing as parent).`;
            if (reportLines.some(l => l.hasKa2Child)) { doc.text(note1, margin, yPos); yPos += 3.5; }
            if (reportLines.some(l => l.ka2Suspect)) { doc.text(note2, margin, yPos); yPos += 3.5; }
            doc.setFont(FONT.DATA, 'normal').setFontSize(SIZE.TABLE_BODY);
        }
    });

    const filename = `Indexing-Report-${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, '0')}${String(now.getDate()).padStart(2, '0')}_${String(now.getHours()).padStart(2, '0')}${String(now.getMinutes()).padStart(2, '0')}.pdf`;
    doc.save(filename);
    showStatus('PDF report generated and saved.', 'success');

} catch (error) {
    console.error("Failed to generate PDF:", error);
    showStatus("An error occurred during PDF generation.", 'error');
} finally {
    ui.reportButton.textContent = 'Generate PDF Report';
    ui.reportButton.disabled = (solutions.length === 0);
    document.body.style.cursor = 'default';
}
};
