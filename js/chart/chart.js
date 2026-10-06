// js/chart/chart.js
// Chart.js setup, plugins, markers and redraws.
//
// Classic script, loaded in order by brutus.html (see the list there); its
// top-level names are shared with the other app scripts.

// Snap indicator: a dashed guide at the position a Ctrl+click would use.
Chart.register({
    id: 'snapIndicator',
    afterDraw: chart => {
        const tth = chart.$snapTth;
        if (typeof tth !== 'number' || !isFinite(tth)) return;
        const xs = chart.scales.x, ys = chart.scales.y;
        const xv = xF(tth);
        if (xv < xs.min || xv > xs.max) return;
        const x = xs.getPixelForValue(xv);
        const ctx = chart.ctx;
        ctx.save();
        ctx.strokeStyle = chart.$snapKind === 'hkl' ? 'rgba(37, 99, 235, 0.9)' : 'rgba(16, 185, 129, 0.9)';
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(x, ys.top); ctx.lineTo(x, ys.bottom); ctx.stroke();
        ctx.setLineDash([]);
        ctx.beginPath(); ctx.arc(x, ys.top + 7, 4, 0, Math.PI * 2); ctx.stroke();
        ctx.restore();
    }
});
Chart.register({ id: 'verticalCursorLine', afterDraw: chart => { if (chart.tooltip?._active?.length) { let x = chart.tooltip._active[0].element.x; let yAxis = chart.scales.y; let ctx = chart.ctx; ctx.save(); ctx.beginPath(); ctx.moveTo(x, yAxis.top); ctx.lineTo(x, yAxis.bottom); ctx.lineWidth = 1; ctx.strokeStyle = 'rgba(156, 163, 175, 0.7)'; ctx.setLineDash([4, 4]); ctx.stroke(); ctx.restore(); } } });
// Plugin that draws the "currently-edited peak" indicator as a thin
// amber band across the full plot height. Implemented as a plugin
// (not a bar dataset) because Chart.js bar geometry adds a small
// centering offset that shifts the bar visibly to the right of the
// peak's true 2θ; getPixelForValue is exact.
// The chart instance carries the 2θ to draw via `chart.$selectedPeakTth`,
// which updateAllMarkers sets each time.
Chart.register({
    id: 'selectedPeakLine',
    afterDraw: chart => {
        const tth = chart.$selectedPeakTth;
        if (typeof tth !== 'number' || !isFinite(tth)) return;
        const xScale = chart.scales.x;
        const yScale = chart.scales.y;
        const xv = xF(tth);
        if (xv < xScale.min || xv > xScale.max) return;
        const x = xScale.getPixelForValue(xv);
        const ctx = chart.ctx;
        ctx.save();
        ctx.fillStyle = 'rgba(245, 158, 11, 0.25)';
        ctx.fillRect(x - 2, yScale.top, 4, yScale.bottom - yScale.top);
        ctx.restore();
    }
});
Chart.register({ id: 'legendMargin', beforeInit(chart) { const originalFit = chart.legend.fit; chart.legend.fit = function() { originalFit.bind(chart.legend)(); this.height += 15; }; } });
//  initializeChart
//  Rebuilt from scratch on every axis-mode change: switching the abscissa
//  rewrites every x value in every dataset, and switching the ordinate
//  rewrites the y bounds, so mutating in place would leave stale geometry.
const initializeChart = () => {
    if (xrdChart) xrdChart.destroy();
    if (!workingExperimentalData || !workingExperimentalData.tth.length) return;

    const experimentalPoints = buildExperimentalPoints();
    const yb = yBoundsFor(workingExperimentalData.intensity);
    const ax = xAx(), ay = yAx();

    xrdChart = new Chart(ui.chartCanvas, {
        type: 'line',
        data: {
            datasets: [
                { label: 'Intensity', data: experimentalPoints, borderColor: 'rgba(107, 114, 128, 0.7)', showLine: true, borderWidth: 0.75, pointRadius: 1.5, pointHoverRadius: 4, pointBackgroundColor: 'rgba(107, 114, 128, 0.7)' },
                { type: 'bar', label: 'Observed Peaks', data: [], backgroundColor: 'rgba(239, 68, 68, 0.7)', barThickness: 1 },
                { type: 'bar', label: 'Calculated Peaks', data: [], backgroundColor: 'rgba(59, 130, 246, 0.9)', barThickness: 1 }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            scales: {
                x: {
                    type: 'linear',
                    reverse: ax.reverse,
                    title: { display: true, text: ax.label },
                    offset: false,
                    ticks: {
                        includeBounds: false,
                        callback: (value) => Number(value).toLocaleString(undefined, { maximumFractionDigits: ax.digits })
                    },
                    grid: { drawTicks: true, drawBorder: true }
                },
                y: {
                    title: { display: true, text: ay.label },
                    min: yb.min,
                    max: yb.max,
                    offset: false,
                    ticks: {
                        includeBounds: false,
                        // Ticks are spaced in transformed space but labelled with the
                        // intensity they actually stand for, so a sqrt or log plot is
                        // still read in counts rather than in sqrt-counts or decades.
                        callback: (value) => {
                            const real = ay.inv(value);
                            if (!isFinite(real)) return '';
                            if (Math.abs(real) >= 100000) return real.toExponential(1);
                            return real.toLocaleString(undefined, { maximumFractionDigits: Math.abs(real) < 10 ? 2 : 0 });
                        }
                    },
                    grid: { drawTicks: true, drawBorder: true }
                }
            },
            plugins: {
                zoom: {
                    pan: { 
                        enabled: true, 
                        mode: 'xy', 
                        modifierKey: 'alt',
                        onPanComplete: () => { updateAllMarkers(); } 
                    },
                    zoom: { 
                        wheel: { enabled: true }, 
                        pinch: { enabled: true }, 
                        drag: { 
                            enabled: true,
                            backgroundColor: 'rgba(59, 130, 246, 0.15)',
                            borderColor: 'rgba(59, 130, 246, 0.5)',
                            borderWidth: 1
                        }, 
                        mode: 'xy',
                        onZoomComplete: () => { updateAllMarkers(); } 
                    }
                },

                legend: { position: 'top' },
                tooltip: {
                    callbacks: {
                        // Every abscissa flavour is shown at once, so switching the
                        // X axis is a change of layout, never a loss of information.
                        title: function(tooltipItems) {
                            if (!tooltipItems.length) return '';
                            const item = tooltipItems[0];
                            const raw = item.raw || {};
                            const tth = (typeof raw.tth === 'number') ? raw.tth : xInv(item.parsed.x);
                            const lam = getLambda();
                            const st = Math.sin(tth * Math.PI / 360);
                            const dsp = st > 0 ? lam / (2 * st) : NaN;
                            const Q = 4 * Math.PI * st / lam;
                            const lines = [
                                `2\u03B8 ${tth.toFixed(4)}\u00B0   \u03B8 ${(tth / 2).toFixed(4)}\u00B0`,
                                `d ${isFinite(dsp) ? dsp.toFixed(5) : '-'} \u00C5   Q ${Q.toFixed(4)} \u00C5\u207B\u00B9`
                            ];
                            const di = item.datasetIndex;
                            if ((di === 1 || di === 2) && currentHklList && currentHklList.length) {
                                let best = null, bd = Infinity;
                                for (const hkl of currentHklList) {
                                    const diff = Math.abs(tth - hkl.tth);
                                    if (diff < bd) { bd = diff; best = hkl; }
                                }
                                // A calculated tick is its own line, so it must match
                                // essentially exactly; an observed peak only has to be
                                // inside the user's stated 2-theta error.
                                const tol = (di === 2) ? 1e-4 : getTthError();
                                if (best && bd < tol) {
                                    lines.push(`hkl (${best.h},${best.k},${best.l})`);
                                    if (di === 1) lines.push(`\u0394 from calc: ${(tth - best.tth).toFixed(4)}\u00B0`);
                                }
                            }
                            return lines;
                        },
                        label: function(context) {
                            const datasetLabel = context.dataset.label || '';
                            if (datasetLabel === 'Observed Peaks' || datasetLabel === 'Calculated Peaks') return null;
                            const raw = context.raw || {};
                            const I = (typeof raw.I === 'number') ? raw.I : yAx().inv(context.parsed.y);
                            return `Intensity: ${Math.round(I)}`;
                        }
                    }
                }
            }
        }
    });
};
const updateAllMarkers = () => {
    if (!xrdChart) return;
    // Filter in 2-theta, plot in whatever the axis currently is.
    const [tthLo, tthHi] = visibleTthRange();
    const yMin = xrdChart.scales.y.min; const yMax = xrdChart.scales.y.max;
    const yRange = yMax - yMin;
    const markerHeight = yRange * 0.02;

    const visibleObsPeaks = pickedPeaks.filter(p => p.tth >= tthLo && p.tth <= tthHi);
    xrdChart.data.datasets[1].data = visibleObsPeaks.map(p => ({ x: xF(p.tth), y: [yMin, yMin + markerHeight], tth: p.tth }));

    if (selectedSolution && currentHklList.length) {
        const calculatedBottom = yMin + markerHeight * 1.2;
        const calculatedTop = calculatedBottom + markerHeight;
        const visibleCalcPeaks = currentHklList.filter(hkl => hkl.tth >= tthLo && hkl.tth <= tthHi);
        xrdChart.data.datasets[2].data = visibleCalcPeaks.map(hkl => ({ x: xF(hkl.tth), y: [calculatedBottom, calculatedTop], tth: hkl.tth }));
    } else {
        xrdChart.data.datasets[2].data = [];
    }

    // Selected-peak indicator: stash the 2-theta on the chart instance; the
    // selectedPeakLine plugin converts it through the current axis on each
    // draw, which is exact (no bar-centering offset). undefined hides it.
    if (selectedPeakIndex !== null && pickedPeaks[selectedPeakIndex]) {
        xrdChart.$selectedPeakTth = pickedPeaks[selectedPeakIndex].tth;
    } else {
        xrdChart.$selectedPeakTth = undefined;
    }
    xrdChart.update('none');
};
// ---- Calculated line list ------------------------------------------------
// The measured pattern is plotted exactly as recorded, zero error and all.
// The refined cell, on the other hand, describes the sample AFTER the zero
// error has been taken out: 2theta_corrected = 2theta_observed - Z. So to
// put a calculated line next to the observed peak it explains, it has to be
// pushed back into the observed frame, 2theta_plot = 2theta_calc + Z.
// Skipping that step leaves every blue tick displaced from its red partner
// by exactly Z - small, uniform, and easily mistaken for a bad cell. The
// untouched value is kept as tth_calc for anything that wants the corrected
// frame instead.
const rebuildHklList = () => {
    if (!selectedSolution || !workingExperimentalData.tth.length) { currentHklList = []; return; }
    const lambda = getLambda();
    const zero = selectedSolution.zero_correction || 0;
    const dataMax = workingExperimentalData.tth[workingExperimentalData.tth.length - 1];
    // Generate past the end of the data by |Z| + 1 so that shifting the list
    // cannot leave the top of the pattern bare.
    const maxTth = Math.min(179.9, dataMax + Math.abs(zero) + 1);
    const raw = generateHKL(maxTth, { ...selectedSolution, lambda }, selectedSolution.system) || [];
    currentHklList = raw.map(r => ({ ...r, tth_calc: r.tth, tth: r.tth + zero }));
};
// Single entry point for "this solution is now the displayed one", so the
// line list can never drift out of step with the highlighted table row.
const applySolutionSelection = (sol) => {
    selectedSolution = sol || null;
    rebuildHklList();
    updateAllMarkers();
};
const rebuildPlot = (resetY = true) => {
    if (!workingExperimentalData || !workingExperimentalData.tth.length) return;
    initializeChart();
    rebuildHklList();
    updatePlotRange(resetY);
};
ui.xAxisMode.addEventListener('change', () => { xAxisMode = ui.xAxisMode.value; rebuildPlot(true); });
ui.yAxisMode.addEventListener('change', () => { yAxisMode = ui.yAxisMode.value; rebuildPlot(true); });
ui.snapMode.addEventListener('change', () => {
    if (xrdChart) { xrdChart.$snapTth = undefined; ui.snapReadout.textContent = ''; xrdChart.render(); }
});
