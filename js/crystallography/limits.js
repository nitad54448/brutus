// Shared policy limits and defaults. Loaded first on the page and in both workers.
// Units: lengths in angstroms, volumes in A^3, angles in degrees, times in ms.
// Mathematical constants, matrix dimensions and WGSL buffer layouts stay with
// their algorithms; changing those is a code/protocol change, not configuration.
const BRUTUS_LIMITS = (() => {
    const minimumMcIterations = 200;
    const definitions = {
        physical: { minAxis: 2, maxAxis: 50, minVolume: 20, minAngle: 10, maxAngle: 170,
            maxMonoclinicBeta: 150, maxPlotTth: 179.9,
            gpuMinTriclinicAngle: 60, gpuMaxTriclinicAngle: 150 },
        meritResidualFloor: 1e-12,
        symmetryTolerance: 0.25,
        tetragonalAxisTolerance: 0.02,
        derivativeRelativeStep: 1e-5,
        derivativeAbsoluteStep: 1e-7,
        exportUniformGridToleranceDeg: 1e-8,
        search: { meritPeaks: 20, minM20: 2, minMixedHkl: 40, minBasis: 10, minPeaks: 3,
            minIndexed: { cubic: 4, tetragonal: 5, hexagonal: 5, orthorhombic: 6, monoclinic: 7, triclinic: 7 },
            cpuPeakDepth: 12, progressTrials: 5000, progressIntervalMs: 50,
            defaultGpuFomPeaks: 7, minGpuFomPeaks: 10 },
        hkl: { monoclinic: 6, monoclinicAxial: 12, triclinic: 5, orthorhombic: 12, highSymH: 12, cpuCubicH: 40,
            highSymL: 24, highSymS: 144, cubicH: 30, cubicN: 900, lineMergeToleranceDeg: 1e-4 },
        niggli: { maxIterations: 1000, initialEpsilon: 1e-5, relaxAfter: 50, relaxAgainAfter: 200,
            relaxedEpsilon: 1e-4, finalEpsilon: 1e-3, minEquivalentAngle: 60, maxEquivalentAngle: 150 },
        solutions: { reservoirMax: 500, reservoirTrim: 400, displayCount: 50,
            finalCount: 50, symmetryM20Ratio: 2 },
        gpu: { threadsPerDispatch: 500000, triclinicThreadsPerDispatch: 50000,
            chunkWarn: 100000, chunkHard: 2000000, chunkLog: 200000,
            defaultCandidates: 20000, defaultWorkgroupsPerDimension: 65535, workgroupsYCap: 16383,
            yieldIntervalMs: 16, chunksPerSync: 8 },
        workers: { maxPool: 12, reservedThreads: 2, defaultConcurrency: 4,
            ledgerMax: 20000, ledgerTrim: 15000, flushSize: 64,
            maxErrorsPerBatch: 5, heartbeatMs: 2000, stallMs: 30000, pollMs: 250 },
        workspace: { logEntries: 1000, historyWarnBytes: 100 * 1024 * 1024,
            projectWarnBytes: 10 * 1024 * 1024, reportPageWarn: 100, fileMaxMB: 50 },
        peak: { minThreshold: 0.1, maxThreshold: 20, initialThreshold: 2, logIntensityFloor: 0.1, ka2Terms: 8,
            positionToleranceFloor: 0.02, positionToleranceCeiling: 0.10,
            snapWindowDeg: 0.35, wavelengthMatchTolerance: 3e-4,
            ka2RatioMaxFactor: 1.7, ka2RatioMinFactor: 0.35 },
        absences: { maxViolations: 2, lowIntensityFraction: 0.05, localWindowDeg: 15,
            listCap: 40, listCeiling: 150, ambiguityMargin: 2,
            detailsToStore: 2, consensusMinCount: 5, consensusMinFraction: 0.15,
            minVerified: 1, minVerifiedStrong: 2, chanceLevel: 0.05, extinctionWeight: 0.5,
            probeRange: 8, signatureRange: 6, decisiveNats: 2.3,
            cleanCapMultiplier: 20, cleanLeak: 1e-3, scoreTieNats: 0.5,
            wilsonMinPeaks: 12, wilsonMinShells: 3,
            wilsonMinBins: 4, wilsonMaxBins: 12, wilsonPointsPerBin: 8, wilsonFitIterations: 60,
            epsilon: { weak: 0.25, ka2: 0.5, ambiguous: 0.5, unindexed: 0.15,
                hardFloor: 1e-4, hardCap: 0.05 } },
        swap: { MAX_PEAKS: 1e9, ALT_WINDOW: 2.5, MAX_ALT: 4, MAX_FREE: 8,
            MAX_FITS: 150, MAX_EVALS: 16, MAX_POST: 3, KEEP_RATIO: 1,
            LOSE_SLACK: 1, ROUNDS: 4, ROUND_GAIN: 1.02, TOP_N: 40, DEBUG: false },
        postProcessBudgets: [
            { maxParents: 8, config: { MAX_FITS: 600, MAX_FREE: 16, MAX_EVALS: 32, MAX_POST: 5, ROUNDS: 6 } },
            { maxParents: 20, config: { MAX_FITS: 300, MAX_FREE: 12, MAX_EVALS: 24, ROUNDS: 5 } },
        ],
        monteCarlo: { iterations: 400, range: 0.01, temperature: 0.05,
            cooling: 0.995, zeroStep: 0.01, zeroMax: 0.20,
            restarts: 2, refineIterations: 700, minIterations: minimumMcIterations, polishZeroMax: 1 },
        parsers: { fullprofHeaderSearchLines: 25, fullprofMaxPoints: 5e6,
            fullprofPackedMinColumns: 4, fullprofPackedMinLines: 2, fullprofPackedFraction: 0.8,
            formatSniffChars: 500, genericWarningMinPoints: 10,
            minTypicalStepDeg: 1e-4, maxTypicalStepDeg: 0.2, positiveFraction: 0.5,
            descendingFraction: 0.8, suspiciousFraction: 0.2,
            minHeaderWavelength: 0.1, maxHeaderWavelength: 5 },
        conversion: { axisRelativeTolerance: 0.004, angleToleranceDeg: 0.12,
            matrixCandidates: 600, namingTimeMs: 30000, maxMisfit: 0.03, resultLimit: 8 },
        dialogs: { sgMinPeaks: 6, sgStageChunk: 4, sgShortlistExtra: 4 },
        inputs: {
            wavelength: { value: 1.54184, min: 0.1, step: 0.00001 },
            'max-volume': { value: 2000, min: 25, max: 12500, step: 100 },
            'impurity-peaks': { value: 1, min: 0, max: 3, step: 1 },
            'tth-error': { value: 0.04, min: 0.0001, step: 0.002 },
            'peak-threshold-slider': { value: 56, min: 0, max: 100 },
            'peak-prominence-slider': { value: 25, min: 0, max: 90, step: 5 },
            'ball-radius-slider': { value: 0.100, min: 0.005, max: 0.300, step: 0.005 },
            'smoothing-width-slider': { value: 0.010, min: 0, max: 0.060, step: 0.002 },
            'gpu-hkl-percent': { value: 5, min: 1, max: 16, step: 0.5 },
            'gpu-depth': { value: 3, min: 0, max: 12, step: 1 },
            'gpu-fom-threshold': { value: 1.25, min: 0.10, max: 5, step: 0.1 },
            'gpu-buffer-size': { value: 100, min: 1, max: 500, step: 1 },
            'mc-iterations': { value: 1000, min: minimumMcIterations, max: 20000, step: 50 },
            'mc-restarts': { value: 8, min: 1, max: 20, step: 1 },
            'sg-iterations': { value: 600, min: minimumMcIterations, max: 20000, step: 50 },
            'sg-restarts': { value: 4, min: 1, max: 20, step: 1 },
            'sg-top': { value: 8, min: 1, max: 50, step: 1 },
        },
    };
    const freeze = object => {
        for (const value of Object.values(object)) if (value && typeof value === 'object') freeze(value);
        return Object.freeze(object);
    };
    return freeze(definitions);
})();

// Per-run overrides have the same meaning on the CPU, in MC and on the GPU.
function getPhysicalLimits(data = {}) {
    const defaults = BRUTUS_LIMITS.physical;
    return {
        minAxis: Number.isFinite(data.min_axis) ? data.min_axis : defaults.minAxis,
        maxAxis: Number.isFinite(data.max_axis) ? data.max_axis : defaults.maxAxis,
        minVolume: Number.isFinite(data.min_volume) ? data.min_volume : defaults.minVolume,
        minAngle: defaults.minAngle, maxAngle: defaults.maxAngle,
    };
}
