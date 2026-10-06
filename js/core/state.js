// js/core/state.js
// Session state shared by the UI modules (data, peaks, solutions, run bookkeeping).
//
// Classic script, loaded in order by brutus.html (see the list there); its
// top-level names are shared with the other app scripts.

// data, si Ka stripped ou pas, on copie les données
let fullExperimentalData = { tth: [], intensity: [] }; // The original, unmodified data
let loadedFileName = ''; // basename of the currently loaded file, for Save-as naming
let workingExperimentalData = { tth: [], intensity: [] }; // The data to be plotted and analyzed (raw or stripped)
let pickedPeaks = [];
// True once the user has hand-edited the peak list (added, deleted, moved
// or confirmed a peak) since the last automatic detection. findPeaks()
// REPLACES pickedPeaks wholesale, so anything that re-runs it without the
// user asking would silently discard that work. Only the peak-finding
// controls themselves are allowed to do that; see the wavelength handler.
let peaksManuallyEdited = false;
// Index of the peak whose row in the side table is currently being
// edited (focused or clicked). When non-null, updateAllMarkers draws a
// tall translucent vertical line across the whole plot at that peak's
// 2θ so the user can immediately see which peak in the diffractogram
// they are editing. Cleared on blur or click outside.
let selectedPeakIndex = null;
let taskProgress = [];
let taskTotals = [];
let lastDurationStr = "";
let solutions = [];
let displayedSolutions = [];
let selectedSolution = null;
let currentHklList = [];
let foundSolutionMap = new Map();
let xrdChart;
let isIndexing = false;
let gpuStopSignal = { stop: false };
let lastIndexingStats = ''; // Stores the final trial count and speed
let cumulativeTrials = 0;
let gpuTotalTrials = 0;
let indexingStartTime = 0;
let activeWorkers = [];
let resolveWorkerTask = null;
// Same role as resolveWorkerTask, for the post-processing worker.
// abortActiveIndexing() terminates that worker, which fires neither 'done'
// nor onerror, so without this the `await new Promise(...)` wrapping it in
// startIndexing() never settles: the async run body is pinned forever,
// holding its closure (peaks, solutions, baseParams) alive for the rest of
// the session and never reaching finalizeIndexing().
let resolvePostProcessTask = null;
// Bumped by abortActiveIndexing() (Stop button, or loading a new file
// while indexing). startIndexing() captures the value at its own start;
// finalizeIndexing() compares against the current value to detect that
// its run was aborted/superseded and should no-op instead of overwriting
// fresher state. See abortActiveIndexing() and finalizeIndexing().
let indexingRunToken = 0;
let indexingFailures = [];
const recordIndexingFailure = (message, token) => {
    if (token !== indexingRunToken) return;
    if (!indexingFailures.includes(message)) indexingFailures.push(message);
    showStatus(`Search incomplete: ${message}`, 'error', 10000);
};
let sortState = { column: 'm20', direction: 'desc' };
let workerURL = null;
// In-run reservoir size. This used to be 50/40, which meant any cell outside
// the top 40 by M20 *at the moment it arrived* was destroyed before
// applyFinalSieve, the post-process worker, or space-group analysis ever saw
// it. On a GPU run producing tens of thousands of candidates that is a very
// early, very lossy cut: a strong cell arriving late could be dropped simply
// because the list was momentarily full of near-duplicates of each other.
// The list is deduped by _solKey on insert and sieved at the end anyway, so
// a larger reservoir costs a few hundred small objects and nothing else.
const MAX_SOLUTIONS_BEFORE_PRUNING = 500;
const PRUNE_TO_COUNT = 400; // Prune down to this many
// The table is rebuilt from scratch on every animation frame during a run,
// so the number of RENDERED rows is what actually costs time -- not the
// number retained. Keep the display at the old order of magnitude.
const MAX_DISPLAYED_SOLUTIONS = 50;
// 1. Declare the throttle flag right above the function
let isTableUpdateScheduled = false;
//  systematic absences
const max_hkl_analysis = 10;
