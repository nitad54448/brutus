// js/crystallography/manifest.js
// The crystallography scripts, in load order. brutus.html includes the same
// files as <script> tags (check_load_order.mjs verifies that the two lists
// agree); the workers load them with importBrutusCrystallography().
const BRUTUS_CRYSTALLOGRAPHY_FILES = [
    'metric.js',
    'hkl.js',
    'cell.js',
    'fit.js',
    'refine.js',
    'transforms.js',
    'monte-carlo.js',
    'niggli.js',
    'absences.js',
    'centering.js',
    'sg-ranking.js',
    'swap-search.js',
    'sg-ops.js',
    'sg-context.js',
    'sg-score.js',
];

// Worker-side loader. importScripts resolves relative to the worker script,
// so the files are addressed from the worker's location (js/workers/), and
// the worker's own ?v= is forwarded so page and workers run the same build.
function importBrutusCrystallography() {
    const query = (self.location && self.location.search) || '';
    const base = new URL('../crystallography/', self.location.href);
    importScripts(...BRUTUS_CRYSTALLOGRAPHY_FILES.map(f => new URL(f + query, base).href));
}
