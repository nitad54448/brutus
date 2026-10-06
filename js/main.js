// js/main.js
// Startup. Runs last, after every module has defined what it owns.
//
// Classic script, loaded in order by brutus.html (see the list there); its
// top-level names are shared with the other app scripts.

setupWorker();
handleWavelengthPresetChange({ onLoad: true });
// Startup work that reports through showStatus and the UI helpers. It used
// to run near the top of this file, before `const showStatus` was
// initialised: on a browser without navigator.gpu the catch block hit the
// TDZ, threw a ReferenceError, and never disabled the GPU checkboxes.
loadSpaceGroupData();
checkWebGPUCapabilities();
window.addEventListener('beforeunload', () => { if (workerURL) URL.revokeObjectURL(workerURL); });
