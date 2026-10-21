// js/core/version.js
// Cache-busting version shared by every runtime URL.
//
// Classic script, loaded in order by index.html (see the list there); its
// top-level names are shared with the other app scripts.

// Cache-busting query for every URL the app builds at runtime: both workers,
// the shaders and the space-group database. It is read off this script's own
// <script> tag, so the single ?v= that bump_version.py stamps into index.html
// reaches all of them. With no query (a test harness, an inlined build) nothing
// is busted, which is harmless.
//
// It matters because the page and the workers share the crystallography code:
// if they ever loaded different builds of it, solution keys and figures of
// merit would disagree across the worker boundary, which shows up as "the
// table and the report don't match" rather than as an error.
const APP_VERSION_QS = (() => {
    try {
        const src = (document.currentScript && document.currentScript.src) || '';
        const i = src.indexOf('?');
        return i >= 0 ? src.slice(i) : '';
    } catch (_) {
        return '';
    }
})();
// Display exactly the build identifier used by the page and its workers,
// including suffixes such as "b" for multiple builds on the same date.
const APP_VERSION = new URLSearchParams(APP_VERSION_QS).get('v') || '';
const APP_VERSION_LABEL = APP_VERSION ? `Brutus, v${APP_VERSION}` : 'Brutus';
