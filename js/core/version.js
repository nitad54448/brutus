// js/core/version.js
// Cache-busting version shared by every runtime URL.
//
// Classic script, loaded in order by index.html (see the list there); its
// top-level names are shared with the other app scripts.

// The version is written once, in version.json. index.html reads it before loading any script
// (window.BRUTUS_VERSION) and stamps ?v=<version> on every script it loads; this file turns the
// same number into the query appended to every URL the app builds at runtime: both workers, the
// shaders and the space-group database. With no version (a test harness, an inlined build)
// nothing is busted, which is harmless.
//
// It matters because the page and the workers share the crystallography code:
// if they ever loaded different builds of it, solution keys and figures of
// merit would disagree across the worker boundary, which shows up as "the
// table and the report don't match" rather than as an error.
const APP_VERSION = (typeof window !== 'undefined' && typeof window.BRUTUS_VERSION === 'string') ? window.BRUTUS_VERSION : '';
const APP_VERSION_QS = APP_VERSION ? `?v=${encodeURIComponent(APP_VERSION)}` : '';
const APP_VERSION_LABEL = APP_VERSION ? `Brutus v${APP_VERSION}` : 'Brutus';
