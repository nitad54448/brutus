// js/parsers/fullprof.js
// FullProf free-format .dat reader.
//
// Classic script, loaded in order by brutus.html (see the list there); its
// top-level names are shared with the other app scripts.

/**
     * Smart file detector. It checks for known headers and extensions
     * and falls back to a generic 2-column parser.
     */
    // --- FullProf free-format .dat -------------------------------------
    //
    //     !COMM  Y2O3 D8 0.3 0.3 0.1      (zero or more comment lines)
    //     15.00 0.01 145.90               (start, step, end in 2-theta)
    //        25   22   19   21  ...       (intensities, N per line)
    //
    // Detection is by CONTENT, never by extension. `.dat` is already the
    // generic whitespace 2-column fallback and is also one of Brutus's own
    // export formats, so keying on the extension would break every existing
    // 2-column .dat. The discriminator is the header triple together with
    // the value count: a file whose second line is three numbers AND that
    // then carries (end-start)/step + 1 values is this format and is very
    // unlikely to be anything else. A 2-column file fails on the triple; a
    // 3-column x/y/sigma file fails on the count.
    //
    // Left unguarded, such a file parses SILENTLY AND WRONGLY through the
    // generic parser: it reads "15.00 0.01" as a point, then "25 22" as the
    // next, producing a plausible-looking pattern made entirely of counts
    // plotted against counts.
    const scanFullProfDat = (lines) => {
        // Find the header: the first line that is exactly three numbers.
        // Written as a search rather than "skip comments, then read line 1"
        // because the preamble is not standardised -- some files open with
        // '!COMM ...', some with '#', some with a bare title line carrying
        // no marker at all, and some with nothing.
        const HEADER_SEARCH_LIMIT = 25;
        let i = -1, start = 0, step = 0, end = 0;
        for (let j = 0; j < lines.length && j < HEADER_SEARCH_LIMIT; j++) {
            const t = lines[j].trim();
            if (t === '') continue;
            const nums = t.split(/[\s,]+/).map(Number);
            if (nums.length !== 3 || !nums.every(Number.isFinite)) continue;
            if (!(nums[1] > 0) || !(nums[2] > nums[0])) continue;
            [start, step, end] = nums;
            i = j;
            break;
        }
        if (i === -1) return null;

        const nExpected = Math.round((end - start) / step) + 1;
        if (!(nExpected >= 2) || nExpected > 5e6) return null;

        // Every remaining token must be a number, or this is not the format.
        const values = [];
        const perLine = [];
        for (let j = i + 1; j < lines.length; j++) {
            const t = lines[j].trim();
            if (t === '') continue;
            if (t.startsWith('!') || t.startsWith('#')) continue;   // trailing comments
            const parts = t.split(/[\s,]+/);
            perLine.push(parts.length);
            for (const part of parts) {
                const v = Number(part);
                if (!Number.isFinite(v)) return null;
                values.push(v);
            }
        }
        if (values.length === 0) return null;

        // TWO independent signatures, either of which is decisive. Relying on
        // the count alone was too brittle: a real file whose header declares
        // a longer scan than it carries -- a truncated collection, a scan
        // stopped early -- was rejected and fell through to the generic
        // two-column reader, which then parsed it silently and wrongly.
        //
        //   (a) the count matches the header arithmetic; or
        //   (b) the data are PACKED, i.e. most lines carry four or more
        //       values. That alone separates this format from everything
        //       else that reaches here: two-column files have two per line
        //       and x/y/sigma files have three.
        const slack = Math.max(2, ...perLine);
        const countMatches = values.length >= nExpected - 1 && values.length <= nExpected + slack;

        const packedLines = perLine.filter(n => n >= 4).length;
        const isPacked = packedLines >= 2 && packedLines >= 0.8 * perLine.length;

        if (!countMatches && !isPacked) return null;

        return { start, step, end, nExpected, values, countMatches };
    };
    const parseFullProfDatFile = (text) => {
        const scan = scanFullProfDat(text.trim().split(/\r?\n/));
        if (!scan) throw new Error('Not a FullProf free-format .dat file.');

        // Trust the header for the count when the file over-runs it (padded
        // final line); trust the file when it carries fewer.
        const n = Math.min(scan.nExpected, scan.values.length);
        const intensity = scan.values.slice(0, n);
        const tth = Array.from({ length: n }, (_, k) => scan.start + k * scan.step);

        if (n < scan.nExpected) {
            console.warn(`FullProf .dat: header declares ${scan.nExpected} points ` +
                         `(${scan.start} to ${scan.end} step ${scan.step}), file carries ` +
                         `${scan.values.length}. Reading ${n}; the pattern ends at ` +
                         `${(scan.start + (n - 1) * scan.step).toFixed(4)} deg.`);
        }
        // No wavelength is recorded in this format, so none is returned and
        // the caller falls back to the Cu preset as it does for .xy.
        return { tth, intensity };
    };
