// js/parsers/uxd.js
// Bruker UXD reader.
//
// Classic script, loaded in order by brutus.html (see the list there); its
// top-level names are shared with the other app scripts.

    
    // Bruker UXD.
    //   _KEY=value header lines (spaces around '=' are allowed), one or more
    //   ranges, each ending with a data section introduced by _COUNTS or
    //   _CPS (one intensity per value) or _2THETACOUNTS / _2THETACPS
    //   (2θ, intensity pairs). Each range keeps its own _START/_STEPSIZE.
    const parseUxdFile = (text) => {
        const lines = text.split(/\r?\n/);
        const radiation = {};
        const ranges = [];
        let header = {};      // keys seen since the last data section
        let current = null;   // the range whose data are being read
        const keyValue = (line) => {
            const m = /^_([A-Z0-9]+)\s*=\s*(.*)$/i.exec(line);
            return m ? [m[1].toUpperCase(), m[2].trim().replace(/^'|'$/g, '')] : null;
        };
        const DATA_KEYS = { COUNTS: 'y', CPS: 'y', '2THETACOUNTS': 'xy', '2THETACPS': 'xy' };
        for (const raw of lines) {
            const line = raw.trim();
            if (line === '' || line.startsWith(';')) continue;
            const upper = line.toUpperCase();
            if (upper.startsWith('_')) {
                const dataKey = upper.slice(1).replace(/\s+/g, '');
                if (DATA_KEYS[dataKey]) {
                    current = { mode: DATA_KEYS[dataKey], header, tth: [], intensity: [] };
                    ranges.push(current);
                    header = {};
                    continue;
                }
                const kv = keyValue(line);
                if (kv) {
                    // A header key after a data block starts the next range.
                    current = null;
                    const [k, v] = kv;
                    header[k] = v;
                    const num = parseFloat(v);
                    if (k === 'WL1' && Number.isFinite(num)) radiation.ka1 = num;
                    else if (k === 'WL2' && Number.isFinite(num)) radiation.ka2 = num;
                    else if (k === 'WLRATIO' && Number.isFinite(num)) radiation.ratio = num;
                }
                continue;
            }
            if (!current) continue;
            const nums = line.split(/[\s,]+/).map(Number).filter(Number.isFinite);
            if (current.mode === 'xy') {
                for (let i = 0; i + 1 < nums.length; i += 2) { current.tth.push(nums[i]); current.intensity.push(nums[i + 1]); }
            } else {
                current.intensity.push(...nums);
            }
        }

        const built = [];
        for (const r of ranges) {
            if (r.mode === 'y') {
                const start = parseFloat(r.header.START ?? r.header['2THETA']);
                const step = parseFloat(r.header.STEPSIZE);
                if (!Number.isFinite(start) || !Number.isFinite(step)) continue;
                r.tth = r.intensity.map((_, i) => start + i * step);
            }
            if (r.intensity.length) built.push(r);
        }
        if (ranges.length === 0) throw new Error("No _COUNTS / _CPS data section found in UXD file.");
        if (built.length === 0) throw new Error("Could not find _START and _STEPSIZE in UXD file.");

        // Several ranges are joined when they follow each other in 2θ;
        // overlapping ranges (re-measurements) cannot be merged blindly, so
        // only the first is kept.
        let tth = [...built[0].tth], intensity = [...built[0].intensity];
        for (let i = 1; i < built.length; i++) {
            const r = built[i];
            if (r.tth[0] > tth[tth.length - 1]) { tth = tth.concat(r.tth); intensity = intensity.concat(r.intensity); }
            else { console.warn(`UXD: range ${i + 1} overlaps the previous one and was not loaded.`); break; }
        }
        return { tth, intensity, radiation };
    };
