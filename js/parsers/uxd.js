// js/parsers/uxd.js
// Bruker UXD reader.
//
// Classic script, loaded in order by index.html (see the list there); its
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
        for (const [lineIndex, raw] of lines.entries()) {
            const line = raw.trim();
            if (line === '' || line.startsWith(';')) continue;
            const upper = line.toUpperCase();
            if (upper.startsWith('_')) {
                const dataKey = upper.slice(1).replace(/\s+/g, '');
                if (DATA_KEYS[dataKey]) {
                    current = { mode: DATA_KEYS[dataKey], cps: dataKey.endsWith('CPS'), header, tth: [], intensity: [] };
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
            const nums = line.split(/[\s,]+/).map(Number);
            if (!nums.every(Number.isFinite)) throw new Error(`UXD: invalid numeric data on line ${lineIndex + 1}.`);
            if (current.mode === 'xy') {
                if (nums.length % 2) throw new Error(`UXD: incomplete angle/intensity pair on line ${lineIndex + 1}.`);
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
        //
        // Raw counts of ranges measured with different step times are put on
        // the first range's time scale; joined as they are, a longer count
        // time at high angle shows as a step in intensity at the join. A
        // range without _STEPTIME inherits the previous one's.
        const notes = [];
        let stepTime = parseFloat(built[0].header.STEPTIME);
        const baseTime = stepTime;
        let tth = [...built[0].tth], intensity = [...built[0].intensity];
        for (let i = 1; i < built.length; i++) {
            const r = built[i];
            const t = parseFloat(r.header.STEPTIME);
            if (Number.isFinite(t)) stepTime = t;
            if (!(r.tth[0] > tth[tth.length - 1])) {
                notes.push(`UXD: range ${i + 1} of ${built.length} overlaps the previous one; it and any later ranges were not loaded.`);
                console.warn(notes[notes.length - 1]);
                break;
            }
            const scale = (!r.cps && !built[0].cps && baseTime > 0 && stepTime > 0 && stepTime !== baseTime) ? baseTime / stepTime : 1;
            if (scale !== 1) notes.push(`UXD: range ${i + 1} counted ${stepTime} s per step (range 1: ${baseTime} s); its counts were scaled to match.`);
            tth = tth.concat(r.tth);
            intensity = intensity.concat(scale === 1 ? r.intensity : r.intensity.map(y => y * scale));
        }
        return notes.length ? { tth, intensity, radiation, notes } : { tth, intensity, radiation };
    };
