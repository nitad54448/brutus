// js/parsers/pdcif.js
// pdCIF (powder CIF) reader.
//
// Classic script, loaded in order by index.html (see the list there); its
// top-level names are shared with the other app scripts.

    // pdCIF (powder CIF). Reads the first loop that carries a 2θ column
    // and an intensity column, or -- for files that give the scan as a
    // range -- an intensity loop together with _pd_meas_2theta_range_*.
    // Values may carry s.u.'s in parentheses ("1234(35)"); loop rows may
    // wrap across lines, so loops are read token by token.
    const parsePdCifFile = (text) => {
        const items = {};          // single-valued tags
        const loops = [];          // { tags: [...], values: [...] }
        // Keep quoted/text-field tokens distinct from syntax. Comments start
        // only outside strings, and a scalar value may be on the next line.
        const tokens = [];
        const source = text.replace(/\r\n?/g, '\n');
        for (let i = 0; i < source.length;) {
            if (/\s/.test(source[i])) { i++; continue; }
            if (source[i] === '#') { while (i < source.length && source[i] !== '\n') i++; continue; }
            if (source[i] === ';' && (i === 0 || source[i - 1] === '\n')) {
                const end = source.indexOf('\n;', i + 1);
                if (end < 0) throw new Error('pdCIF: unterminated text field.');
                tokens.push({ value: source.slice(i + 1, end), quoted: true });
                i = end + 2;
            } else if (source[i] === "'" || source[i] === '"') {
                const quote = source[i++], start = i;
                while (i < source.length && !(source[i] === quote && (i + 1 === source.length || /\s/.test(source[i + 1])))) i++;
                if (i === source.length) throw new Error('pdCIF: unterminated quoted value.');
                tokens.push({ value: source.slice(start, i++), quoted: true });
            } else {
                const start = i;
                while (i < source.length && !/\s/.test(source[i])) i++;
                tokens.push({ value: source.slice(start, i), quoted: false });
            }
        }
        const isTag = token => token && !token.quoted && token.value.startsWith('_');
        const isSyntax = token => token && !token.quoted &&
            (isTag(token) || /^(?:loop_|stop_|global_|data_.*|save_.*)$/i.test(token.value));
        for (let i = 0; i < tokens.length;) {
            const token = tokens[i++];
            if (!token.quoted && /^loop_$/i.test(token.value)) {
                const loop = { tags: [], values: [] };
                while (isTag(tokens[i])) loop.tags.push(tokens[i++].value.toLowerCase());
                if (!loop.tags.length) throw new Error('pdCIF: loop has no column tags.');
                while (tokens[i] && !isSyntax(tokens[i])) loop.values.push(tokens[i++].value);
                if (loop.values.length % loop.tags.length) throw new Error('pdCIF: incomplete data loop row.');
                loops.push(loop);
            } else if (isTag(token)) {
                if (!tokens[i] || isSyntax(tokens[i])) throw new Error(`pdCIF: missing value for ${token.value}.`);
                items[token.value.toLowerCase()] = tokens[i++].value;
            }
        }
        const cifNumber = (v) => {
            if (v === undefined || v === null) return NaN;
            return parseFloat(String(v).replace(/\(.*\)$/, ''));
        };
        const column = (lp, tag) => {
            const i = lp.tags.indexOf(tag);
            if (i < 0) return null;
            const n = lp.tags.length, out = [];
            for (let r = 0; r + n <= lp.values.length; r += n) out.push(cifNumber(lp.values[r + i]));
            return out;
        };
        const TTH_TAGS = ['_pd_meas_2theta_scan', '_pd_proc_2theta_corrected', '_pd_meas_angle_2theta', '_pd_proc_2theta_scan'];
        const INT_TAGS = ['_pd_meas_counts_total', '_pd_meas_intensity_total', '_pd_proc_intensity_total',
                          '_pd_proc_intensity_net', '_pd_meas_intensity_net', '_pd_meas_counts_net'];
        let tth = null, intensity = null;
        for (const lp of loops) {
            const tTag = TTH_TAGS.find(t => lp.tags.includes(t));
            const iTag = INT_TAGS.find(t => lp.tags.includes(t));
            if (!iTag) continue;
            const y = column(lp, iTag);
            if (tTag) { tth = column(lp, tTag); intensity = y; break; }
            // Range-described scan: _pd_meas_2theta_range_min / _max / _inc.
            const pre = ['_pd_meas_2theta_range_', '_pd_proc_2theta_range_']
                .find(p => Number.isFinite(cifNumber(items[p + 'min'])) && Number.isFinite(cifNumber(items[p + 'inc'])));
            if (pre) {
                const start = cifNumber(items[pre + 'min']), inc = cifNumber(items[pre + 'inc']);
                tth = y.map((_, k) => start + k * inc);
                intensity = y;
                break;
            }
        }
        if (!tth || !intensity) {
            throw new Error("Could not find _pd_meas_2theta_scan and intensity data in pdCIF file.");
        }
        const keep = tth.map((t, k) => Number.isFinite(t) && Number.isFinite(intensity[k]));
        const tthOut = tth.filter((_, k) => keep[k]);
        const intOut = intensity.filter((_, k) => keep[k]);
        if (tthOut.length === 0) throw new Error("pdCIF data loop contains no numeric points.");

        // Radiation: either one wavelength, or a looped list (doublet) with
        // optional weights. The doublet is Kα1 + Kα2; its ratio is the
        // weight of the longer line relative to the shorter one.
        const radiation = {};
        const wlLoop = loops.find(lp => lp.tags.includes('_diffrn_radiation_wavelength'));
        if (wlLoop) {
            const wl = column(wlLoop, '_diffrn_radiation_wavelength');
            const wt = column(wlLoop, '_diffrn_radiation_wavelength_wt');
            const order = wl.map((v, k) => [v, wt ? wt[k] : NaN])
                .filter(([v]) => Number.isFinite(v)).sort((x, y) => x[0] - y[0]);
            if (order.length >= 2) {
                radiation.ka1 = order[0][0];
                radiation.ka2 = order[1][0];
                if (Number.isFinite(order[0][1]) && Number.isFinite(order[1][1]) && order[0][1] > 0) {
                    radiation.ratio = order[1][1] / order[0][1];
                }
            } else if (order.length === 1) {
                radiation.single = order[0][0];
            }
        } else if (Number.isFinite(cifNumber(items['_diffrn_radiation_wavelength']))) {
            radiation.single = cifNumber(items['_diffrn_radiation_wavelength']);
        }
        return { tth: tthOut, intensity: intOut, radiation };
    };
