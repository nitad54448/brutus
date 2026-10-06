// js/parsers/pdcif.js
// pdCIF (powder CIF) reader.
//
// Classic script, loaded in order by brutus.html (see the list there); its
// top-level names are shared with the other app scripts.

    // pdCIF (powder CIF). Reads the first loop that carries a 2θ column
    // and an intensity column, or -- for files that give the scan as a
    // range -- an intensity loop together with _pd_meas_2theta_range_*.
    // Values may carry s.u.'s in parentheses ("1234(35)"); loop rows may
    // wrap across lines, so loops are read token by token.
    const parsePdCifFile = (text) => {
        const lines = text.split(/\r?\n/);
        const items = {};          // single-valued tags
        const loops = [];          // { tags: [...], values: [...] }
        let loop = null;
        let inTextField = false;
        const tokenize = (line) => {
            const out = [];
            const re = /'([^']*)'|"([^"]*)"|(\S+)/g;
            let m;
            while ((m = re.exec(line)) !== null) out.push(m[1] ?? m[2] ?? m[3]);
            return out;
        };
        for (const raw of lines) {
            if (raw.startsWith(';')) { inTextField = !inTextField; continue; }   // multi-line text field
            if (inTextField) continue;
            const line = raw.trim();
            if (line === '' || line.startsWith('#')) continue;
            if (/^loop_$/i.test(line)) { loop = { tags: [], values: [] }; loops.push(loop); continue; }
            if (/^data_/i.test(line)) { loop = null; continue; }
            if (line.startsWith('_')) {
                const toks = tokenize(line);
                const tag = toks[0].toLowerCase();
                if (loop && loop.values.length === 0 && toks.length === 1) { loop.tags.push(tag); continue; }
                loop = null;
                if (toks.length > 1) items[tag] = toks[1];
                else items[tag] = null;   // value on a following line; not needed here
                continue;
            }
            if (loop) loop.values.push(...tokenize(line));
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
            const wl = column(wlLoop, '_diffrn_radiation_wavelength').filter(Number.isFinite);
            const wt = column(wlLoop, '_diffrn_radiation_wavelength_wt');
            if (wl.length >= 2) {
                const order = wl.map((v, k) => [v, wt ? wt[k] : NaN]).sort((x, y) => x[0] - y[0]);
                radiation.ka1 = order[0][0];
                radiation.ka2 = order[1][0];
                if (Number.isFinite(order[0][1]) && Number.isFinite(order[1][1]) && order[0][1] > 0) {
                    radiation.ratio = order[1][1] / order[0][1];
                }
            } else if (wl.length === 1) {
                radiation.single = wl[0];
            }
        } else if (Number.isFinite(cifNumber(items['_diffrn_radiation_wavelength']))) {
            radiation.single = cifNumber(items['_diffrn_radiation_wavelength']);
        }
        return { tth: tthOut, intensity: intOut, radiation };
    };
