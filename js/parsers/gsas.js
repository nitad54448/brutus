// js/parsers/gsas.js
// GSAS raw powder data reader (STD / ESD / FXY / FXYE).
//
// Classic script, loaded in order by brutus.html (see the list there); its
// top-level names are shared with the other app scripts.

    // --- GSAS raw powder data ------------------------------------------
    //
    //   line 1   title
    //   BANK IBANK NCHAN NREC BINTYP BCOEF1 BCOEF2 BCOEF3 BCOEF4 [TYPE]
    //   records, laid out according to TYPE:
    //     STD (the default, no TYPE)   10(I2,F6.0)   NCTR, Y
    //     ESD                          10F8.0        Y, sigma(Y)
    //     FXY / FXYE                   free format   X, Y [, sigma]
    //
    // With BINTYP CONST (or CONS), BCOEF1/BCOEF2 are start and step in
    // centidegrees; FXY(E) records give X explicitly, also in centidegrees.
    // Only the first bank is read.
    //
    // Replaces two readers that were each wrong in their own way. The ESD one
    // took tokens 1, 3, 5... -- the SIGMA column -- so a real ESD file loaded
    // as roughly sqrt(I): peak positions survived, intensities did not. The
    // STD one took every token, so a non-blank NCTR became a spurious data
    // point; and a BANK line with no TYPE (i.e. STD) went to the ESD reader.
    //
    // Fields are read by COLUMN first, as the format defines them: a large
    // value can fill its 8 characters and touch its neighbour, which
    // whitespace splitting cannot undo. Fortran right-justifies numbers, so if
    // every field parses cleanly in place the layout is confirmed; otherwise
    // (a free-format file) whitespace tokens are used.
    const parseGsasFile = (text) => {
        const lines = text.split(/\r?\n/);
        const bankAt = lines.findIndex(l => /^\s*BANK\b/i.test(l));
        if (bankAt < 0) throw new Error("GSAS Parse Error: no BANK line.");
        const bank = lines[bankAt].trim().split(/\s+/);
        const binTyp = (bank[4] || '').toUpperCase();
        if (binTyp !== 'CONST' && binTyp !== 'CONS') {
            throw new Error(`GSAS Parse Error: BINTYP '${bank[4] || ''}' is not supported (constant-step 2θ data only).`);
        }
        const type = bank.slice(5).map(t => t.toUpperCase())
            .find(t => ['STD', 'ESD', 'ALT', 'FXY', 'FXYE'].includes(t)) || 'STD';
        if (type === 'ALT') throw new Error("GSAS Parse Error: ALT (compressed) records are not supported.");

        let wavelength = null;
        for (const l of lines) {
            const m = /wavelength\s+([0-9.]+)/i.exec(l);
            if (m && isFinite(parseFloat(m[1]))) wavelength = parseFloat(m[1]);
        }

        // Records: everything after the BANK line up to the next BANK, minus
        // blank, comment and marker lines (any letter other than an exponent's
        // d/e means the line is not data).
        const recs = [];
        for (let i = bankAt + 1; i < lines.length; i++) {
            const l = lines[i].replace(/\s+$/, '');
            if (/^\s*BANK\b/i.test(l)) break;
            if (l === '' || /^\s*#/.test(l) || /[a-cf-z]/i.test(l)) continue;
            recs.push(l);
        }
        if (recs.length === 0) throw new Error("GSAS Parse Error: no data records after the BANK line.");

        const NUM = /^ *[+-]?(\d+\.?\d*|\.\d+)([eEdD][+-]?\d+)?$/;   // right-justified number
        const toNum = s => parseFloat(s.trim().replace(/[dD]/, 'e'));
        const tokens = r => r.trim().split(/\s+/).map(toNum);

        let tth = [], intensity = [];
        if (type === 'FXY' || type === 'FXYE') {
            for (const r of recs) {
                const t = tokens(r);
                if (t.length >= 2 && Number.isFinite(t[0]) && Number.isFinite(t[1])) {
                    tth.push(t[0] / 100); intensity.push(t[1]);
                }
            }
        } else {
            const start = parseFloat(bank[5]) / 100, step = parseFloat(bank[6]) / 100;
            if (!(Number.isFinite(start) && Number.isFinite(step) && step > 0)) {
                throw new Error("GSAS Parse Error: invalid CONST start/step on the BANK line.");
            }
            // Column read: `width` characters per record field; pick() returns
            // the intensity text, or null if the field breaks the layout.
            const byColumns = (width, pick) => {
                const out = [];
                for (const r of recs) {
                    for (let c = 0; c < r.length; c += width) {
                        const f = r.slice(c, c + width);
                        if (f.trim() === '') break;          // unused trailing fields
                        const y = pick(f);
                        if (y === null) return null;
                        out.push(toNum(y));
                    }
                }
                return out;
            };
            if (type === 'ESD') {
                // (Y, sigma) pairs, 16 characters each, intensity FIRST.
                intensity = byColumns(16, f => {
                    const y = f.slice(0, 8), e = f.slice(8);
                    return NUM.test(y) && (e.trim() === '' || NUM.test(e)) ? y : null;
                });
                if (!intensity) {
                    // Free format. Files saved by Brutus before this fix hold
                    // (sigma, Y) in 12-character columns, which the column test
                    // rejects; read those the old way round so they still load.
                    const parity = /^\s*Exported by Brutus/i.test(lines[0] || '') ? 1 : 0;
                    intensity = recs.flatMap(r => tokens(r).filter((_, j) => j % 2 === parity));
                }
            } else {
                // STD: NCTR in the first 2 characters of each 8, Y in the other 6.
                intensity = byColumns(8, f => {
                    const nctr = f.slice(0, 2), y = f.slice(2);
                    return (nctr.trim() === '' || /^ *\d+$/.test(nctr)) && NUM.test(y) ? y : null;
                }) || recs.flatMap(tokens);
            }
            intensity = intensity.filter(Number.isFinite);
            tth = intensity.map((_, i) => start + i * step);
        }

        // NCHAN is the declared point count: drop padding past it, warn if short.
        const nChan = parseInt(bank[2], 10);
        if (Number.isInteger(nChan) && nChan > 0) {
            if (intensity.length > nChan) { intensity = intensity.slice(0, nChan); tth = tth.slice(0, nChan); }
            else if (intensity.length < nChan) console.warn(`GSAS: BANK declares ${nChan} points, file carries ${intensity.length}.`);
        }
        if (intensity.length === 0) throw new Error("GSAS Parse Error: no intensity data could be parsed.");
        return { tth, intensity, wavelength };
    };
