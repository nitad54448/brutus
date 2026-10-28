// js/parsers/rigaku-ras.js
// Rigaku RAS reader.
//
// Classic script, loaded in order by index.html (see the list there); its
// top-level names are shared with the other app scripts.

    // Rigaku RAS. Header values are usually quoted ("1.540593"), which
    // parseFloat cannot read without stripping the quotes first.
    const parseRigakuRasFile = (text) => {
        const lines = text.trim().split(/\r?\n/);
        const tth = [], intensity = [];
        const radiation = {};
        let hasAttenuation = false;
        let inDataSection = false;
        const headerValue = (line) => {
            const m = /^\*\S+\s+(.*)$/.exec(line.trim());
            if (!m) return null;
            const v = parseFloat(m[1].trim().replace(/^"|"$/g, ''));
            return Number.isFinite(v) ? v : null;
        };
        for (const line of lines) {
            const upperLine = line.trim().toUpperCase();
            if (!inDataSection) {
                if (upperLine.startsWith('*HW_XG_WAVE_LENGTH_ALPHA1')) radiation.ka1 = headerValue(line);
                else if (upperLine.startsWith('*HW_XG_WAVE_LENGTH_ALPHA2')) radiation.ka2 = headerValue(line);
                else if (upperLine.startsWith('*WAVE_LENGTH') || upperLine.startsWith('*MEAS_COND_XG_WAVE_LENGTH')) {
                    const v = headerValue(line);
                    if (v !== null) radiation.single = v;
                }
            }
            if (upperLine.startsWith('*RAS_INT_START')) { inDataSection = true; continue; }
            if (upperLine.startsWith('*RAS_INT_END')) break;
            if (inDataSection) {
                const parts = line.trim().split(/[\s,]+/);
                if (parts.length >= 2) {
                    const x = parseFloat(parts[0]);
                    const y = parseFloat(parts[1]);
                    // GSAS-II preserves column 2 and does not apply column 3.
                    // Without a documented correction state, multiplying an
                    // attenuator coefficient can correct the same data twice.
                    const att = parts.length >= 3 ? parseFloat(parts[2]) : 1;
                    if (Number.isFinite(att) && att !== 1) hasAttenuation = true;
                    if (Number.isFinite(x) && Number.isFinite(y)) { tth.push(x); intensity.push(y); }
                }
            }
        }
        if (tth.length === 0) throw new Error("No data found in RAS file data section.");
        const result = { tth, intensity, radiation };
        if (hasAttenuation) result.notes = ['RAS: non-unit attenuator factors are present; stored intensities were preserved, following GSAS-II.'];
        return result;
    };
