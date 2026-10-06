// js/parsers/philips-udf.js
// Philips UDF reader.
//
// Classic script, loaded in order by brutus.html (see the list there); its
// top-level names are shared with the other app scripts.

    
    const parsePhilipsUdfFile = (text) => {
const lines = text.trim().split(/\r?\n/);
const isRawScan = lines.some(l => l.trim().toUpperCase() === 'RAWSCAN');

if (isRawScan) {
    let startTth, endTth, stepSize, wavelength = null, ka2 = null, ratio = null;
    let inDataSection = false;
    const intensity = [];

    for (const line of lines) {
        const trimmedLine = line.trim();
        if (!inDataSection) {
            const upper = trimmedLine.toUpperCase();
            if (upper === 'RAWSCAN') { inDataSection = true; continue; }
            const parts = trimmedLine.split(',').map(p => p.trim());
            const key = parts[0].toUpperCase();
            if (key === 'DATAANGLERANGE') {
                startTth = parseFloat(parts[1]);
                endTth = parseFloat(parts[2]);
            } else if (key === 'SCANSTEPSIZE') {
                stepSize = parseFloat(parts[1]);
            } else if (key === 'LABDAALPHA1') {
                wavelength = parseFloat(parts[1]);
            } else if (key === 'LABDAALPHA2') {
                ka2 = parseFloat(parts[1]);
            } else if (key === 'RATIOALPHA21') {
                ratio = parseFloat(parts[1]);
            }
        } else {
            trimmedLine.split(',').forEach(part => {
                const val = parseFloat(part.trim());
                if (!isNaN(val)) intensity.push(val);
            });
        }
    }

    if (intensity.length === 0) throw new Error("No intensity data found after RawScan in UDF file.");
    if (startTth === undefined || stepSize === undefined) throw new Error("Could not find DataAngleRange/ScanStepSize in UDF file.");

    const tth = Array.from({ length: intensity.length }, (_, i) => startTth + i * stepSize);
    return { tth, intensity, radiation: { ka1: wavelength, ka2, ratio } };
}

// Legacy [DATA]-section UDF format
const tth = [], intensity = [];
let inDataSection = false;
let wavelength = null;
for (const line of lines) {
    const trimmedLine = line.trim();
    if (trimmedLine.toUpperCase().startsWith('LAMBDA')) {
        const parts = trimmedLine.split('=');
        if (parts.length > 1) wavelength = parseFloat(parts[1]);
    }
    if (trimmedLine.toUpperCase() === '[DATA]') { inDataSection = true; continue; }
    if (trimmedLine.startsWith('[') && trimmedLine.toUpperCase() !== '[DATA]') inDataSection = false;
    if (inDataSection) {
        const parts = trimmedLine.split(/,/).map(p => p.trim());
        if (parts.length >= 2) {
            const x = parseFloat(parts[0]);
            const y = parseFloat(parts[1]);
            if (!isNaN(x) && !isNaN(y)) { tth.push(x); intensity.push(y); }
        }
    }
}
if (tth.length === 0) throw new Error("No [Data] section found in UDF file.");
return { tth, intensity, wavelength };
};
