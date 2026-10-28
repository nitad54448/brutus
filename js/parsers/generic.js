// js/parsers/generic.js
// Generic two/three-column text reader (.xy, .dat, .csv, .txt ...).
//
// Classic script, loaded in order by index.html (see the list there); its
// top-level names are shared with the other app scripts.

    /**
     * Generic 2-column parser. This is the fallback for most text files.
     * Includes validation logic for 2-theta (X) and step size (dX).
     */
    const parseDataFile = (text, fileName = "") => {
        const limits = BRUTUS_LIMITS.parsers;
        const lines = text.trim().split(/\r?\n/);
        const tth = [], intensity = [];
        let last_x = -Infinity;
        let suspicious_steps = 0;
        let positive_x_values = 0;
        let negative_steps = 0;
        let headerLines = 0;
        let dataStarted = false;
        // Many two-column exports note the wavelength in a header line
        // ("Wavelength 0.79764", "# lambda = 1.5406"). Without it the file
        // silently loaded as Cu Kα, which mis-scales every d-spacing.
        let headerWavelength = null;
        const WAVELENGTH_RX = /(?:wave\s*-?\s*length|lambda|\u03bb)\s*[:=]?\s*([0-9]*\.[0-9]+)/i;
        const noteWavelength = (line) => {
            if (headerWavelength !== null) return;
            const m = WAVELENGTH_RX.exec(line);
            const v = m ? parseFloat(m[1]) : NaN;
            if (v > limits.minHeaderWavelength && v < limits.maxHeaderWavelength) headerWavelength = v;
        };

        // Per-line decimal-comma test (see below). A row whose angle happens
        // to be whole ("11\t1234,5") cannot be told from CSV on its own, so
        // the verdict of the unambiguous rows is applied to the whole file.
        const lineHasDecimalComma = (rawLine) => !rawLine.includes('.') && /,/.test(rawLine) &&
            (rawLine.includes(';') || /\d,\d+\s+[+-]?\d/.test(rawLine));
        const fileUsesDecimalComma = lines.some(line => lineHasDecimalComma(line.trim()));

        lines.forEach(line => {
            // Skip commented or empty lines
            if (line.startsWith('#') || line.startsWith('//') || line.startsWith('!') || line.startsWith(';') || line.trim() === '') {
                if (!dataStarted) { headerLines++; noteWavelength(line); }
                return;
            }
            
            // Skip non-commented header lines (that contain letters). An
            // exponent marker is not a letter here: a file written as
            // "1.000e+01 2.345e+03" (numpy.savetxt's default format) is all
            // data, and testing the raw line rejected every row of it.
            if (!dataStarted) {
                if (/[a-zA-Z]/.test(line.replace(/([0-9.])[eE][+-]?(?=\d)/g, '$1'))) {
                    headerLines++;
                    noteWavelength(line);
                    return;
                }
            }

            // Decimal comma vs comma separator.
            // "10,5 200" (European decimal) and "10.5,200.7" or "10, 200" (CSV)
            // are all valid and cannot be told apart by a blanket substitution,
            // so decide per line. A comma is a DECIMAL MARK when there is no dot
            // anywhere AND either a semicolon separates the columns or a digit-comma-digit
            // token is separated from the next column by whitespace. Otherwise
            // the comma is the field separator.
            const rawLine = line.trim();
            const commaCount = (rawLine.match(/,/g) || []).length;
            const hasDot = rawLine.includes('.');
            const commaIsDecimal = commaCount > 0 && !hasDot &&
                (fileUsesDecimalComma || lineHasDecimalComma(rawLine));
            const sanitizedLine = commaIsDecimal
                ? rawLine.replace(/,(\d)/g, '.$1')
                : rawLine.replace(/,/g, ' ');
            const parts = sanitizedLine.split(/[\s;]+/);
            if (parts.length < 2) return;

            const x = parseFloat(parts[0]);
            const y = parseFloat(parts[1]);

            // If we get non-numeric data, it's either a header or a bad line
            if (isNaN(x) || isNaN(y)) {
                if (!dataStarted) headerLines++; // Still in the header
                return;
            }
            
            dataStarted = true; // First valid numeric pair found

            // vérif
            if (x > 0) positive_x_values++;

            if (last_x !== -Infinity) {
                const dX = x - last_x;
                if (dX < 0) {
                    negative_steps++; // Data is descending
                } else if (dX > 0 && (dX < limits.minTypicalStepDeg || dX > limits.maxTypicalStepDeg)) {
                    suspicious_steps++; // Step size is weird
                }
            }
            last_x = x;
            

            tth.push(x);
            intensity.push(y);
        });

        // Final checks (log warnings to console) 
        if (tth.length > limits.genericWarningMinPoints) {
            if (positive_x_values / tth.length < limits.positiveFraction) {
                console.warn(`Data File (${fileName}) Warning: Most 2-theta (X) values are zero or negative. This is unusual for XRD data.`);
            }
            if (negative_steps / tth.length > limits.descendingFraction) {
                 console.warn(`Data File (${fileName}) Warning: Data appears to be sorted in descending 2-theta order.`);
            }
            if (suspicious_steps / tth.length > limits.suspiciousFraction) {
                console.warn(`Data File (${fileName}) Warning: Many data points have a step size outside the typical range (${limits.minTypicalStepDeg}° - ${limits.maxTypicalStepDeg}°). Check file format.`);
            }
        } else if (tth.length === 0) {
             throw new Error(`Could not parse any 2-column data from ${fileName}. File may be binary or have an unknown header.`);
        }

        return headerWavelength !== null
            ? { tth, intensity, radiation: { single: headerWavelength } }
            : { tth, intensity };
    };
