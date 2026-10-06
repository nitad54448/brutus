// js/parsers/xrdml.js
// PANalytical XRDML reader.
//
// Classic script, loaded in order by brutus.html (see the list there); its
// top-level names are shared with the other app scripts.

    // PANalytical XRDML.
    const parseXrdmlFile = (xmlString) => {
        const xmlDoc = parseXmlDocument(xmlString, 'XRDML');

        // usedWavelength carries the whole radiation description:
        // intended="K-Alpha" (doublet) or "K-Alpha 1" (monochromated), the
        // Kα1 / Kα2 lines and their intensity ratio. Reading only kAlpha1
        // loaded every ordinary Cu-tube scan as monochromatic Kα1, which
        // disabled Kα2 stripping and the Kα2 ghost flags.
        const radiation = {};
        const uw = xmlDoc.querySelector('usedWavelength');
        if (uw) {
            radiation.intended = uw.getAttribute('intended') || '';
            radiation.ka1 = xmlNumber(uw.querySelector('kAlpha1'));
            radiation.ka2 = xmlNumber(uw.querySelector('kAlpha2'));
            radiation.ratio = xmlNumber(uw.querySelector('ratioKAlpha2KAlpha1'));
        } else {
            radiation.ka1 = xmlNumber(xmlDoc.querySelector('kAlpha1'));
        }

        // A file can hold several scans; use the first one that has a 2θ
        // axis and take its intensities from the SAME dataPoints block.
        const blocks = [...xmlDoc.querySelectorAll('dataPoints')];
        const dataPoints = blocks.find(dp => dp.querySelector('positions[axis="2Theta"]')) || blocks[0] || null;
        const scope = dataPoints || xmlDoc;
        const intensityNode = scope.querySelector("intensities") || scope.querySelector("counts");
        if (!intensityNode) throw new Error("Could not find <intensities> or <counts> in XRDML file.");
        const intensity = numberList(intensityNode.textContent);
        if (intensity.length === 0) throw new Error("XRDML file contains no intensity points.");
        if (blocks.length > 1) console.warn(`XRDML: ${blocks.length} scans in file; reading the first 2θ scan only.`);

        const positionsNode = scope.querySelector('positions[axis="2Theta"]');
        if (!positionsNode) throw new Error("Could not find <positions> in XRDML file.");

        // Non-equidistant scans list every position explicitly.
        const listNode = positionsNode.querySelector('listPositions');
        if (listNode) {
            const tth = numberList(listNode.textContent);
            if (tth.length !== intensity.length || !tth.every(Number.isFinite)) {
                throw new Error(`XRDML listPositions has ${tth.length} values for ${intensity.length} intensities.`);
            }
            return { tth, intensity, radiation };
        }

        const startPosNode = positionsNode.querySelector("startPosition");
        const endPosNode = positionsNode.querySelector("endPosition");
        if (!startPosNode || !endPosNode) throw new Error("Could not find start/end positions in XRDML.");
        const startPos = parseFloat(startPosNode.textContent);
        const endPos = parseFloat(endPosNode.textContent);
        if (!isFinite(startPos) || !isFinite(endPos)) throw new Error("XRDML start/end positions are not numeric.");
        // With a single point the step is undefined; emit the start position.
        const step = intensity.length > 1 ? (endPos - startPos) / (intensity.length - 1) : 0;
        const tth = Array.from({ length: intensity.length }, (_, i) => startPos + i * step);
        return { tth, intensity, radiation };
    };
