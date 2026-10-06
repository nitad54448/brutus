// js/parsers/bruker-xml.js
// Bruker RawDataFile XML reader.
//
// Classic script, loaded in order by brutus.html (see the list there); its
// top-level names are shared with the other app scripts.

    // Bruker RawDataFile XML (the uncompressed XML found inside a .brml).
    const parseBrukerXmlFile = (xmlString) => {
        const xmlDoc = parseXmlDocument(xmlString, 'Bruker XML');
        // Radiation: the kAlpha1 attribute is what this reader has always
        // used; kAlpha2 / ratio are read when present so a doublet tube is
        // recognised as one (Kα-average preset, Kα2 stripping available).
        const radiation = {};
        const wlNode = xmlDoc.querySelector('usedWavelength');
        if (wlNode) {
            const attr = (name) => { const v = parseFloat(wlNode.getAttribute(name)); return Number.isFinite(v) ? v : null; };
            radiation.ka1 = attr('kAlpha1');
            radiation.ka2 = attr('kAlpha2');
            radiation.ratio = attr('ratioKAlpha2KAlpha1') ?? attr('ratio');
            radiation.intended = wlNode.getAttribute('intended') || '';
        }
        const intensityNode = xmlDoc.querySelector("dataPoints > counts");
        if (!intensityNode) throw new Error("No <counts> data found in Bruker XML file.");
        const intensity = numberList(intensityNode.textContent);
        const startPosNode = xmlDoc.querySelector('startPosition[axis="TwoTheta"]');
        const stepSizeNode = xmlDoc.querySelector('increment[axis="TwoTheta"]');
        if (!startPosNode || !stepSizeNode) throw new Error("Could not find scan parameters in Bruker XML file.");
        const startPos = parseFloat(startPosNode.textContent);
        const stepSize = parseFloat(stepSizeNode.textContent);
        const tth = Array.from({ length: intensity.length }, (_, i) => startPos + i * stepSize);
        return { tth, intensity, radiation };
    };
