// js/parsers/bruker-xml.js
// Bruker DIFFRAC RawData XML reader, and the .brml (zip) container around it.
//
// Classic script, loaded in order by index.html (see the list there); its
// top-level names are shared with the other app scripts.

    // A .brml file is a zip archive; each scan is a RawData<n>.xml inside it
    // (Experiment0/RawData0.xml ...). In that XML the measured route is
    //
    //   <RawData>
    //     <DataRoutes><DataRoute RouteFlag="Measured">
    //       <ScanInformation> ... <ScanAxes><ScanAxisInfo AxisId="TwoTheta">
    //       <Datum>192,1,2.0102,1.0051,1024</Datum>     one row per step
    //       <RawDataView xsi:type="FixedRawDataView" Start="0" LogicName="MeasuredTime"/>
    //       <RawDataView xsi:type="FixedRawDataView" Start="1" LogicName="AbsorptionFactor"/>
    //       <RawDataView xsi:type="VaryingRawDataView" Start="2" Length="2">
    //           <FieldDefinitions FieldName="TwoTheta"/> <FieldDefinitions FieldName="Theta"/>
    //       <RawDataView xsi:type="RecordedRawDataView" Start="4" MeasuredDataIndex="0"/>  counts
    //   ... <Tube><WaveLengthAlpha1 Value="1.5406"/><WaveLengthAlpha2 .../><WaveLengthRatio .../>
    //
    // The column of each quantity is read from the RawDataView declarations:
    // it moves with the scan type (a 2-circle scan has 5 columns, an HRXRD
    // scan with reciprocal-space columns has 8, counts last).
    const BRUKER_BLANKING_VALUE = -9999;

    // Children of `parent` with the given tag name (direct children only, so a
    // nested element of the same name elsewhere is never picked up).
    const xmlChildren = (parent, tag) => parent ? [...parent.children].filter(el => el.localName === tag) : [];

    const parseBrukerRawDataXml = (xmlDoc) => {
        const routes = [...xmlDoc.getElementsByTagName('DataRoute')];
        if (!routes.length) throw new Error('Bruker RawData XML: no <DataRoute>.');
        const route = routes.find(r => r.getAttribute('RouteFlag') === 'Measured') || routes[0];
        const rows = xmlChildren(route, 'Datum');
        if (rows.length < 2) {
            throw new Error('Bruker RawData XML: this scan has no per-step data rows ' +
                '(a still / stationary-detector frame?). Export it as XY or UXD from DIFFRAC instead.');
        }

        // Column layout from the RawDataView declarations.
        let tthCol = null, countsCol = null, timeCol = null;
        for (const view of route.getElementsByTagName('RawDataView')) {   // under <DataViews>
            const type = view.getAttributeNS('http://www.w3.org/2001/XMLSchema-instance', 'type') || '';
            const start = parseInt(view.getAttribute('Start'), 10);
            if (!Number.isInteger(start)) continue;
            const logic = view.getAttribute('LogicName') || '';
            if (/FixedRawDataView/.test(type)) {
                if (logic === 'MeasuredTime') timeCol = start;
            } else if (/VaryingRawDataView/.test(type)) {
                const fields = [...view.getElementsByTagName('FieldDefinitions')];
                const at = fields.findIndex(f => f.getAttribute('FieldName') === 'TwoTheta' || f.getAttribute('AxisId') === 'TwoTheta');
                if (at >= 0 && tthCol === null) tthCol = start + at;
            } else if (/RecordedRawDataView/.test(type)) {
                // The first recorded channel (MeasuredDataIndex 0) is the detector.
                const index = parseInt(view.getAttribute('MeasuredDataIndex'), 10);
                if (countsCol === null || index === 0) countsCol = start;
            }
        }

        const table = rows.map(row => row.textContent.split(',').map(Number));
        const width = table.reduce((min, row) => Math.min(min, row.length), Infinity);
        // Undeclared layout: the common one (time, absorption, 2θ, θ, ..., counts).
        if (tthCol === null) tthCol = 2;
        if (countsCol === null) countsCol = width - 1;
        if (tthCol >= width || countsCol >= width) {
            throw new Error(`Bruker RawData XML: the declared columns (2θ ${tthCol}, counts ${countsCol}) ` +
                            `do not fit rows of ${width} values.`);
        }

        // Match GSAS-II's BRML reader: y = stored intensity * measured time /
        // TimePerStepEffective. AbsorptionFactor is deliberately not multiplied
        // again. Fractional intensities alone cannot prove how an instrument
        // corrected them (dead-time correction also produces fractions).
        // https://gsas-ii.readthedocs.io/en/latest/_modules/GSASII/imports/G2pwd_BrukerBRML.html
        const scanInfo = xmlChildren(route, 'ScanInformation')[0];
        const effectiveTime = xmlNumber(scanInfo?.getElementsByTagName('TimePerStepEffective')[0]);
        const times = timeCol !== null && timeCol < width ? table.map(r => r[timeCol]) : null;
        const validTimes = times && times.every(t => Number.isFinite(t) && t > 0);
        const timesVary = validTimes && times.some(t => t !== times[0]);
        const scaleTimes = validTimes && effectiveTime > 0;

        const tth = [], intensity = [];
        let blanked = 0;
        for (const r of table) {
            const x = r[tthCol], counts = r[countsCol];
            if (!Number.isFinite(x) || !Number.isFinite(counts)) continue;
            if (counts === BRUKER_BLANKING_VALUE || x === BRUKER_BLANKING_VALUE) { blanked++; continue; }
            const timeScale = scaleTimes ? r[timeCol] / effectiveTime : 1;
            tth.push(x);
            intensity.push(counts * timeScale);
        }
        if (!tth.length) throw new Error('Bruker RawData XML: no readable data rows.');

        // Radiation: the tube's lines. A bare <WaveLength> elsewhere belongs to
        // other blocks (HRXRD alignment, for one) and is used only as a fallback.
        const radiation = {};
        const value = (el) => { const v = el ? parseFloat(el.getAttribute('Value')) : NaN; return Number.isFinite(v) ? v : null; };
        const tube = xmlDoc.getElementsByTagName('Tube')[0] || null;
        if (tube) {
            radiation.ka1 = value(tube.getElementsByTagName('WaveLengthAlpha1')[0]);
            radiation.ka2 = value(tube.getElementsByTagName('WaveLengthAlpha2')[0]);
            radiation.ratio = value(tube.getElementsByTagName('WaveLengthRatio')[0]);
        }
        if (!radiation.ka1) {
            const single = value(xmlDoc.getElementsByTagName('WaveLength')[0]);
            if (single) radiation.single = single;
        }

        const notes = [];
        if (blanked) notes.push(`Bruker RawData: ${blanked} blanked step(s) skipped.`);
        if (scaleTimes && times.some(t => t !== effectiveTime)) notes.push(`Bruker RawData: stored intensities scaled by measured time / effective time (${effectiveTime} s), following GSAS-II.`);
        else if (timesVary) notes.push('Bruker RawData: counting times vary but no effective time is recorded; stored intensities were preserved.');
        return notes.length ? { tth, intensity, radiation, notes } : { tth, intensity, radiation };
    };

    const parseBrukerXmlFile = (xmlString) => {
        const xmlDoc = parseXmlDocument(xmlString, 'Bruker XML');
        return parseBrukerRawDataXml(xmlDoc);
    };

    // --- .brml: read RawData<n>.xml out of the zip -------------------------
    // Minimal zip reader (central directory, stored or deflated entries); the
    // browser's DecompressionStream inflates. Returns the first scan's XML and
    // how many scans the archive holds.
    const extractBrmlRawData = async (arrayBuffer) => {
        const view = new DataView(arrayBuffer);
        const bytes = new Uint8Array(arrayBuffer);
        let eocd = -1;
        for (let i = bytes.length - 22; i >= Math.max(0, bytes.length - 65557); i--) {
            if (view.getUint32(i, true) === 0x06054b50) { eocd = i; break; }
        }
        if (eocd < 0) throw new Error('Not a valid .brml file (no zip directory found).');
        const entryCount = view.getUint16(eocd + 10, true);
        let p = view.getUint32(eocd + 16, true);
        const decoder = new TextDecoder();
        const scans = [];
        for (let n = 0; n < entryCount; n++) {
            if (view.getUint32(p, true) !== 0x02014b50) throw new Error('Damaged .brml file (bad zip directory).');
            const method = view.getUint16(p + 10, true);
            const compressedSize = view.getUint32(p + 20, true);
            const nameLength = view.getUint16(p + 28, true);
            const extraLength = view.getUint16(p + 30, true);
            const commentLength = view.getUint16(p + 32, true);
            const localOffset = view.getUint32(p + 42, true);
            const name = decoder.decode(bytes.subarray(p + 46, p + 46 + nameLength));
            // Either path separator: .NET zip writers on Windows store '\'.
            const m = /(?:^|[\/\\])RawData(\d*)\.xml$/i.exec(name);
            if (m) scans.push({ name, index: m[1] === '' ? 0 : Number(m[1]), method, compressedSize, localOffset });
            p += 46 + nameLength + extraLength + commentLength;
        }
        if (!scans.length) throw new Error('This .brml file holds no RawData scan.');
        scans.sort((a, b) => a.index - b.index);
        const entry = scans[0];
        const lp = entry.localOffset;
        if (view.getUint32(lp, true) !== 0x04034b50) throw new Error('Damaged .brml file (bad zip entry).');
        const dataStart = lp + 30 + view.getUint16(lp + 26, true) + view.getUint16(lp + 28, true);
        const raw = bytes.subarray(dataStart, dataStart + entry.compressedSize);
        let xmlBytes;
        if (entry.method === 0) xmlBytes = raw;
        else if (entry.method === 8) {
            if (typeof DecompressionStream !== 'function') throw new Error('This browser cannot decompress .brml files; unzip it and load RawData0.xml.');
            const stream = new Blob([raw]).stream().pipeThrough(new DecompressionStream('deflate-raw'));
            xmlBytes = new Uint8Array(await new Response(stream).arrayBuffer());
        } else throw new Error(`Unsupported .brml compression (method ${entry.method}).`);
        return { text: decoder.decode(xmlBytes), entryName: entry.name, scanCount: scans.length };
    };
