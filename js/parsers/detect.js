// js/parsers/detect.js
// Picks the reader for a file from its content and extension.
//
// Classic script, loaded in order by brutus.html (see the list there); its
// top-level names are shared with the other app scripts.

    const detectAndParseFile = (fileName, fileContent) => {
        const name = fileName.toLowerCase();
        const lines = fileContent.trim().split(/\r?\n/);
        const firstLine = lines.length > 0 ? lines[0].trim() : '';
        const upperContent = fileContent.substring(0, 500).toUpperCase(); // Check first 500 chars

        // Parser Registry
        const PARSER_REGISTRY = [
            { // XRDML
                test: (name, content) => name.endsWith('.xrdml') || (content.includes('<?xml') && content.includes('<xrdMeasurement')),
                parser: parseXrdmlFile
            },
            { // Bruker XML (uncompressed RawDataFile; e.g. RawData0.xml from an unzipped .brml)
                test: (name, content) => (name.endsWith('.xml') && content.includes('<RawDataFile')) || (content.includes('<?xml') && content.includes('<RawDataFile')),
                parser: parseBrukerXmlFile
            },
            { // UXD
                test: (name, content, firstLine) => name.endsWith('.uxd') || firstLine.startsWith('_FILEVERSION'),
                parser: parseUxdFile
            },
            { // Rigaku RAS
                test: (name, content, firstLine, upper) => name.endsWith('.ras') || upper.includes('*RAS_HEADER_START'),
                parser: parseRigakuRasFile
            },
            { // Philips UDF/RD/SD
                test: (name) => name.endsWith('.udf') || name.endsWith('.rd') || name.endsWith('.sd'),
                parser: parsePhilipsUdfFile
            },
            { // GSAS raw data: STD / ESD / FXY / FXYE (the BANK line says which)
                test: (name, content, firstLine, upper, allLines) => allLines.some(line => line.trim().toUpperCase().startsWith('BANK')),
                parser: (content) => parseGsasFile(content)
            },
            { // FullProf free format (start/step/end + N values per line)
                test: (name, content, firstLine, upper, allLines) => scanFullProfDat(allLines) !== null,
                parser: parseFullProfDatFile
            },
            { // Jade MDI (treat as 2-column)
                 test: (name, content, firstLine, upper) => name.endsWith('.mdi') && (upper.includes('2-THETA, INTENSITY') || upper.startsWith('(SAMPLE')),
                 parser: parseDataFile
            },
            { // pdCIF
                test: (name, content) => name.endsWith('.cif') || content.includes('_pd_meas_2theta_scan'),
                parser: parsePdCifFile
            }
        ];
        
        // registres
        for (const rule of PARSER_REGISTRY) {
            try {
                if (rule.test(name, fileContent, firstLine, upperContent, lines)) {
                    // Pass 'content' to parser, but 'lines' to the special GSAS one
                    if (rule.parser.length > 1) {
                         return rule.parser(fileContent, lines); // For GSAS parser
                    }
                    return rule.parser(fileContent);
                }
            } catch (e) {
                console.warn(`Parser ${rule.parser.name} failed, trying next...`, e.message);
            }
        }

        // Fallback for all other 2-column-like formats
        // This will attempt to parse: .xy, .csv, .txt, .dat, .asc, etc.... à revoir les fichiers type dans Convert 2 ?
        return parseDataFile(fileContent, fileName);
    };
