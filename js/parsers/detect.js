// js/parsers/detect.js
// Picks the reader for a file from its content and extension.
//
// Classic script, loaded in order by index.html (see the list there); its
// top-level names are shared with the other app scripts.

    const detectAndParseFile = (fileName, fileContent) => {
        const name = fileName.toLowerCase();
        const lines = fileContent.trim().split(/\r?\n/);
        const firstLine = lines.length > 0 ? lines[0].trim() : '';
        const upperContent = fileContent.substring(0, BRUTUS_LIMITS.parsers.formatSniffChars).toUpperCase();

        // Parser Registry
        const PARSER_REGISTRY = [
            { // XRDML
                test: (name, content) => name.endsWith('.xrdml') || (content.includes('<?xml') && content.includes('<xrdMeasurement')),
                parser: parseXrdmlFile
            },
            { // Bruker DIFFRAC XML: <RawData> (RawData0.xml from a .brml, or a .brml
              // already extracted by file-load.js), or the <RawDataFile> layout that
              // earlier Brutus versions exported. '<RawData ' / '<RawData>' does not
              // match the <RawDataView> elements of other formats.
                test: (name, content) => /<RawData(File)?[\s>]/.test(content.substring(0, 4000)),
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
                // A real BANK line carries numbers ("BANK 1 4000 800 CONST ...");
                // a ".xy" header such as "Bank: sample 3" must not claim the file.
                test: (name, content, firstLine, upper, allLines) => allLines.some(line => /^BANK\s+\d+\s+\d+/i.test(line.trim())),
                parser: (content) => parseGsasFile(content)
            },
            { // FullProf free format (start/step/end + N values per line)
                test: (name, content, firstLine, upper, allLines) => scanFullProfDat(allLines) !== null,
                parser: parseFullProfDatFile
            },
            { // Jade MDI (treat as 2-column)
                 test: (name, content, firstLine, upper) => name.endsWith('.mdi') && (upper.includes('2-THETA, INTENSITY') || upper.startsWith('(SAMPLE')),
                 parser: (content) => parseDataFile(content, fileName)
            },
            { // pdCIF
                test: (name, content) => name.endsWith('.cif') || content.includes('_pd_meas_2theta_scan'),
                parser: parsePdCifFile
            }
        ];
        
        // A matching reader owns the file. An unsupported or damaged known
        // format must not be reinterpreted as generic angle/intensity data.
        for (const rule of PARSER_REGISTRY) {
            if (rule.test(name, fileContent, firstLine, upperContent, lines))
                return rule.parser(fileContent, fileName);
        }

        // Fallback for all other 2-column-like formats (.xy, .csv, .txt, .dat, .asc, etc.)
        return parseDataFile(fileContent, fileName);
    };
