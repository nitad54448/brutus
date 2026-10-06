// js/parsers/xml-utils.js
// Helpers shared by the XML readers.
//
// Classic script, loaded in order by brutus.html (see the list there); its
// top-level names are shared with the other app scripts.

    // Small helpers shared by the XML readers.
    const xmlNumber = (node) => {
        if (!node) return null;
        const v = parseFloat(String(node.textContent || '').trim());
        return Number.isFinite(v) ? v : null;
    };
    const parseXmlDocument = (xmlString, label) => {
        const xmlDoc = new DOMParser().parseFromString(xmlString, "application/xml");
        if (xmlDoc.querySelector("parsererror")) throw new Error(`Error parsing ${label} file.`);
        return xmlDoc;
    };
    const numberList = (text) => String(text || '').trim().split(/\s+/).filter(Boolean).map(Number);
