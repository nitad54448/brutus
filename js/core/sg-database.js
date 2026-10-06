// js/core/sg-database.js
// Loads the space-group operator database (sg_ops.json).
//
// Classic script, loaded in order by brutus.html (see the list there); its
// top-level names are shared with the other app scripts.

// space group, fichier crée avec Gemmi, je mets le script aussi sur git
// nouvelle version, règles de CCTBX, depuis le 17 janvier 2026
//v2 depuis le 14 juillet 2026
let spaceGroupData = null;
   // Space group database, built straight from cctbx by build_sg_db.py:
   //     python3 build_sg_db.py --out sg_ops.json
   //
   // One script, one point of failure. It was previously produced by running a
   // separate generator to write 527 per-setting files and a second script to
   // merge them; that put two schemas between cctbx and this app, and a field
   // renamed upstream broke the merge silently.
   //
   // The old cctbx_space_groups_all_settings_v4.json is gone. It stored only
   // reflection-condition strings, which forced the app to guess zone
   // membership from the label -- 'hhl' as |h| == |k|, matching the separate
   // h-hl zone as well -- and to reconstruct implied conditions with an
   // inheritance rule. sg_ops.json carries the symmetry operators, so absences
   // are computed rather than parsed, and the zone normals, so membership is
   // arithmetic. Loading a v4 file here will fail the operator check below.
async function loadSpaceGroupData() {
    try {
        // Versioned like the scripts. The database format changed when the
        // operators arrived, so a browser holding a cached copy of the old
        // file fails the operator check below with a message about v4 --
        // confusing, because the file on disk is correct.
        const response = await fetch('sg_ops.json' + APP_VERSION_QS);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        if (!data || !data.space_groups) throw new Error("no 'space_groups' member");
        if (!data.rotations || !data.zone_defs) {
            throw new Error("no operator/zone tables -- this looks like the old " +
                            "v4 file. Rebuild with build_sg_db.py.");
        }
        spaceGroupData = data;
        const nSettings = Object.values(data.space_groups)
            .reduce((n, g) => n + ((g.settings || []).length), 0);
        console.log(`Space group database loaded: ${nSettings} settings, ` +
                    `${data.rotations.length} distinct rotations, ` +
                    `${Object.keys(data.zone_defs).length} zone labels.`);
    } catch (error) {
        console.error("Could not load sg_ops.json:", error);
        showStatus("Warning: could not load sg_ops.json (" + error.message +
                   "). Space group analysis will be disabled. Build it with " +
                   "python3 build_sg_db.py --out sg_ops.json", "error", 10000);
    }
}
