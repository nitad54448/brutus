# Brutus

Ab initio indexing of powder diffraction patterns, in a browser tab.

You give it a list of peak positions; it works out the unit cell. The name is
honest about the method — it is a brute-force search, and force is not
necessarily smart. But GPUs are very good at doing dumb things quickly, so
Brutus searches every crystal system, from cubic down to triclinic, one after
the other on the GPU, and exhausts search spaces that would be impractical on a
CPU.

Brutus runs entirely on your machine. Nothing is uploaded anywhere, there is no
account, and there is no install step.

---

## Running it

This program can be run by accessing <https://nitad54448.github.io/brutus/brutus.html>.

If you want, you can copy all these files to a folder of your choice and run it
from there. This is a static site, so any web server will work; you can launch
one in Visual Studio or with:

```bash
cd brutus
python -m http.server 8000
# then open http://localhost:8000/brutus.html
```

Opening `brutus.html` straight off the disk (`file://`) will *not* work — the
app fetches the shaders and the space-group database at runtime, and browsers
block that for local files.

You need a browser with **WebGPU** (recent Chrome, Edge, or Safari 18+ with
graphics acceleration set to ON). You probably have a GPU, so if it does not
work it is the settings of your browser rather than the device. I tested this
program on many devices, even on an Android phone. Without WebGPU, cubic,
tetragonal and hexagonal are still indexed, on the CPU in a Web Worker;
orthorhombic, monoclinic and triclinic need the GPU.

---

## How it works

The program reads a data file and you detect the peak positions. Peak positions
are converted to Q-space, where $Q = 1/d^2$, because the relationship between
$Q$ and the Miller indices is linear in the reciprocal cell parameters:

$$Q_{hkl} = Ah^2 + Bk^2 + Cl^2 + Dkl + Ehl + Fhk$$

The assumption underneath everything is that the strongest low-angle
reflections have small integer indices. So the program picks some observed
peaks, guesses an $(hkl)$ for each, solves the linear system, and sees what
falls out. A trial that survives a cheap filter is then refined properly by
weighted least squares — including the zero-point error — and scored against
the whole peak list with M(20) and F(N).

The interesting part is the filtering, because the search generates enormous
numbers of candidates and almost all of them are nonsense. Each GPU thread
solves one system, throws away anything geometrically implausible, and scores
the survivor against the first ten peaks. Very few cells survive this test, but
since the program tests about 10 million cells per second, it will probably
find a valid cell, if the peaks, the volume and the systems are correctly
selected.

The checked systems are searched in order of decreasing symmetry: cubic,
hexagonal, tetragonal, orthorhombic, monoclinic, triclinic. While the GPU works
on one system, the candidates it has already found are refined on the CPU in
parallel. There is no separate rhombohedral search: every R lattice can be
described by its hexagonal triple cell, so the hexagonal search finds it. The
refinement then recognises it, flags it **R**, and counts M(20) on the R lines
only.

The full methodology is in **`brutus_help.html`**: the system
parameterisations, the figure-of-merit definitions, the weighting scheme, the
two-round zero-point strategy and the space-group statistics. It is more
thorough than this file, and it is the place to look if you want to know why
something behaves the way it does.

---

## Quick start

1. **Load a data file.** Two-column text (`.xy`, `.csv`, `.txt`; a header line
   such as `Wavelength 0.79764` is picked up), `.xrdml`, Bruker RawData `.xml`
   (the XML inside a `.brml`, unzipped), `.ras`, `.uxd`, `.udf`, GSAS (`.gsa`,
   `.esd`, `.std`, `.xra`), FullProf `.dat` and pdCIF. When the file records its
   radiation, the matching preset is selected: a Kα doublet gives the Kα-average
   preset (and Kα₂ stripping), a monochromated Kα1 line the Kα1 preset.
2. **Detect peaks.** Adjust `Min peak (%)`, `Radius (pts)` and `Points` until
   the marks match what you see.
3. **Curate the peak list.** This is the step that decides whether indexing
   works. Fix positions, delete impurity lines and Kα₂ shoulders, add anything
   the detector missed with `Ctrl + Click`. Fifteen to twenty clean peaks, with
   nothing spurious at low angle, is the target.
4. **Set parameters.** Radiation preset, `Strip K-alpha2` if you want it,
   a chemically sensible `Max Volume`, and a `2θ Error` that matches your data
   (≈0.02° synchrotron, ≈0.05° typical lab). Then pick the crystal systems:
   every checked system is searched, one after the other. All are checked by
   default except triclinic, which is rare and the most expensive. The lines
   under the search parameters show, for each checked system, the HKL basis,
   the number of peaks combined and the number of trials, before you start.
5. **Index.** Sort the solutions by M(20) and click a row to overlay the
   calculated tick marks on your pattern.

---

## When it finds nothing

Almost always the problem is the peak list. Check that the first ten to fifteen
lines really do belong to one phase and that their positions are accurate.

After that, the two settings that most often shut the search out:

- **`Max Volume`** (default 2000 Å³). This is a hard cut — a candidate cell
  larger than this is discarded before it is ever scored. A small-molecule
  organic with Z = 4 and thirty non-hydrogen atoms is already around
  2000–2500 Å³, and anything pharmaceutical-sized is well past it. If your
  sample is molecular, try 8000 before changing anything else.
- **`2θ Error`.** The GPU pre-filter has no zero-point freedom — that is
  refined later, on the CPU, for candidates that already survived. A systematic
  zero offset therefore shows up as a uniform error on every peak, and once it
  approaches the stated tolerance nothing gets through.

Since v2026-08-28 the app tells you which of these it was. A run that finds
nothing reports the volume range of the candidates it saw and how many peaks the
best of them kept inside the error budget, so "no solutions" comes with a reason
and a setting to change.

The search parameters are shared by all systems and rarely need touching:

| Setting | Default | Effect |
|---|---|---|
| `HKL Basis (% / unknown)` | 5 | Size of the HKL basis, in percent of each system's full list, per unknown cell parameter: 5 % for cubic (1 unknown), 10 % for tetragonal and hexagonal, 15 % for orthorhombic, 20 % for monoclinic, 30 % for triclinic. Orthorhombic and monoclinic never go below all 36 axial reflections plus 40 mixed ones. Typical range 2–12. |
| `Depth` | 3 | Peaks combined = number of unknowns + Depth: 6 for orthorhombic, 7 for monoclinic, 9 for triclinic. Cubic, tetragonal and hexagonal have only one or two unknowns and use three times that: 12 and 15. Always capped by the peaks you have picked. |
| `FoM Tolerance` | 1.25 | Threshold of the GPU pre-filter. Lower is stricter. |
| `Candidates` | 100 | Candidate buffer, in thousands of cells, per system. |

If a system fills the candidate buffer, its search stops early and the run moves
on to the next system. The status line and the PDF report then say how much of
it was actually searched, for example `Orthorhombic at 0.67%`. A cell missing
from the part that was never searched means nothing, so make the search more
selective rather than enlarging the buffer: lower `FoM Tolerance` or
`2θ Error`, reduce `Max Volume`, or use a smaller `Depth` or HKL basis.

---

## Space groups

Once you have a cell, Brutus works out which space groups are compatible with
the systematic absences. Two things do this, and they answer different
questions.

The **automatic analysis** runs on every solution and produces a compatibility
list: which settings the observed reflections contradict, and by how many. It is
grouped by violation count, not ranked — it tells you what the data rule out,
not which survivor is likeliest. Every setting with no hard violation is
listed.

**Space Group MC** runs on request, for one cell. It refines the cell under each
hypothesis in turn and ranks *extinction classes* — sets of space groups that
powder data provably cannot separate — by a log-odds score, with a margin over
the runner-up and a `NOT DECISIVE` flag when that margin is too small to call.
This is the one that claims a winner, and the PDF report reproduces its
conclusion rather than recomputing anything.

Underneath, absences are computed from the **symmetry operators**, not looked up
from reflection-condition strings. For every operator $(R, \mathbf{t})$ of a
group,

$$F(\mathbf{h}) = \exp(2\pi i\,\mathbf{h}\cdot\mathbf{t})\,F(\mathbf{h}R)$$

so a reflection is extinguished exactly when some operator leaves it fixed
($\mathbf{h}R = \mathbf{h}$) while shifting its phase
($\mathbf{h}\cdot\mathbf{t} \notin \mathbb{Z}$). That single test replaces the
whole business of parsing condition strings, guessing which zone a reflection
belongs to, and reconstructing the conditions the tables leave implied. It is
integer arithmetic throughout, with no tolerance to get wrong.

A powder peak, however, is a *line*, not a single reflection, and every
reflection that falls on it contributes. A line counts as forbidden only if
every reflection on it is: its whole metric orbit (320 and 230 in a cubic cell,
$(h,k,l)$ and $(k,h,l)$ in a hexagonal one), plus any reflection that coincides
with it exactly (cubic 221 and 300). Both routes apply this rule. Testing only
one representative reflection used to reject the R groups, Pa-3, Ia-3 and the
cubic c-, n- and d-glide groups on patterns that obey them exactly.

---

## The space-group database

`sg_ops.json` holds all **530 settings of the 230 space groups** — every
symmetry operator, the zone definitions, and the printed reflection conditions —
in about 240 KB. Rotation matrices are dictionary-encoded against a shared table
of the 64 distinct ones.

`sg_ops.json` is generated directly from [cctbx](https://cctbx.github.io/):

```bash
python build_sg_db.py --out sg_ops.json
```

You only need this if you want to rebuild the database; a working copy ships
with the repository. It requires a cctbx environment — nothing else in Brutus
does.

The zones and conditions are *derived from* the operators rather than copied
from a table, so they cannot drift out of step with what the application
actually uses. They come out in International Tables form (`00l: l=6n` for
P6₁, `h-hl: h+l=3n, l=2n` for R-3c). The rule strings are then checked against
the operators, read exactly as the app reads them, first per setting and again
on the finished file. The file is written only if every check passes, so a
failed build leaves the previous `sg_ops.json` in place. Three commands let you
check it without trusting me:

```bash
python build_sg_db.py --self-test        # checks the maths, no cctbx needed
python build_sg_db.py --check sg_ops.json # re-verifies a finished file
node check_sg_ops.mjs sg_ops.json         # checks the app can consume it
```

`--self-test` builds eleven space groups by closing published generators,
checks each group's order against its published value first (so a typo in a
generator fails loudly rather than quietly testing the wrong group), then
derives the reflection conditions and compares them against the International
Tables. `--limit N` builds only the first N settings, as a smoke test; the
result goes to `sg_ops.partial.json`, so it cannot replace the real database.

`check_sg_ops.mjs` also reports how many settings are actually *reachable*.
About 77 are deliberately excluded: monoclinic settings that are not b-unique,
and settings written on rhombohedral rather than hexagonal axes. That is
correct, not a loss — Brutus produces b-unique monoclinic cells and indexes
R lattices in hexagonal axes, so a condition list written for other axes refers
to different indices, and applying it would be wrong.

---

## Repository layout

**Runtime** — everything below is needed to serve the app:

| File | |
|---|---|
| `brutus.html` | the application |
| `brutus_help.html` | full technical documentation |
| `js/core/`, `js/data/`, `js/parsers/`, `js/peaks/`, `js/chart/`, `js/dialogs/`, `js/report/`, `js/indexing/`, `js/main.js` | the UI: state, file readers, peak picking, chart, dialogs, exports and report, and the indexing orchestration (`js/indexing/run.js`) |
| `js/crystallography/` | the crystallography: HKL generation, least squares, figures of merit, Niggli reduction, space-group analysis. The same files run on the main thread and in both workers; `manifest.js` lists them in load order |
| `js/workers/` | `index-worker.js` (CPU search of cubic, tetragonal and hexagonal when WebGPU is not available, and post-processing) and `refinement-worker.js` (batch refinement; a pool of these runs alongside the GPU search) |
| `js/gpu/` | `webgpu-engine.js` (device, buffers, dispatch chunking, combinadics), `gpu-setup.js` (the search plan: basis size, peaks and trials for each system) and `gpu-limits.js` (pre-flight checks and the Start button) |
| `shaders/*.wgsl` | the compute kernels: `highsym_solver.wgsl` (cubic, tetragonal, hexagonal), `ortho_solver.wgsl`, `monoclinic_solver.wgsl`, `triclinic_solver.wgsl` |
| `sg_ops.json` | the space-group database |
| `styles.css`, `inter-font.css`, `Inter-Variable.ttf`, `tex-svg.js`, `scripts/` | styling, fonts, MathJax, and the vendored libraries |

**Tooling** — not deployed, not needed to run anything:

| Script | |
|---|---|
| `build_sg_db.py` | builds `sg_ops.json` from cctbx |
| `check_sg_ops.mjs` | validates the database against the application |
| `bump_version.py` | stamps one cache-busting `?v=` across `brutus.html` |
| `test_sg_ops.mjs` | derives reflection conditions from the shipping operator code and checks them against the International Tables |
| `check_pipeline.mjs` | verifies the app (`js/indexing/run.js`), the engine and the shaders still agree on how the HKL basis is packed, for all six systems |
| `check_load_order.mjs` | checks that the scripts in `brutus.html` and the worker manifest load in a safe order (nothing runs before what it uses is defined) and that the two lists agree. Needs `npm install --no-save typescript` |
| `regression_test.mjs` | Node regression tests of the crystallography code: known cubic, tetragonal, hexagonal and orthorhombic cells, Niggli reduction, and error propagation against Monte Carlo (about three minutes) |

`test_sg_ops.mjs` and `check_pipeline.mjs` are worth running after touching the
code they cover, because both failures are otherwise invisible: the run
completes, every candidate cell is nonsense, and you get no solutions and no
error. Run `check_load_order.mjs` after moving code between files or adding a
script: the scripts share one global scope, and a name used before the script
that defines it has run fails only when that code path executes.

### A note on the browser cache

The page, the two workers and the GPU shaders must all come from the same build:
if the browser serves an old crystallography file to the workers while the page
runs a new one, the result is not an obvious caching error but, for example, a
results table and a PDF report that disagree. Every URL therefore carries the
same `?v=`: the `<script>` tags in `brutus.html`, and — through
`js/core/version.js` — the workers, the shaders and `sg_ops.json`. Run
`python bump_version.py` after any change; it sets every tag together and warns
if they have drifted apart.

---

## References

Brutus was developed by Nita Dragoe at Université Paris-Saclay (2024–2026), as
a successor to the earlier program *Powder* (1999–2000). If you use it, please
cite [https://doi.org/10.13140/RG.2.2.18182.84806](https://doi.org/10.13140/RG.2.2.18182.84806).

1. **M(20):** de Wolff, P. M. (1968). *J. Appl. Cryst.* **1**, 108–113.
2. **F(N):** Smith, G. S. & Snyder, R. L. (1979). *J. Appl. Cryst.* **12**, 60–65.
3. **Richardson–Lucy deconvolution:** Richardson, W. H. (1972). *J. Opt. Soc. Am.* **62**, 55–59; Lucy, L. B. (1974). *Astron. J.* **79**, 745.
4. **cctbx:** Grosse-Kunstleve, R. W., Sauter, N. K., Moriarty, N. W. & Adams, P. D. (2002). *J. Appl. Cryst.* **35**, 126–136.
5. **Previous software:** Dragoe, N. (2001). *J. Appl. Cryst.* **34**, 535.

Bug reports and awkward patterns that refuse to index are both welcome:
[open an issue](https://github.com/nitad54448/brutus/issues/new?template=bug_report.yml).

---

## License

Licensed under a
[Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](http://creativecommons.org/licenses/by-nc-nd/4.0/).

<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">
  <img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" />
</a>

*This document and most of the code porting was done by an AI. Last updated: 10 October 2026.*
