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

This program can be run by accessing <https://nitad54448.github.io/brutus/>.

If you want, you can copy all these files to a folder of your choice and run it
from there. This is a static site, so any web server will work; you can launch
one in Visual Studio or with:

```bash
cd brutus
python -m http.server 8000
# then open http://localhost:8000/index.html
```

Opening `index.html` straight off the disk (`file://`) will *not* work — the
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

1. **Load a data file** (`File → Load pattern…`). Two-column text (`.xy`, `.csv`, `.txt`; a header line
   such as `Wavelength 0.79764` is picked up, and numbers may be written in
   scientific notation, as `numpy.savetxt` does), `.xrdml`, Bruker RawData `.xml`
   (the XML inside a `.brml`, unzipped), `.ras`, `.uxd`, `.udf`, GSAS (`.gsa`,
   `.esd`, `.std`, `.xra`), FullProf `.dat` and pdCIF. When the file records its
   radiation, the matching preset is selected: a Kα doublet gives the Kα-average
   preset (Kα₂ stripping becomes available, but stays off), a monochromated Kα1
   line the Kα1 preset. A file that fails to load replaces nothing: the current
   pattern, peaks and solutions are kept.
2. **Detect peaks** (Peaks tab). Adjust `Min peak (%)`, `Prom. (%)`, `Radius (Å⁻¹)` and
   `Smooth. (Å⁻¹)` until the marks match what you see.
3. **Curate the peak list.** This is the step that decides whether indexing
   works. Fix positions, delete impurity lines and any Kα₂ peak that was not
   flagged (keep the yellow, flagged ones; see *Kα₂ doublets* below), add
   anything the detector missed with `Ctrl + Click`. Fifteen to twenty clean
   peaks, with nothing spurious at low angle, is the target.
4. **Set parameters** (Data and Indexing tabs). Radiation preset, `Strip K-alpha2` whenever the Kα
   doublet is resolved in your pattern (see *Kα₂ doublets* below),
   a chemically sensible `Max Volume`, and a `2θ Error` that matches your data
   (≈0.02° synchrotron, ≈0.05° typical lab). Then pick the crystal systems:
   every checked system is searched, one after the other. All are checked by
   default except triclinic, which is rare and the most expensive. The lines
   under the search parameters show, for each checked system, the HKL basis,
   the number of peaks combined and the number of trials, before you start.
5. **Index.** Press `Index` in the title bar (it becomes a red `Stop`; Esc
   stops too). Sort the solutions by M(20) or volume, either way, and open one
   to overlay its calculated tick marks on your pattern. Of two descriptions of the same
   lattice the more symmetric one is kept; when the less symmetric one fits
   more than twice as well (M(20)), both are listed, marked ≡ with each
   other's system: decide whether the distortion is real, and erase the other.

---

## Kα₂ doublets

With a Kα-average preset and `Strip K-alpha2` off, Brutus flags resolved Kα₂
peaks (yellow rows) and keeps them out of indexing. Their Kα₁ parents (rows
marked `*`) are pure Kα₁ lines, so they are indexed and refined with λ(Kα₁);
every other peak uses the wavelength in the `Wavelength (Å)` box, the Kα
average, which is right for an unresolved doublet. The same rule holds in the
Refine MC and Space Group MC dialogs, the absence analysis and the
PDF report, and the plot draws those lines at their Kα₁ positions. Keep the
yellow rows in the list: they are what marks their parents as Kα₁ lines.

Reading those lines at the average instead would make the cell too large by
λ(avg)/λ(Kα₁) − 1 (0.08 % for Cu, 0.20 % for Mo) with a normal M(20). On
synthetic Cu Kα patterns of a cubic cell with a = 5.4309 Å:

| Pattern | Flagged | Stripped |
|---|---|---|
| all doublets resolved, zero 0.06° | 5.4309 | 5.4309 |
| low angle unresolved, high angle resolved, zero 0.04° | 5.4305 | 5.4309 |

Only a peak whose Kα₂ companion was found gets λ(Kα₁). A resolved Kα₁ line
whose companion was not picked, or a partly resolved peak whose maximum sits
between the two lines, stays at the average and shows up as a misfit (a
lower M(20)) instead of scaling the whole cell. Add the missed companion with
`Ctrl + Click` on its shoulder (`Snap` off) and the pair is recognised.
**Stripping Kα₂ remains the most accurate route**: every peak moves to its Kα₁
position and the wavelength switches to λ(Kα₁). A Kα₁ preset flags nothing, so
with one, delete any Kα₂ peaks by hand.

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
| `HKL Basis (% / unknown)` | 5 | Size of the HKL basis, in percent of each system's full list, per unknown cell parameter: 10 % for tetragonal and hexagonal (2 unknowns), 15 % for orthorhombic, 20 % for monoclinic, 30 % for triclinic. Cubic always searches its whole list (752 hkl, 9,024 trials), which costs next to nothing. Orthorhombic and monoclinic never go below all 36 axial reflections plus 40 mixed ones. Typical range 2–12. |
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

## How many solutions are kept

The list keeps at most 50 solutions (plus the partner of an ≡ pair), best M(20) first, after equivalent
descriptions of one lattice are merged. Every system that has cells keeps a reserved share of those 50
(50 divided by the number of systems present, so 25 each for two, 8 for six); places a system does not fill go
to the best remaining cells. While a run goes on the working list is cut the same way, from 500 to 400. The
Log gives the count per system after the sieve. Details: `brutus_help.html`, *Evaluating Solutions*.

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
one representative reflection would reject the R groups, Pa-3, Ia-3 and the
cubic c-, n- and d-glide groups on patterns that obey them exactly.

The **Solution** menu works on the open solution. Besides Test Space Group it
offers two tools for choosing a group by hand:

- **Explore Space Group…** attaches any of the 530 settings to the solution and
  draws its allowed reflections (**Sim**). A solution can hold several groups;
  they are saved in the project, listed in the PDF report and written to the
  pdCIF.
- **Convert…** carries the cell to a maximal **subgroup** (translationengleiche
  or klassengleiche, listed from the operators with
  `sg_subgroups.js`) or to another group related by a change of axes
  (`sg_convert.js`), and adds the result as a new solution. The subgroups are
  named by matching their operators against `sg_ops.json`, which can leave a
  few unnamed (the search stops after 30 s). The M(20) of the new solution is
  the parent's: it is not recomputed.

The technical guide (`Help` in the title bar) describes both in full.

---

## The space-group database

`sg_ops.json` holds all **530 settings of the 230 space groups** — every
symmetry operator, the zone definitions, and the printed reflection conditions —
in about 240 KB. Rotation matrices are dictionary-encoded against a shared table
of the 64 distinct ones.

`sg_ops.json` is generated directly from [cctbx](https://cctbx.github.io/):

```bash
python scripts/build_sg_db.py --out sg_ops.json
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
python scripts/build_sg_db.py --self-test         # checks the maths, no cctbx needed
python scripts/build_sg_db.py --check sg_ops.json # re-verifies a finished file
node tests/check_sg_ops.mjs sg_ops.json           # checks the app can consume it
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
| `index.html` | the application: the markup and the ordered list of scripts, which the loader runs in order with the version from `js/core/version.js` |
| `js/core/version.js` | the single source of truth for the application version (`APP_VERSION`); everything else reads it |
| `js/crystallography/limits.js` | shared runtime limits, numerical policy thresholds, input bounds and defaults; loaded before the application and in both workers |
| `brutus_help.html` | full technical documentation |
| `js/core/workspace.js`, `js/core/shell.js` | the window around the plot: log, history, projects, theme, title-bar menus, Index / Stop and the run lock, plot strip, metadata, status bar |
| `js/convert/`, `js/dialogs/convert.js` | Convert: `sg_subgroups.js` and `sg_convert.js` (maximal subgroups and basis transformations) and the dialog that uses them |
| `js/core/`, `js/data/`, `js/parsers/`, `js/peaks/`, `js/chart/`, `js/dialogs/`, `js/report/`, `js/indexing/`, `js/main.js` | the UI: state, file readers, peak picking, chart, dialogs, exports and report, and the indexing orchestration (`js/indexing/run.js`) |
| `js/crystallography/` | the crystallography: HKL generation, least squares, figures of merit, Niggli reduction, space-group analysis. The same files run on the main thread and in both workers; `manifest.js` lists them in load order |
| `js/workers/` | `index-worker.js` (CPU search of cubic, tetragonal and hexagonal when WebGPU is not available, and post-processing) and `refinement-worker.js` (batch refinement; a pool of these runs alongside the GPU search) |
| `js/gpu/` | `webgpu-engine.js` (device, buffers, dispatch chunking, combinadics), `gpu-setup.js` (the search plan: basis size, peaks and trials for each system) and `gpu-limits.js` (pre-flight checks and the Index button) |
| `shaders/*.wgsl` | the compute kernels: `highsym_solver.wgsl` (cubic, tetragonal, hexagonal), `ortho_solver.wgsl`, `monoclinic_solver.wgsl`, `triclinic_solver.wgsl` |
| `sg_ops.json` | the space-group database |
| `css/` | `styles.css`, `workspace.css`, `dialogs.css`, `inter-font.css`, and `Inter-Variable.ttf` (styling and bundled font) |
| `vendor/` | third-party browser libraries (`chart.umd.js`, `hammer.min.js`, `chartjs-plugin-zoom.min.js`, `jspdf.umd.min.js`) |

**Tooling** — not deployed, not needed to run anything:

| Script | |
|---|---|
| `scripts/build_sg_db.py` | builds `sg_ops.json` from cctbx |
| `tests/check_sg_ops.mjs` | validates the database against the application |
| `tests/check_pipeline.mjs` | verifies the app (`js/indexing/run.js`), the engine and the shaders still agree on how the HKL basis is packed, for all six systems |
| `tests/check_manual_sg.mjs` | checks that all 530 settings calculate, the F extinctions, Pa-3 and the filter restoration |
| `tests/check_workspace_browser.mjs` | real-page tests including project precision, restored PDF analysis, import security and irregular XRDML round trips; `BRUTUS_CHROME` overrides the browser path, and `BRUTUS_TEST_GPU=1` also compiles and dispatches all six GPU solvers |
| `tests/check_load_order.mjs` | checks that the scripts in `index.html` and the worker manifest load in a safe order (nothing runs before what it uses is defined) and that the two lists agree. Needs `npm install --no-save typescript` |
| `tests/check_ui_regressions.mjs` | checks file-load races, scan sorting, numeric inputs, Kα₂ tolerance updates, indexing locks and Save-dialog keyboard behavior. Run with `node --test tests/check_ui_regressions.mjs`; no dependencies needed |
| `tests/check_parser_regressions.mjs` | malformed input rejection, pdCIF tokenization, explicit-angle exports and consistent space-group identifiers |
| `tests/check_project_regressions.mjs` | lossless project values, absence Sets and infinite margins, import validation and rollback, and escaped solution markup |
| `tests/check_science_boundaries.mjs` | exact fits, uncertainty at the monoclinic 90° fold, geometric volumes and bounds after refinement |
| `tests/check_device_loss.mjs` | interruption bookkeeping, partial-result analysis, and protection from stale GPU/run callbacks |
| `tests/regression_test.mjs` | Node regression tests of the crystallography code: known cubic, tetragonal, hexagonal and orthorhombic cells, Niggli reduction, error propagation against Monte Carlo, Kα₁ lines indexed with λ(Kα₁), and the final sieve keeping the most symmetric description of a lattice, or both when a less symmetric one fits clearly better (about three minutes) |

Run the scripts in `tests/` from the app folder, the one holding `index.html`
(for example `node tests/check_pipeline.mjs`): `check_pipeline.mjs`,
`check_sg_ops.mjs` look for the app files in the current directory.

`check_sg_ops.mjs` and `check_pipeline.mjs` are worth running after touching the
code they cover, because both failures are otherwise invisible: the run
completes, every candidate cell is nonsense, and you get no solutions and no
error. Run `check_load_order.mjs` after moving code between files or adding a
script: the scripts share one global scope, and a name used before the script
that defines it has run fails only when that code path executes.

Change application policy in `js/crystallography/limits.js`. Numeric controls
receive their bounds and defaults from that file; CPU refinement, Monte Carlo,
swap fitting and GPU dispatch read the same physical limits. GPU angle bounds
travel in the shared uniform buffer. Matrix dimensions, format widths and other
mathematical/protocol constants remain beside their algorithms.

Project saving retains the measured and fitted numerical precision. Invalid
projects are validated before replacement, with rollback if restoration fails.
UXD, XRDML and GSAS exports preserve irregular angle coordinates; Bruker XML
export requires a regular grid. Recognized invalid or unsupported data files
report their parser error instead of being retried as generic two-column data.

### A note on the browser cache

The page, the two workers and the GPU shaders must all come from the same build:
if the browser serves an old crystallography file to the workers while the page
runs a new one, the result is not an obvious caching error but, for example, a
results table and a PDF report that disagree. The version is therefore written
once, in `js/core/version.js` (`const APP_VERSION = '...'`). `index.html` loads
`js/core/version.js` first and appends `APP_VERSION_QS` (`?v=<version>`) to each
subsequent script; the same number reaches the workers, the shaders and
`sg_ops.json`, and it is shown at the bottom right of the window and in the PDF
report. Edit `APP_VERSION` in `js/core/version.js` after any change. The app
must be served over HTTP, as `sg_ops.json` already required.

---

## References

Brutus was developed by Nita Dragoe at Université Paris-Saclay (2024–2026), as
a successor to the earlier program *Powder* (1999–2000). If you use it, please
cite [https://doi.org/10.13140/RG.2.2.18182.84806](https://doi.org/10.13140/RG.2.2.18182.84806).

1. **M(20):** de Wolff, P. M. (1968). *J. Appl. Cryst.* **1**, 108–113.
2. **F(N):** Smith, G. S. & Snyder, R. L. (1979). *J. Appl. Cryst.* **12**, 60–65.
3. **Kα₂ elimination in q-space:** Zhang, Z. *et al.* (2026). *Measurement* **276**, 121448. Brutus inverts the doublet exactly, starting from this method, as described in `brutus_help.html`.
4. **cctbx:** Grosse-Kunstleve, R. W., Sauter, N. K., Moriarty, N. W. & Adams, P. D. (2002). *J. Appl. Cryst.* **35**, 126–136.
5. **Earlier program:** Dragoe, N. (2001). *J. Appl. Cryst.* **34**, 535.

Bug reports and awkward patterns that refuse to index are both welcome:
[open an issue](https://github.com/nitad54448/brutus/issues/new?template=bug_report.yml).

---

## License

Original Brutus material is licensed under a
[Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](http://creativecommons.org/licenses/by-nc-nd/4.0/).

<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">
  <img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" />
</a>

Bundled third-party components keep their own terms; see [LICENCE](LICENCE).

*This document and most of the code porting was done by an AI. Last updated: 28 October 2026.*

## Manual space-group comparison

Open a solution in **Solutions**, then choose **Solution → Explore Space Group…**
(the crystal system filter starts on that of the solution's cell)
and filter by crystal system, centering, international
number, Hermann–Mauguin symbol, Hall symbol or setting. **Add to solution** attaches
the chosen setting to the solution: it is listed under the solution, saved in
the project, and written to the PDF report and the pdCIF.
The database contains 230 space groups in 530 settings. Green reflection ticks
show the chosen setting alongside the observed peaks and the solution's blue
calculated ticks. **Remove from solution** in the same dialog clears them.

This is a fixed-cell extinction comparison. It uses the displayed cell axes,
wavelength and zero correction, without refining the cell or converting its
basis. Alternative monoclinic and rhombohedral settings therefore require a
cell expressed in those axes. Selecting a different solution clears the manual
hypothesis. Groups with identical extinctions cannot be distinguished by this
comparison.


## Workspace and projects

The title bar holds **About** (shown in the right panel), **⇄ Panels**,
the **File** and **Solution** menus, **Report**, **Index** and a Dark switch.
**File** contains pattern loading, pattern export, project saving/loading and
unloading. **Solution** acts on the solution open in the Solutions tab and is
disabled without one: **Refine** (Monte-Carlo), **Test Space Group** (Space Group
MC), **Convert…** (to a maximal subgroup, translationengleiche or klassengleiche,
or to another group; the subgroup lists come from `sg_subgroups.js` and
`sg_convert.js`, the cell is converted and its M(20) is the parent's),
**Explore Space Group…**, **Export pdCIF** and **Erase**.

The left panel is **Data** (pattern and its metadata, 2θ range, radiation, search
parameters), **Peaks**, **Indexing** (crystal systems, GPU parameters) and
**Solutions** (one accordion per solution, one open at a time, with symmetry,
cell, volume, M(20), Niggli cell and space group).

While a search runs, **Index** turns into a red **Stop** (Esc does the same) and
everything else is locked until it ends; the plot, Log and History stay readable.

The plot has Linear / √I / log I, then Background, **Data** (the experimental
pattern), **Exp** (experimentally detected peaks), **Sol** (reflections of the open
solution) and **Sim** (reflections simulated for the space group of the open solution:
the one chosen for it, or the first group Test Space Group proposed), Save PNG and a
mouse-help tooltip. Opening a solution shows its Sol and Sim markers; closing it
deselects it, which also switches the Solution menu off. A solution can hold several
space groups; the radio button beside each one picks the group drawn as Sim.
**Report** prints the open solution only, or all listed solutions when none is open.
The Log tab narrates each run: parameters, the plan, time and candidate cells of every
crystal system, the sieve and the space-group analysis. The status bar is one line: the
current status under the controls, the position under the plot pointer where the
plot starts (at the left when the panels are swapped), then the open solution and
the version (right).

The version is written once, in `js/core/version.js` (`APP_VERSION`).
`index.html` loads it first and stamps it on every script, worker, shader and the
space-group file, and the window shows it at the bottom right.

**Plot**, **Log** and **History** select result views. History records indexing
runs and can restore their pattern, peaks, parameters and solutions (the Restore
button is at the right of each entry). At 100 MB of history the program offers, once, to remove the oldest runs.
It asks before saving a project larger than 10 MB, and before saving a report of
more than 100 pages (these two only when you save). Every dialog and notice follows the light or dark theme
of the window. Messages
and run summaries appear in Log. Green manual space-group ticks have the same
position/HKL tooltips as blue calculated ticks. In **Snap: HKL** mode, a manual
comparison uses its allowed reflections as snap targets.

**File > Save project** downloads a `.brutus` project containing the original
and working patterns, edited peak list, solutions and selection, input settings,
manual space-group comparison, plot bounds, log and run history. **Open
project** restores it without rerunning peak detection or indexing. Projects
are local JSON files; no data is uploaded. Finish or stop an indexing run before
opening or saving a project. Pattern export remains a separate File command.
