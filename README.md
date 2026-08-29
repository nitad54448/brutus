# Brutus

Ab initio indexing of powder diffraction patterns, in a browser tab.

You give it a list of peak positions; it works out the unit cell. The name is
honest about the method — it is a brute-force search, and force is not
necessarily smart. But GPUs are very good at doing dumb things quickly, so for
orthorhombic, monoclinic and triclinic patterns it exhausts a search space that
would be impractical on a CPU, and for the high-symmetry systems it just
enumerates the answer directly.

Brutus runs entirely on your machine. Nothing is uploaded anywhere, there is no
account, and there is no install step.

---

## Running it

This program can be run by accessing <https://nitad54448.github.io/brutus/brutus.html>.

If you want, you can copy all these files in a folder of your choice, then run it directly. This is a static site, so any web server will work, you can launch one in Visual Studio or by:

```bash
cd brutus
python -m http.server 8000
# then open http://localhost:8000/brutus.html
```

Opening `brutus.html` straight off the disk (`file://`) will *not* work — the
app fetches the shaders and the space-group database at runtime, and browsers
block that for local files.

For the low-symmetry systems you need a browser with **WebGPU** (recent Chrome,
Edge, or Safari 18+ with graphic acceleration set to ON). You have probably a GPU, so if it does not work it is the settings of your browser, rather than the device. I tested this program on many devices, even on a Android phone. For indexing Cubic, tetragonal and hexagonal this program uses the CPU in a Web Worker and work anywhere. 

---

## How it works

The program will read a data file and then you detect peak positions. Peak positions are converted to Q-space, where $Q = 1/d^2$, because the
relationship between $Q$ and the Miller indices is linear in the reciprocal cell
parameters:

$$Q_{hkl} = Ah^2 + Bk^2 + Cl^2 + Dkl + Ehl + Fhk$$

The assumption underneath everything is that the strongest low-angle
reflections have small integer indices. So the program pick some observed peaks, guess an $(hkl)$ for each, solve the linear system, and see what falls out. A trial that survives a cheap filter is then refined properly by weighted least squares — including the zero-point error — and scored against the
whole peak list with M(20) and F(N).

The interesting part is the filtering, because the search generates enormous
numbers of candidates and almost all of them are nonsense. Each GPU thread
solves one system, throws away anything geometrically implausible, and scores
the survivor against the first ten peaks. Very few cells survuve this test, but since the program will test about 10 millions cells per second, it will probably find a valid cell, if peaks, volume and system are correctly selected.

The full methodology with the system parameterisations, the figure-of-merit
definitions, the weighting scheme, the two-round zero-point strategy, the
space-group statistics — is in **`brutus_help.html`**, which is more thorough than this file and is the place to look if you want to know why
something behaves the way it does.

---

## Quick start

1. **Load a data file.** Two-column text (`.xy`, `.csv`, `.txt`), or `.xrdml`,
   `.brml`, `.ras`, `.uxd`, `.udf`, `.esd`, `.xra`.
2. **Detect peaks.** Adjust `Min peak (%)`, `Radius (pts)` and `Points` until
   the marks match what you see.
3. **Curate the peak list.** This is the step that decides whether indexing
   works. Fix positions, delete impurity lines and Kα₂ shoulders, add anything
   the detector missed with `Ctrl + Click`. Fifteen to twenty clean peaks, with
   nothing spurious at low angle, is the target.
4. **Set parameters.** Radiation preset, `Strip K-alpha2` if you want it,
   a chemically sensible `Max Volume`, and a `2θ Error` that matches your data
   (≈0.02° synchrotron, ≈0.05° typical lab). Then pick the crystal systems.
   Orthorhombic, monoclinic and triclinic are mutually exclusive — selecting one
   unselects the others.
5. **Index.** Sort the solutions by M(20) and click a row to overlay the
   calculated tick marks on your pattern.

---

## When it finds nothing

Almost always the problem is the peak list. Check that the first ten to fifteen lines really
do belong to one phase and that their positions are accurate.

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

Since v2026-08-28 the app tells you which of these it was. A run that finds nothing
now reports the volume range of the candidates it saw and how many peaks the
best of them kept inside the error budget, so "no solutions" comes with a reason
and a setting to change.

The GPU parameters are per-system and rarely need touching:

| | Orthorhombic | Monoclinic | Triclinic |
|---|---|---|---|
| HKL Basis Size | 300 | 100 | 40 |
| Peaks to Combine | 7 | 7 | 9 |

`FoM Tolerance` (1.5) and `Candidates` (50 = 50 000 cells) are shared. If the
candidate buffer fills, the search stops early — tighten `2θ Error` or reduce
`Max Volume` rather than enlarging the buffer.

---

## Space groups

Once you have a cell, Brutus works out which space groups are compatible with
the systematic absences. Two things do this, and they answer different
questions.

The **automatic analysis** runs on every solution and produces a compatibility
list: which settings the observed reflections contradict, and by how many. It is
grouped by violation count, not ranked — it tells you what the data rule out,
not which survivor is likeliest.

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

---

## The space-group database

`sg_ops.json` holds all **530 settings of the 230 space groups** — every
symmetry operator, the zone definitions, and the printed reflection conditions —
in about 240 KB. Rotation matrices are dictionary-encoded against a shared table
of the 64 distinct ones.

The sg_ops.json is generated directly from [cctbx](https://cctbx.github.io/):

```bash
python build_sg_db.py --out sg_ops.json
```

You only need this if you want to rebuild the database; a working copy ships
with the repository. It requires a cctbx environment — nothing else in Brutus
does.

The zones and conditions are *derived from* the operators rather than copied
from a table, so they cannot drift out of step with what the application
actually uses, and every setting is verified before the file is written. Two
modes let you check it without trusting me:

```bash
python build_sg_db.py --self-test        # checks the maths, no cctbx needed
python build_sg_db.py --check sg_ops.json # re-verifies a finished file
node check_sg_ops.mjs sg_ops.json         # checks the app can consume it
```

`--self-test` builds nine space groups by closing published generators, checks
each group's order against its published value first (so a typo in a generator
fails loudly rather than quietly testing the wrong group), then derives the
reflection conditions and compares them against the International Tables.

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
| `main_app.js` | UI, chart, file parsing, orchestration |
| `worker-logic.js` | the crystallography: HKL generation, least squares, figures of merit, Niggli reduction, space-group analysis. Loaded three ways — main thread, CPU index worker, and inside each refinement worker |
| `webgpu-engine.js` | WebGPU device, buffers, dispatch chunking, combinadics |
| `refinement-worker.js` | batch refinement; a pool of these runs alongside the GPU search |
| `*_solver.wgsl` | the three compute kernels |
| `sg_ops.json` | the space-group database |
| `styles.css`, `inter-font.css`, `Inter-Variable.ttf`, `tex-svg.js`, `scripts/` | styling, fonts, MathJax, and the vendored libraries |

**Tooling** — not deployed, not needed to run anything:

| Script | |
|---|---|
| `build_sg_db.py` | builds `sg_ops.json` from cctbx |
| `check_sg_ops.mjs` | validates the database against the application |
| `bump_version.py` | stamps one cache-busting `?v=` across `brutus.html` |
| `test_sg_ops.mjs` | derives reflection conditions from the shipping operator code and checks them against the International Tables |
| `check_pipeline.mjs` | verifies `main_app.js`, the engine and the shaders still agree on how the HKL basis is packed |

The last two are worth running after touching the code they cover, because both
failures are otherwise invisible: the run completes, every candidate cell is
nonsense, and you get no solutions and no error.

### A note on the browser cache

`worker-logic.js` is fetched under three different URLs, and each is a separate
cache entry. A partial version bump can leave the main thread and the workers
running *different builds of the same file*, which does not present as a caching
problem — it presents as the results table and the PDF report disagreeing. Run
`python bump_version.py` after any change; it sets every `?v=` together and
warns if they have drifted apart. The `.wgsl` shaders are still fetched
unversioned, so a shader change also requires a hard reload .

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

*Last updated: 29 August 2026.*
