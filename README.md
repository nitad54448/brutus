# Technical Overview & Methodology

<a id="introduction"></a>
This document is a technical reference for the Brutus powder indexing software. It explains the main algorithms, search parameters, and methodology, and is intended for users familiar with powder X-ray diffraction. As the name of the program suggests it, this is a brute force method (force is not necessarily smart).

### Core Goal

The aim of *ab initio* powder indexing is to determine the unit-cell parameters
($a, b, c, \alpha, \beta, \gamma$) from a list of observed diffraction peak positions ($2\theta$).
Brutus performs this task using a system-specific, exhaustive search algorithm.

The central assumption is that a small subset of the most intense, low-angle reflections corresponds
to simple crystal planes with low-integer Miller indices $(hkl)$. For a given crystal system,
the program chooses exactly as many observed peaks as there are unknown lattice parameters, and solves
the resulting system of linear equations.

### Q-Space Formulation

All peak positions are first converted from $2\theta$ to Q-space, where $Q = 1/d^2$.
The general quadratic relationship between $Q$, the Miller indices, and the reciprocal cell parameters
($A, B, C, D, E, F$) is

$$Q_{hkl} = Ah^2 + Bk^2 + Cl^2 + Dkl + Ehl + Fhk$$

Brutus solves for these reciprocal parameters (or the subset relevant to the current crystal system),
then converts them to real-space cell parameters. Each candidate cell is refined immediately and scored
against the full peak list.

---

<a id="quick-start"></a>
## Quick Start Guide

Use the following workflow for a typical single-phase powder pattern.

1.  **Load the data file.**
    Click `Select Data File`. Supported formats include generic 2-column text (`.xy`, `.csv`, `.txt`), as well as `.xrdml`, `.brml`, `.ras`, `.uxd`, `.udf`, and GSAS files (`.esd`, `.xra`).
2.  **Detect peaks.**
    On the **Peaks** tab, adjust the `Min peak (%)`, `Radius (pts)`, and `Points`
    sliders until the automatically detected peaks match the visual pattern.
3.  **Curate the peak list.**
    Carefully review all peaks:
    * Edit $2\theta$ positions for accuracy.
    * Delete spurious peaks (noise, Kα<sub>2</sub> shoulders if you are not stripping Kα<sub>2</sub>).
    * Add any missing reflections using `Ctrl + Click` on the chart.
    
    A clean list of about 15–20 peaks, free of impurities at low angle, is ideal.
4.  **Set parameters.**
    On the **Parameters** tab:
    * Select the correct X-ray `Radiation Preset` (e.g. Cu Kα, Co, Fe, Mo).
    * Choose whether to enable `Strip K-alpha2`. This also updates the `Ka1 Wavelength` field. The default is OFF (average Kα wavelength).
    * Set a chemically reasonable `Max Volume (Å³)` to limit the search space.
    * Set `2θ Error (°)` according to your data quality (e.g. ≈0.02° for synchrotron, ≈0.05° for a typical lab diffractometer).
    * Leave **Refine Zero-Point Error** enabled unless you have a specific reason to turn it off.
    * Select the crystal systems to search. Enabling Orthorhombic, Monoclinic, or Triclinic activates GPU-accelerated searches. **Note:** These three low-symmetry systems are mutually exclusive; selecting one will automatically uncheck the others.
5.  **(Optional) Tune GPU Parameters.**
    If searching a GPU-accelerated system, a new box appears.
    * `HKL Basis Size`: (Defaults: Ortho 300 / Mono 100 / Tri 40).
    * `Peaks to Combine`: (Defaults: Ortho 7 / Mono 7 / Tri 9).
    * `FoM Tolerance`: (Default: 0.8) The filter strictness for the GPU pre-check.
    * `Candidates (kCells)`: (Default: 50) The size of the return buffer (50 = 50,000 cells).
    * Leave these at their defaults unless you have trouble finding a solution.
6.  **Start indexing.**
    Click `Start Indexing`. Progress is shown on the main bar.
7.  **Inspect solutions.**
    On the **Solutions** tab:
    * Sort solutions by M(20).
    * Click a row to display calculated (blue) and observed (red) tick marks on the chart.
    * A plausible solution will show excellent alignment and reasonable space-group suggestions.

---

<a id="ui"></a>
## The User Interface

The application window is divided into a **Controls Panel** (left) and a **Results Area** (right).

### Controls Panel

The controls are organized into three main tabs.

#### 1. Peaks Tab

* **Peak finding sliders:**
    * `Radius` – background subtraction radius (rolling-ball algorithm).
    * `Points` – Savitzky–Golay smoothing window width.
    * `Min peak (%)` – peak detection threshold on a logarithmic scale.
* **2θ range sliders:**
    Restrict the angular window for peak finding, for example to exclude noisy low-angle or high-angle regions.
* **Peak table:**
    Lists all detected peaks. The `2θ Obs (°)` column is editable to allow fine corrections.

#### 2. Parameters Tab

* **Radiation Preset:**
    Select the X-ray source (e.g. Cu Kα, Co Kα, Fe, Mo). This sets the internal wavelength values based on Bearden (1967).
* **Ka1 Wavelength (Å):**
    Displays the wavelength used for calculations. It is updated automatically when a preset is chosen and
    depends on the `Strip K-alpha2` setting. It becomes editable only when a `Custom` preset is selected.
* **Strip K-alpha2:**
    Applies a **Damped Richardson-Lucy** deconvolution to remove the Kα<sub>2</sub> component before peak analysis.
    This iterative Bayesian method is superior to traditional subtraction as it enforces positivity (no negative artifacts).
    When enabled, the `Ka1 Wavelength` field is set to the pure Kα<sub>1</sub> value.
    The default is OFF.
* **Max Volume (Å³):**
    Upper bound on the allowed unit-cell volume. This is a strong constraint on the search.
* **Impurity Peaks:**
    Number of unindexed peaks allowed among the first 20 when computing M(20).
* **2θ Error (°):**
    Matching tolerance between observed and calculated peak positions. Default is 0.06°.
* **Refine Zero-Point Error:**
    When enabled, a full zero-point refinement is performed for the final solution.
* **Crystal systems to search:**
    Checkboxes for Cubic, Tetragonal, Hexagonal, Orthorhombic, Monoclinic, and Triclinic.

#### GPU Search Parameters

When `Orthorhombic`, `Monoclinic`, or `Triclinic` is selected, this section appears. These systems are mutually exclusive (only one GPU task runs at a time).

* **HKL Basis Size ($N_{hkl}$):**
    The size of the pool of theoretical Miller indices used to construct trial matrices.
    *(Defaults: 300 for Ortho, 100 for Mono, 40 for Tri).*
* **Peaks to Combine ($N_{peaks}$):**
    Controls how many observed peaks (starting from the first one) are considered for generating the solving tuple. Increasing this provides robustness against impurities ("sliding window" effect) but increases search time factorially.
    *(Defaults: 7 for Ortho, 7 for Mono, 9 for Tri).*
* **FoM Tolerance:**
    The "Fail-Fast" threshold for the GPU's internal filter. A value of 1.0 means the average error of the trial cell matches your `2θ Error`. Default is 0.8.
* **Buffer (kCells):**
    The maximum number of candidate solutions (in thousands) the GPU allows. If the search finds more candidates than this limit (default 50 = 50,000), the search stops early to prevent UI freezing.

#### 3. Solutions Tab

* **Solutions table (The Ledger):**
    Shows valid solutions found during your session. New solutions are added to this list and ranked by M(20). The list is preserved until a new file is loaded.
* **Context Menu:**
    Right-click any solution to **Erase** it from the list or generate a specific **PDF Report**.

### Results Area

* **Chart:**
    Shows the experimental diffraction pattern, observed peaks (red ticks) and calculated peaks for the
    selected solution (blue ticks). A good solution exhibits visually convincing overlap.
* **Chart interaction:**
    * Zoom: mouse wheel (different zoom type if the mouse at the left of Y axis or below the Y axis).
    * Pan: click and drag.
    * Reset zoom: right-click.
    * Add peak: `Ctrl + Click`.

---

<a id="peak-finding"></a>
## Peak Finding in Detail

Accurate peak positions are the single most important input for successful indexing.
Brutus uses a multi-step procedure to detect peaks from raw intensity data.

### Algorithm Steps

1.  **Kα<sub>2</sub> stripping (optional):**
    If `Strip K-alpha2` is enabled, the **Richardson-Lucy** algorithm is applied to the raw intensities.
2.  **Background subtraction:**
    A rolling-ball style algorithm estimates and removes the background.
    The `Radius` slider controls the ball radius.
3.  **Data smoothing:**
    A Savitzky–Golay filter is applied to the background-subtracted signal to reduce noise
    while preserving peak shapes. The `Points` slider sets the window width.
4.  **Initial peak detection:**
    Local maxima above the `Min peak (%)` threshold are identified.
5.  **Position refinement:**
    For each peak, a five-point least-squares quadratic fit (based on Savitzky–Golay coefficients) is carried out around
    the maximum to obtain a the position. If too close to the data edge, the algorithm falls back to a three-point fit.

### Practical Recommendations

* Start from the default slider values and inspect the result visually. Only the peaks shown in the table will be used in the calculations.
* If weak but real peaks are missed, reduce `Min peak (%)`. If noise is detected as peaks, increase it.
* For broad, slowly varying backgrounds, increase `Radius`.
* For noisy data, increase `Points` (smoothing), but avoid over-smoothing, which can merge or shift peaks.
* Always manually curate the final peak list. Remove artifacts and known impurity peaks.
    If you are not stripping Kα<sub>2</sub>, delete Kα<sub>2</sub> shoulders explicitly.
* Kα<sub>2</sub> stripping can simplify the pattern but may introduce small artifacts.
    If indexing fails with stripping ON, try turning it OFF and manually cleaning the peak list.

---

<a id="indexing-method"></a>
## Indexing Algorithm and Search Parameters

Brutus uses an exhaustive, symmetry-specific trial-and-refine indexing algorithm.
For each crystal system, the program generates trial solutions from combinations of the lowest-angle peaks and low-index Miller indices, solves the corresponding linear system of equations in reciprocal space, rejects unphysical cells, and finally performs a full refinement.

### Linear System Formulation

The search is formulated as a system of linear equations:

$$Q_{obs} = \sum P_i \cdot H_i$$

where $Q_{obs}$ are the observed $1/d^2$ values, $H_i$ are terms derived from trial Miller indices
(e.g. $h^2$, $k^2$, $l^2$), and $P_i$ are the reciprocal lattice parameters (e.g. $A = 1/a^2$, $B = 1/b^2$, …).

### CPU Searches (Cubic, Tetra, Hexa)

* **Cubic (1 parameter, $A = 1/a^2$)**<br>
    Solves a 1×1 system by iterating through the first 12 observed peaks and assigning trial $(hkl)$ vectors (integers up to 8), this requires 2448 direct solves.

* **Tetragonal & Hexagonal (2 parameters)**<br>
    Solves all the combinations of a 2×2 system using pairs of peaks from the first 12, assigning pairs of trial $(hkl)$ vectors up to 5; ~34 million tests.

### GPU-Accelerated Searches (Orthorhombic, Monoclinic & Triclinic)

For low-symmetry systems and orthorhombic, the number of combinations becomes too large for the CPU. These searches are offloaded to the WebGPU. Their scope is now user-configurable.

#### GPU Search Parameters

The total number of cells tested (which is displayed in the status text) is:
**Total Cells = $C(N_p, k) \times C(N_h, k) \times k!$**

where:
* $N_p$ is `Peaks to Combine`.
* $N_h$ is `HKL Basis Size`.
* $k$ is the number of parameters ($3, 4, 6$).
* $k!$ is the number of permutations ($6, 24, 720$).

#### Orthorhombic (3 parameters, $A, B, C$)

* **Logic:** Solves a 3x3 system $M \cdot \vec{x} = \vec{q}$ for $\vec{x} = [A, B, C]$.
* **Defaults:** $C(7, 3) \times C(300, 3) \times 6 \approx$ **936 million** trial cells.

#### Monoclinic (4 parameters, $A, B, C, D$)

* **Logic:** Solves a 4x4 system $M \cdot \vec{x} = \vec{q}$ for $\vec{x} = [A, B, C, D]$.
* **Defaults:** $C(7, 4) \times C(100, 4) \times 24 \approx$ **3.3 billion** trial cells.

#### Triclinic (6 parameters, $A \dots F$)

* **Logic:** Solves the full 6x6 system $M \cdot \vec{x} = \vec{q}$ for $\vec{x} = [A, B, C, D, E, F]$.
* **Defaults:** $C(9, 6) \times C(40, 6) \times 720 \approx$ **232 billion** trial cells.

#### Two-Stage GPU Filtering

Most of these candidate cells are discarded on the GPU before they reach the CPU:

1.  **Stage 1 – Basic filter (`extractCell`):**
    A candidate is immediately rejected if the system is singular, parameters are unphysical (e.g., $a < 2Å$), angles are invalid (e.g., $\beta > 150^\circ$), or the volume exceeds `Max Volume`.
2.  **Stage 2 – Mini Figure of Merit:**
    For surviving cells, the GPU computes a fast Figure of Merit by comparing the first 10 observed peaks to the cell's theoretical pattern. Only cells that pass the `FoM Tolerance` are saved to the buffer.

<a id="tuning-the-search"></a>
### Tuning the Search & Troubleshooting

#### If no solutions (or only poor ones) are found

* **Re-examine the peak list.** This is the dominant failure mode. Check that the first 10–15 peaks belong to a single phase, are free of impurities, and have accurate $2\theta$ positions.
* **Relax `2θ Error`.** If this value is too strict for your data resolution, valid solutions may be discarded.
* **Increase `Max Volume`.** The true cell may be larger than initially expected.
* **Increase GPU Parameters.** Try increasing `HKL Basis Size` or `Peaks to Combine`.

#### If you get too many solutions (GPU buffer fills)

The GPU buffer is limited (default 50,000) candidate solutions. If this buffer fills up, the search stops earlier than the full space requested.

* **Tighten `2θ Error`.** This is the most effective way to reduce the number of false positives.
* **Reduce `Max Volume`.** This is another strong constraint to remove unphysical large cells.
* **Decrease GPU Parameters.** Reducing `HKL Basis Size` or `Peaks to Combine` will run a smaller, faster search.

#### GPU Buffers and Chunking

The total list of HKL combinations can be too large for a single GPU buffer. Brutus therefore generates HKL combinations in large JavaScript chunks and further splits each of these into small “dispatch chunks” (e.g. 256 workgroups) before sending them to the GPU. This avoids both memory overflow and TDR (Timeout Detection and Recovery) crashes.

---

<a id="evaluating-solutions"></a>
## Evaluating Solutions

The indexing search usually produces several candidate cells. Brutus keeps at most the best 50, ranked by M20.

### de Wolff Figure of Merit: M(20)

| M(20) value | Interpretation                                              |
| :---------- | :---------------------------------------------------------- |
| > 20        | Very likely correct.                                        |
| > 10        | Likely correct, provided the cell volume is chemically plausible. |
| 5–10        | Plausible; requires further inspection.                     |
| < 5         | Probably spurious; treat with caution.                      |

### F(N) Figure of Merit

As a complementary metric, Brutus computes the **F(N)** figure of merit, usually with N = 20.
$$F_N = \frac{N}{\langle |\Delta(2\theta)| \rangle \cdot N_{calc}}$$
A high F(20) indicates a precise fit with low average error.

### Zero-Shift Logic

If two cells are geometrically similar, but one has a refined zero-shift and the other does not (e.g., fixed at 0), Brutus treats them as distinct solutions and preserves both in the list.

---

<a id="space-group-analysis"></a>
## Space Group Analysis

After a high-quality unit cell is obtained, Brutus suggests likely space groups based on systematic absences.

1.  **Generate unique reflections** for the refined cell.
2.  **Index observed peaks** against the theoretical pattern.
3.  **Build a high-confidence subset** (unambiguous peaks only).
4.  **Determine extinctions** by checking if any unambiguous peak violates a centering or glide rule.
5.  **Rank space groups** by violation count (0 violations is ideal).

---

<a id="references"></a>
## References

Brutus was developed by Nita Dragoe at Université Paris-Saclay (2024–2026). If you use Brutus in your work, please cite:
[https://doi.org/10.13140/RG.2.2.13443.57126](https://doi.org/10.13140/RG.2.2.13443.57126)

1.  **M(20):** de Wolff, P. M. (1968). *J. Appl. Cryst.* **1**, 108–113.
2.  **F(N):** Smith, G. S. & Snyder, R. L. (1979). *J. Appl. Cryst.* **12**, 60–65.
3.  **Richardson-Lucy Deconvolution:** Richardson, W. H. (1972). *J. Opt. Soc. Am.* **62**, 55–59; Lucy, L. B. (1974). *Astron. J.* **79**, 745.
4.  **Previous software:** Dragoe, N. (2001). *J. Appl. Cryst.* **34**, 535.

---
*Last updated: 17 January 2026.*