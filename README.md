# Technical Overview & Methodology

<a id="introduction"></a>
This document is a technical reference for the Brutus powder indexing software. It explains the main algorithms, search parameters, and methodology, and is intended for users familiar with powder X-ray diffraction. As the name of the program suggests it, this is brute force method (force is not necessarily smart).

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
    Click `Select Data File`. Supported formats include `.xy`, `.xrdml`, `.ras`, and others.
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
    * Select the correct X-ray `Radiation Preset` (e.g. Cu Kα).
    * Choose whether to enable `Strip K-alpha2`. This also updates the `Ka1 Wavelength` field. The default is OFF (average Kα wavelength).
    * Set a chemically reasonable `Max Volume (Å³)` to limit the search space.
    * Set `2θ Error (°)` according to your data quality (e.g. ≈0.02° for synchrotron, ≈0.05° for a typical lab diffractometer).
    * Leave **Refine Zero-Point Error** enabled unless you have a specific reason to turn it off.
    * Select the crystal systems to search. Enabling Orthorhombic, Monoclinic, or Triclinic activates GPU-accelerated searches.
5.  **(Optional) Tune GPU Parameters.**
    If searching a GPU-accelerated system, a new box appears.
    * `HKL Basis Size`: (Default: 300/80/40) Number of simple HKLs to use for the search.
    * `Peaks to Combine`: (Default: 5/7/8) Number of observed peaks to use for combinations.
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
    Select the X-ray source (e.g. Cu Kα, Co Kα). This sets the internal wavelength values.
* **Ka1 Wavelength (Å):**
    Displays the wavelength used for calculations. It is updated automatically when a preset is chosen and
    depends on the `Strip K-alpha2` setting. It becomes editable only when a `Custom` preset is selected.
* **Strip K-alpha2:**
    Applies a vanCittert-type correction to remove the Kα<sub>2</sub> component before peak analysis.
    When enabled, the `Ka1 Wavelength` field is set to the pure Kα<sub>1</sub> value.
    The default is OFF.
* **Max Volume (Å³):**
    Upper bound on the allowed unit-cell volume. This is a strong constraint on the search.
* **Impurity Peaks:**
    Number of unindexed peaks allowed among the first 20 when computing M(20).
* **2θ Error (°):**
    Matching tolerance between observed and calculated peak positions.
* **Refine Zero-Point Error:**
    When enabled, a full zero-point refinement is performed for the final solution.
* **Crystal systems to search:**
    Checkboxes for Cubic, Tetragonal, Hexagonal, Orthorhombic, Monoclinic, and Triclinic.

#### GPU Search Parameters

When `Orthorhombic`, `Monoclinic`, or `Triclinic` is selected, this section appears, allowing you to control the scope of the GPU-accelerated search.

* **HKL Basis Size:**
    Controls how many simple HKL triplets (from a pre-generated list) are used in the search combinations.
    *(Default: 300 for Orthorhombic, 80 for Monoclinic, 40 for Triclinic. Range: 10-600)*
* **Peaks to Combine:**
    Controls how many of your observed peaks (from the top of the peak list) are used to generate combinations.
    *(Default: 5 for Orthorhombic, 7 for Monoclinic, 8 for Triclinic. Min: 3/4/6)*

When you select a GPU system, the status text at the bottom of the panel will show you the total number of cells to be tested, which is a direct result of these two parameters. Be careful, this number can increase very fast.

#### 3. Solutions Tab

* **Solutions table:**
    Shows each valid solution with crystal system, unit-cell parameters, volume and M(20).
    Selecting a row updates the main chart.
    After the search finishes, you can filter visible solutions by enabling or disabling crystal systems.

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
    If `Strip K-alpha2` is enabled, the van Cittert algorithm is applied to the raw intensities.
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
For each crystal system, the program generates trial solutions from combinations of the lowest-angle peaks and low-index Miller indices, solves the corresponding linear system of equations in reciprocal space, rejects unphysical cells, and finally performs a full refinement and figure-of-merit evaluation.

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

You can now control the two key inputs for the GPU search:

* `<strong>Peaks to Combine</strong>`: This sets $N_p$, the number of observed peaks from your list used to generate combinations. The program calculates $C(N_p, k)$, where $k=3$ for orthorhombic, $k=4$ for monoclinic, and $k=6$ for triclinic.
    *(Defaults: $N_p=5$ for Orthorhombic, $N_p=7$ for Monoclinic, $N_p=8$ for Triclinic)*
* `<strong>HKL Basis Size</strong>`: This sets $N_h$, the number of simple HKL vectors used as the "basis set". The program calculates $C(N_h, k)$.
    *(Defaults: $N_h=300$ for Orthorhombic, $N_h=80$ for Monoclinic, $N_h=40$ for Triclinic)*

> **Note on HKL Basis Generation (Axial Promotion)**
>
> For the Orthorhombic and Monoclinic systems, the HKL basis is generated with a special "promotion" step. All simple axial reflections like (H00), (0K0), and (00L) are moved to the front of the list *before* it is sliced to the `HKL Basis Size`.
>
> This ensures that essential reflections for long axes (e.g., `(0 0 10)`) are not accidentally cut off, which dramatically improves the indexing stability for highly skewed or anisotropic cells.

The total number of cells tested (which is displayed in the status text) is:
**Total Cells = $C(N_p, k) \times C(N_h, k) \times k!$**

where the number of permutations is defined by k: $k=3$ ($k!=6$) for Orthorhombic, $k=4$ ($k!=24$) for Monoclinic, and $k=6$ ($k!=720$) for Triclinic.

#### Orthorhombic (3 parameters, $A, B, C$)

* **Logic:** Solves a 3x3 system $M \cdot \vec{x} = \vec{q}$ for $\vec{x} = [A, B, C]$.
* **GPU Work:** Each GPU thread tests one (peak combo, HKL combo) pair, and loops 6 times (3!) to test all permutations.
* **Defaults:** $C(5, 3) \times C(300, 3) \times 6 \approx$ **268 million** trial cells.

#### Monoclinic (4 parameters, $A, B, C, D$)

* **Logic:** Solves a 4x4 system $M \cdot \vec{x} = \vec{q}$ for $\vec{x} = [A, B, C, D]$.
* **GPU Work:** Each GPU thread tests one (peak combo, HKL combo) pair, and loops 24 times (4!) to test all permutations.
* **Defaults:** $C(7, 4) \times C(80, 4) \times 24 \approx$ **1.33 billion** trial cells.

#### Triclinic (6 parameters, $A \dots F$)

* **Logic:** Solves the full 6x6 system $M \cdot \vec{x} = \vec{q}$ for $\vec{x} = [A, B, C, D, E, F]$.
* **GPU Work:** Each GPU thread tests one (peak combo, HKL combo) pair, and loops 720 times (6!) to test all permutations.
* **Defaults:** $C(8, 6) \times C(40, 6) \times 720 \approx$ **77.4 billion** trial cells.

#### Two-Stage GPU Filtering

Most of these candidate cells are discarded on the GPU before they reach the CPU:

1.  **Stage 1 – Basic filter (`extractCell`):**
    A candidate is immediately rejected if the system is singular, parameters are unphysical (e.g., $a < 2Å$), angles are invalid (e.g., $\beta > 150^\circ$), or the volume exceeds `Max Volume`.
2.  **Stage 2 – Mini Figure of Merit (`validate_fom_avg_diff`):**
    For surviving cells, the GPU computes a fast Figure of Merit by comparing the first 20 observed peaks to the cell's theoretical pattern. Only cells that pass a strict internal threshold are sent back to the CPU.

<a id="tuning-the-search"></a>
### Tuning the Search & Troubleshooting

If you obtain no solutions, or too many, consider the following adjustments.

#### If no solutions (or only poor ones) are found

* **Re-examine the peak list.**
    This is the dominant failure mode. Check that the first 10–15 peaks belong to a single phase,
    are free of impurities, and have accurate $2\theta$ positions.
* **Relax `2θ Error`.**
    If this value is too strict for your data resolution, valid solutions may be discarded.
* **Increase `Max Volume`.**
    The true cell may be larger than initially expected.
* **Increase GPU Parameters.**
    Try increasing `HKL Basis Size` (e.g., from 80 to 120) or `Peaks to Combine` (e.g., from 7 to 9). This will significantly increase the search time but covers more combinations.

#### If you get too many solutions (GPU buffer fills)

The GPU buffer is now limited to 50,000 candidate solutions. If this buffer fills up, the search stops earlier than the full space requested. If you have no solution that appears valid:

* **Tighten `2θ Error`.**
    This is the most effective way to reduce the number of false positives.
* **Reduce `Max Volume`.**
    This is another strong constraint to remove unphysical large cells.
* **Decrease GPU Parameters.**
    Reducing `HKL Basis Size` or `Peaks to Combine` will run a smaller, faster search, but may miss the solution if it relies on a higher-index HKL or a peak further down the list.

#### GPU Buffers and Chunking

The total list of HKL combinations can be too large for a single GPU buffer. Brutus therefore:

1.  Generates HKL combinations in large JavaScript chunks in `brutus.html`.
2.  Further splits each of these into small “dispatch chunks” (e.g. 256 workgroups) in
    `webgpu-engine.js` before sending them to the GPU.

This two-level chunking avoids both memory overflow and long-running GPU commands.

> **Performance: WebGPU vs CPU**
>
> Cubic, Tetragonal, and Hexagonal searches run on the CPU (in a Web Worker) and are typically
> very fast (under a second).
>
> Orthorhombic, Monoclinic, and Triclinic searches are offloaded to the GPU. This allows testing of billions of trial cells in a time frame (few minutes for 1 billion cells) that would be
> impractical on the CPU.
>
> Note: at the end of a GPU search all candidates (up to 50 000) are passed to refinement, Niggli cell reduction, M(20) and F(N) calculations. These are made on the CPU so it might take a few minutes, be patient.

---

<a id="evaluating-solutions"></a>
## Evaluating Solutions

The indexing search usually produces several candidate cells. Brutus keeps at most the best 50, ranked by M20, after the search.
Selecting the correct one requires interpreting figures of merit and checking the refined fit.

### de Wolff Figure of Merit: M(20)

The primary ranking indicator is the **de Wolff Figure of Merit, M(20)**, calculated from
the first 20 observed reflections. It combines both positional accuracy and completeness. In the PDF report additional values are listed, for instance F(20).

| M(20) value | Interpretation                                              |
| :---------- | :---------------------------------------------------------- |
| > 20        | Very likely correct.                                        |
| > 10        | Likely correct, provided the cell volume is chemically plausible. |
| 5–10        | Plausible; requires further inspection.                     |
| < 5         | Probably spurious; treat with caution.                      |

### F(N) Figure of Merit

As a complementary metric, Brutus computes the **F(N)** figure of merit, usually with N = 20 (F(20)).
While M(20) emphasizes the completeness of indexing, F(N) focuses on the average positional accuracy.

$$F_N = \frac{N}{\langle |\Delta(2\theta)| \rangle \cdot N_{calc}}$$

* $N$ – number of observed lines used (e.g. 20).
* $\langle |\Delta(2\theta)| \rangle$ – mean absolute difference between observed and calculated $2\theta$ for those lines.
* $N_{calc}$ – number of theoretical reflections (observed or not) up to the $2\theta$ of the $N$-th observed line.

A high F(20) indicates a precise fit with low average error. A solution with both high M(20) and high F(20)
is usually reliable.

### Least-Squares and Zero-Point Refinement

For each promising candidate, Brutus performs a two-stage refinement.

#### Stage 1: Internal zero-point correction

First, a constrained refinement is carried out including a zero-point error parameter.
This internal zero correction is limited by the user-defined `2θ Error`, which allows
the algorithm to compensate for modest alignment errors without overfitting. The corrected
peak positions are then used to refine a stable baseline cell.

#### Stage 2: Final refinement (optional)

If the Stage-1 cell achieves a sufficiently high M(20) and the
`Refine Zero-Point Error` option is enabled, a second, full refinement (with an
unconstrained zero-point) is performed. This yields the final reported parameters and standard deviations.
If the option is disabled, the Stage-1 cell parameters are reported directly.

---

<a id="space-group-analysis"></a>
## Space Group Analysis

After a high-quality unit cell is obtained, Brutus can suggest likely space groups based
on systematic absences. This serves as input for subsequent structure solution or Rietveld refinement.

### Method

1.  **Generate unique reflections.**
    Using the refined cell, Brutus computes a complete list of theoretical reflections and keeps
    only crystallographically unique ones (e.g. it includes (100) but omits Friedel equivalent (−100)).
2.  **Index observed peaks.**
    All peaks in the curated list are indexed against the theoretical pattern.
3.  **Build a high-confidence subset.**
    To avoid ambiguity due to overlapping lines, Brutus retains only reflections for which no other
    theoretical $(hkl)$ lies within the `2θ Error` window of the observed peak.
4.  **Determine centering and extinctions.**
    This subset of unambiguous $(hkl)$ indices is compared with the extinction rules of
    candidate lattices (centerings and glide/screw symmetries). A rule is considered violated only if
    an *observed unambiguous* peak contradicts it.
5.  **Rank space groups.**
    Based on the crystal system and plausible centerings, the internal database is filtered.
    Each space group is assigned a violation count and ranked accordingly. The correct name of the space group is given, based on the orientation found by the program.

### Interpreting the Output

* **0 violations:**
    Ideal case. No unambiguous reflection breaks the extinction rules; these are the strongest candidates.
* **1–2 violations:**
    Still plausible; minor violations can result from weak forbidden lines or experimental artifacts.
* **Ambiguous peaks in italics:**
    In the PDF report, peaks classified as ambiguous (and therefore excluded from extinction analysis)
    are printed in italics to distinguish them from the high-confidence subset.

---

<a id="advanced-topics"></a>
## Advanced Topics: Enhanced Search and Sieving

Beyond the core indexing routine, Brutus applies several “fishing” strategies and reduction steps
to improve robustness and simplify the set of final solutions.

### 1. Swap Fishing for Ambiguity

For each promising solution, Brutus re-examines the indexing of the first four low-angle peaks.
It identifies the two peaks closest in angle (a common source of mis-assignment) and creates a new
hypothesis by swapping their HKL labels. The resulting cell is fully refined and rescored.
If this swap resolves an ambiguity, the new solution might have a higher M(20) and this solution will be retained.

### 2. Matrix-Based Cell Transformations

Candidate cells are transformed using crystallographic matrices to test for related primitive
or higher-symmetry descriptions. For example, a body-centered (I) cell can be mapped to an
equivalent primitive (P) cell and evaluated.

### 3. HKL Divisor Analysis

The list of indexed HKLs is inspected for common divisors. If all indices along one axis
(e.g. all *h* values) share a common factor, Brutus tests a cell with the corresponding
axis shortened accordingly (sub-cell).

### 4. Orthorhombic–Hexagonal Relationship

A hexagonal lattice can sometimes be indexed as C-centered orthorhombic with
$b/a \approx \sqrt{3}$. All orthorhombic solutions are checked against this condition,
and potential hexagonal equivalents are generated and evaluated.

### 5. Niggli Cell Standardization

For each high-ranking solution, Brutus computes the **Niggli reduced cell**,
i.e. the standardized primitive cell describing the same lattice.

The conventional (possibly centered) cell identified by the program is first converted to a primitive cell
using the detected centering (e.g. I, F, C). A reduction algorithm is then applied to obtain the
most compact set of basis vectors ($a, b, c$) and angles ($\alpha, \beta, \gamma$) satisfying
the Niggli conditions.

The Niggli cell is useful for:

* **Database searching:** Different conventional cells that represent the same lattice
    reduce to the same Niggli cell, which can be used as a canonical key.
* **Symmetry analysis:** Niggli parameters correlate directly with Bravais lattices
    and crystal systems.
* **Standardized reporting:** It removes ambiguity when comparing unit cells between sources.

Niggli cells are included in the detailed section of the PDF report for each major candidate.

### 6. Final Sieving

After all transformations and refinements, Brutus applies a final de-duplication step.
If two solutions have volumes within 1% of each other, the one with higher symmetry is preferred.

If their symmetry is identical (e.g. two monoclinic cells), M(20) is compared.
Cells with M(20) values within a small tolerance (e.g. ΔM(20) < 0.05) are considered equivalent in
quality and the first solution found is retained.

---

<a id="troubleshooting"></a>
## Troubleshooting & FAQ

### Why were no solutions found?

* **Poor peak list:**
    This is the most common cause. Ensure the list is accurate, impurity peaks are removed,
    and positions are refined. A minimum of 15–20 clean peaks is recommended.
    For low symmetry, the first 10 peaks (largest d-spacings) must be especially reliable.
* **Incorrect parameters:**
    Check the wavelength, the selected `Radiation Preset`, and the `Max Volume` value.
* **GPU parameters are too restrictive:**
    The true solution might require HKLs or peaks beyond the default `HKL Basis Size` or `Peaks to Combine`. Try increasing these values.
* **Large zero-point error:**
    Brutus can correct moderate misalignments, but very large zero errors may still prevent indexing.
* **Sample is a mixture:**
    Successful indexing requires peaks from a single crystalline phase.

### Why is M(20) low although the fit looks good?

* **Sub-multiple or super-multiple cell:**
    The found cell may be a multiple or sub-cell of the true one. It can index only a subset of peaks
    and is penalized for being too sparse or too dense in calculated reflections.
* **High error / low resolution:**
    The `2θ Error` setting may be too strict for broad peaks. Try slightly increasing it.
* **Spurious solution:**
    Random solutions can fit a few peaks by chance. Always check whether major observed peaks are
    reproduced by the calculated pattern. If not, the solution is incorrect regardless of M(20).

### Test Files

The GitHub repository contains several example data sets:
* **Monoclinic\_test\_1.xy** –
    Synchrotron XRD, monoclinic, monochromatic radiation λ = 0.79764 Å.
    Lattice parameters: a = 19.877 Å, b = 8.196 Å, c = 11.243 Å, β = 106.08°.
* **C61Br2\_079764.XY** –
    Dibromo-methano fullerene measured at ESRF, λ = 0.79764 Å.
    Contains an impurity peak at about 16.24° 2θ.
    A likely solution is cubic I with a ≈ 18.92 Å.
* **SPDDRR1\_sample2\_0692.xy** –
    Round-robin data measured at Daresbury, λ = 0.692 Å.
    Probable cell: orthorhombic with a = 10.983 Å, b = 12.852 Å, c = 15.740 Å.
* **SPDDRR1\_sample2\_Cu.xy** –
    Same sample as above, measured with a Cu Kα laboratory source.
* **SPDDRR1\_zhu1\_Cu.xy** –
    Round-robin sample 1, measured with Cu Kα; likely monoclinic with
    a = 7.672 Å, b = 9.624 Å, c = 7.076 Å, β = 106.24°.
* **PbSO4.xra** and **FAP.xra** –
    Laboratory Cu Kα samples representative of orthorhombic and hexagonal structures (these datafiles were taken from GSAS-2 tutorials).
* **P-1\_sim\_5\_6\_7\_86\_91\_96.txt** –
    Simulated powder pattern for a P−1 lattice, λ = 1.7 Å,
    a = 5.0 Å, b = 6.0 Å, c = 7.0 Å, α = 86.0°, β = 91.0°, γ = 96.0°.

---

<a id="references"></a>
## References

Brutus was developed by Nita Dragoe at Université Paris-Saclay (2024–2025) as a successor to the earlier
program *Powder* (1999–2000). If you use Brutus in your work, please cite:
[https://doi.org/10.13140/RG.2.2.13443.57126](https://doi.org/10.13140/RG.2.2.13443.57126)

For further background on the methodology, the following references are recommended:

1.  **M(20) figure of merit**<br>
    de Wolff, P. M. (1968).
    “A Simplified Criterion for the Reliability of a Powder Pattern Indexing.”
    *Journal of Applied Crystallography* **1**, 108–113.
2.  **F(N) figure of merit**<br>
    Smith, G. S. & Snyder, R. L. (1979).
    “F(N): A Criterion for Rating Powder Diffraction Patterns and Evaluating the Reliability of Powder-Pattern Indexing.”
    *Journal of Applied Crystallography* **12**, 60–65.
3.  **General powder diffraction text**<br>
    Klug, H. P. & Alexander, L. E. (1974).
    *X-Ray Diffraction Procedures for Polycrystalline and Amorphous Materials*, 2nd ed.
    New York: Wiley-Interscience.
4.  **Deconvolution Algorithm (van Cittert):**<br>
    van Cittert, P. H. (1931).
    “Zum Einfluß der Spaltbreite auf die Intensitätsverteilung in Spektrallinien II.”
    *Zeitschrift für Physik*, **69**, 298–308.
5.  **Alternative indexing approaches**<br>
    Ito, T. (1949). “A General Powder X-ray Photography.” *Nature* **164**, 755–756.<br>
    Werner, P.-E., Eriksson, L. & Westdahl, M. (1985).
    “TREOR, a Semi-exhaustive Trial-and-Error Powder Indexing Program for All Symmetries.”
    *Journal of Applied Crystallography* **18**, 367–370.<br>
    Visser, J. W. (1969).
    “A Fully Automatic Program for Finding the Unit Cell from Powder Data.”
    *Journal of Applied Crystallography* **2**, 89–95.<br>
    Le Bail, A. (2004).
    “Monte Carlo Indexing with McMaille.” *Powder Diffraction* **19(3)**, 249–254.<br>
    Boultif, A. & Louër, D. (2004).
    “Powder Pattern Indexing with the Dichotomy Method.”
    *Journal of Applied Crystallography* **37**, 724–731.
6.  **Previous software**<br>
    Dragoe, N. (2001).
    “PowderV2: A Suite of Applications for Powder X-Ray Diffraction Calculations.”
    *Journal of Applied Crystallography* **34**, 535.

---
*Help guide generated with the assistance of an AI, last updated 16 November 2025.*