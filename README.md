# Technical Overview & Methodology

<a id="introduction"></a>
This document is a technical reference for the Brutus powder indexing software. It explains the main algorithms, search parameters, and methodology, and is intended for users familiar with powder X-ray diffraction.

### Core Goal

The aim of *ab initio* powder indexing is to determine the unit-cell parameters ($a, b, c, \alpha, \beta, \gamma$) from a list of observed diffraction peak positions ($2\theta$). Brutus performs this task using a system-specific, exhaustive search algorithm.

The central assumption is that a small subset of the most intense, low-angle reflections corresponds to simple crystal planes with low-integer Miller indices $(hkl)$. For a given crystal system, the program chooses exactly as many observed peaks as there are unknown lattice parameters, and solves the resulting system of linear equations.

### Q-Space Formulation

All peak positions are first converted from $2\theta$ to Q-space, where $Q = 1/d^2$. The general quadratic relationship between $Q$, the Miller indices, and the reciprocal cell parameters ($A, B, C, D, E, F$) is

$$Q_{hkl} = Ah^2 + Bk^2 + Cl^2 + Dkl + Ehl + Fhk$$

Brutus solves for these reciprocal parameters (or the subset relevant to the current crystal system), then converts them to real-space cell parameters. Each candidate cell is refined immediately and scored against the full peak list.

---

<a id="quick-start"></a>
## Quick Start Guide

Use the following workflow for a typical single-phase powder pattern.

1.  **Load the data file.**
    Click `Select Data File`. Supported formats include `.xy`, `.xrdml`, `.ras`, and others.
2.  **Detect peaks.**
    On the **Peaks** tab, adjust the `Min peak (%)`, `Radius (pts)`, and `Points` sliders until the automatically detected peaks match the visual pattern.
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
    * Select the crystal systems to search. Enabling Monoclinic or Triclinic activates GPU-accelerated searches.
5.  **Start indexing.**
    Click `Start Indexing`. Progress is shown on the main bar.
6.  **Inspect solutions.**
    On the **Solutions** tab:
    * Sort solutions by M(20) and F(20).
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
    Displays the wavelength used for calculations. It is updated automatically when a preset is chosen and depends on the `Strip K-alpha2` setting. It becomes editable only when a `Custom` preset is selected.
* **Strip K-alpha2:**
    Applies a vanCittert-type correction to remove the Kα<sub>2</sub> component before peak analysis. When enabled, the `Ka1 Wavelength` field is set to the pure Kα<sub>1</sub> value. The default is OFF.
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

#### 3. Solutions Tab

* **Solutions table:**
    Shows each valid solution with crystal system, unit-cell parameters, volume, M(20) and F(20). Selecting a row updates the main chart. After the search finishes, you can filter visible solutions by enabling or disabling crystal systems.

### Results Area

* **Chart:**
    Shows the experimental diffraction pattern, observed peaks (red ticks) and calculated peaks for the selected solution (blue ticks). A good solution exhibits visually convincing overlap.
* **Chart interaction:**
    * Zoom: mouse wheel.
    * Pan: click and drag.
    * Reset zoom: right-click.
    * Add peak: `Ctrl + Click`.

---

<a id="peak-finding"></a>
## Peak Finding in Detail

Accurate peak positions are the single most important input for successful indexing. Brutus uses a multi-step procedure to detect peaks from raw intensity data.

### Algorithm Steps

1.  **Kα<sub>2</sub> stripping (optional):**
    If `Strip K-alpha2` is enabled, the Rachinger algorithm is applied to the raw intensities.
2.  **Background subtraction:**
    A rolling-ball style algorithm estimates and removes the background. The `Radius` slider controls the ball radius.
3.  **Data smoothing:**
    A Savitzky–Golay filter is applied to the background-subtracted signal to reduce noise while preserving peak shapes. The `Points` slider sets the window width.
4.  **Initial peak detection:**
    Local maxima above the `Min peak (%)` threshold are identified.
5.  **Position refinement:**
    For each peak, a five-point least-squares quadratic fit (based on Savitzky–Golay coefficients) is carried out around the maximum to obtain a sub-channel position. If too close to the data edge, the algorithm falls back to a three-point fit.

### Practical Recommendations

* Start from the default slider values and inspect the result visually.
* If weak but real peaks are missed, reduce `Min peak (%)`. If noise is detected as peaks, increase it.
* For broad, slowly varying backgrounds, increase `Radius`.
* For noisy data, increase `Points` (smoothing), but avoid over-smoothing, which can merge or shift peaks.
* Always manually curate the final peak list. Remove artifacts and known impurity peaks. If you are not stripping Kα<sub>2</sub>, delete Kα<sub>2</sub> shoulders explicitly.
* Kα<sub>2</sub> stripping can simplify the pattern but may introduce small artifacts. If indexing fails with stripping ON, try turning it OFF and manually cleaning the peak list.

---

<a id="indexing-method"></a>
## Indexing Algorithm and Search Parameters

Brutus uses a dedicated search routine for each crystal system. All are exhaustive trial methods that iterate over combinations of low-angle peaks and low-index Miller indices. The number of peaks used for the direct solve is equal to the number of unknown lattice parameters.

### Linear System Formulation

The search is formulated as a system of linear equations:

$$Q_{obs} = \sum P_i \cdot H_i$$

where $Q_{obs}$ are the observed $1/d^2$ values, $H_i$ are terms derived from trial Miller indices (e.g. $h^2$, $k^2$, $l^2$), and $P_i$ are the reciprocal lattice parameters (e.g. $A = 1/a^2$, $B = 1/b^2$, …).

### System-by-System Logic

* **Cubic (1 parameter, $A = 1/a^2$) – CPU worker**<br>
    Solves a 1×1 system:
    $$Q_{obs, 1} = (h_1^2 + k_1^2 + l_1^2) \cdot A$$
    The program iterates through the first 10 observed peaks and assigns trial $(hkl)$ vectors (integers up to 8) to obtain candidate values for $a$.

* **Tetragonal & Hexagonal (2 parameters) – CPU worker**<br>
    Solves a 2×2 system using pairs of peaks from the first 10:
    $$Q_{obs, 1} = H_{1,a} \cdot P_1 + H_{1,c} \cdot P_2$$
    $$Q_{obs, 2} = H_{2,a} \cdot P_1 + H_{2,c} \cdot P_2$$
    Pairs of trial $(hkl)$ vectors are assigned to determine $P_1$ and $P_2$, which are then converted to $a$ and $c$.

* **Orthorhombic (3 parameters, $A, B, C$) – CPU worker**<br>
    Solves a 3×3 system for $1/a^2$, $1/b^2$, and $1/c^2$.
    The algorithm:
    * Loops over all combinations of 3 peaks from the first 10 (120 triplets).
    * For each peak triplet, tests all combinations of 3 HKLs drawn from a basis of 80 simple reflections (82,160 triplets).
    * Total search space: 120 × 82,160 ≈ 9.86 million combinations, forming a truly exhaustive search over this HKL basis.

* **Monoclinic (4 parameters, $A, B, C, D$) – WebGPU accelerated**
    #### Direct 4-Peak Solve
    <p>
        The monoclinic $Q$-spacing formula can be written as
    </p>
    <p>
        $$Q_{hkl} = A h^2 + B k^2 + C l^2 + D h l$$
    </p>
    <p>
        One combination of 4 observed peaks and 4 trial HKLs defines a 4×4 matrix $M$ and right-hand side vector $\vec{q}$.
        The system
        $$M \cdot \vec{x} = \vec{q}$$
        is solved for $\vec{x} = [A, B, C, D]$.
    </p>

    #### GPU Parallelism and Permutations
    <p>
        The correct assignment between peaks and HKLs is unknown, so each GPU thread:
    </p>
    <ul>
        <li>Receives one unique (4-peak combination, 4-HKL combination) pair.</li>
        <li>Loops 24 times (4!) over all permutations of the 4 peaks.</li>
        <li>Solves the 4×4 system for each permutation using a dedicated solver.</li>
    </ul>
    <p>
        The program scans all combinations of 4 peaks from the first 7 (35 combinations)
        and all combinations of 4 HKLs from a basis of 80 (≈1.58 million combinations).
        The total number of trial cells is therefore:
    </p>
    <p><strong>≈ 1.33 billion trial cells.</strong></p>

    #### Two-Stage GPU Filtering
    <p>
        Most of these candidate cells are discarded on the GPU before they reach the CPU:
    </p>
    <ol>
        <li>
            <strong>Stage 1 – Basic filter (<code>extractCell</code>):</strong>
            A candidate is immediately rejected if:
            <ul>
                <li>The 4×4 system is singular.</li>
                <li>The derived monoclinic angle β is invalid ($D^2 \ge 4AC$) or outside 90°–150°.</li>
                <li>$a$, $b$, or $c$ is outside the range 2–50 Å.</li>
                <li>The cell volume exceeds <code>Max Volume</code>.</li>
            </ul>
        </li>
        <li>
            <strong>Stage 2 – Mini Figure of Merit (<code>validate_fom_avg_diff</code>):</strong>
            For surviving cells, the GPU:
            <ul>
                <li>Computes theoretical $q$ values for the first 100 HKLs.</li>
                <li>Compares them with the first 20 observed peaks.</li>
                <li>Evaluates the mean squared normalized error
                    $\text{avg}((\frac{q_{obs}-q_{calc}}{\text{tolerance}})^2)$.</li>
                <li>Accepts only cells whose score is below an internal threshold (e.g. 2.25).</li>
            </ul>
        </li>
    </ol>

* **Triclinic (6 parameters, $A \dots F$) – WebGPU accelerated**
    #### Direct 6-Peak Solve
    <p>
        The triclinic case solves the full quadratic form:
    </p>
    <p>
        $$Q_{hkl} = Ah^2 + Bk^2 + Cl^2 + Dkl + Ehl + Fhk$$
    </p>
    <p>
        A combination of 6 peaks and 6 trial HKLs defines a 6×6 matrix $M$ and
        $\vec{q}$, and Brutus solves
        $$M \cdot \vec{x} = \vec{q}$$
        for $\vec{x} = [A, B, C, D, E, F]$.
    </p>

    #### GPU Parallelism and Permutations
    <p>
        Each GPU thread receives one unique (6-peak combination, 6-HKL combination) pair and:
    </p>
    <ul>
        <li>Loops 720 times (6!) over all permutations of the 6 peaks.</li>
        <li>Solves the 6×6 system for each permutation.</li>
    </ul>
    <p>
        The search uses all combinations of 6 peaks from the first 8 (28 combinations)
        and all combinations of 6 HKLs from a basis of 40
        (≈3.84 million combinations). The total number of trial cells is
    </p>
    <p><strong>≈ 77.4 billion trial cells.</strong></p>

    #### Two-Stage GPU Filtering
    <ol>
        <li>
            <strong>Stage 1 – Basic filter (<code>extractCell</code>):</strong>
            A candidate is rejected if:
            <ul>
                <li>The 6×6 matrix is singular.</li>
                <li>The reciprocal metric tensor is not positive-definite or cannot be inverted.</li>
                <li>$a$, $b$, or $c$ is outside 2–50 Å.</li>
                <li>$\alpha$, $\beta$, or $\gamma$ is outside 60°–150°.</li>
                <li>The cell volume exceeds <code>Max Volume</code>.</li>
            </ul>
        </li>
        <li>
            <strong>Stage 2 – Mini Figure of Merit:</strong>
            Identical logic to the monoclinic case: a mean squared normalized error is evaluated
            against the first 20 peaks, and only sufficiently good candidates are retained.
        </li>
    </ol>

<h3 id="tuning-the-search">Tuning the Search & Troubleshooting</h3>
<p>
    The GPU searches are exhaustive over the specified bases, but their usefulness depends strongly on the input parameters.
    If you obtain no solutions, or too many, consider the following adjustments.
</p>

<h4>If no solutions (or only poor ones) are found</h4>
<ul>
    <li>
        <strong>Re-examine the peak list.</strong>
        This is the dominant failure mode. Check that the first 10–15 peaks belong to a single phase,
        are free of impurities, and have accurate $2\theta$ positions. For low symmetry,
        the first 10 peaks with largest interplanar distances are particularly critical.
    </li>
    <li>
        <strong>Relax <code>2θ Error</code>.</strong>
        If this value is too strict for your data resolution, valid solutions may be discarded.
    </li>
    <li>
        <strong>Increase <code>Max Volume</code>.</strong>
        The true cell may be larger than initially expected.
    </li>
    <li>
        <strong>Increase the HKL basis (code-level option).</strong>
        If the default set of HKLs does not include the right 4 or 6 reflections,
        the direct solve may never hit the correct combination. In <code>brutus.html</code> (inside
        <code>startIndexing</code>), increase the slice, for example from
        <code>hkl_basis_raw.slice(0, 80)</code> to <code>slice(0, 120)</code>. The search will become
        much longer.
    </li>
</ul>

<h4>If you get too many solutions (GPU buffer fills)</h4>
<ul>
    <li>
        <strong>Tighten <code>2θ Error</code>.</strong>
        The GPU buffer is now set at 100k solutions, if you have more than 100k candidates you need to adjust some parameters. A loose tolerance allows many marginal cells to pass so this is the first thing to check. 
    </li>
    <li>
        <strong>Reduce <code>Max Volume</code>.</strong>
        This is an effective way to remove unphysical large cells.
    </li>
    <li>
        <strong>Increase the number of peaks used (code-level option).</strong>
        Using more peaks in the direct solve (e.g. increasing <code>max_p</code> inside the
        <code>triTask</code> or <code>monoTask</code>) greatly enlarges the computation but
        can dramatically suppress false positives. You can also decrease the FOM tolerance in the wgsl code.
    </li>
</ul>

<h4>GPU Buffers and Chunking</h4>
<p>
    The total list of HKL combinations for triclinic indexing (≈1.19 billion combinations)
    is far too large for a single GPU buffer (tens of GB of VRAM). Brutus therefore:
</p>
<ol>
    <li>
        Generates HKL combinations in large JavaScript chunks (e.g. ≈50 million combinations at a time)
        in <code>brutus.html</code>.
    </li>
    <li>
        Further splits each of these into small “dispatch chunks” (e.g. 256 workgroups) in
        <code>webgpu-engine.js</code> before sending them to the GPU.
    </li>
</ol>
<p>
    This two-level chunking avoids both memory overflow and long-running GPU commands that may trigger
    “device hung” errors. The progress bar follows the JavaScript-side generation chunks.
</p>

> **Performance: WebGPU vs CPU**
> <p>
>     Cubic, Tetragonal, Hexagonal, and Orthorhombic searches run on the CPU (in a Web Worker) and are typically
>     very fast (seconds to a few minutes).
> </p>
> <p>
>     Monoclinic and Triclinic searches are offloaded to the GPU through WebGPU, supported by all major
>     modern browsers. This allows testing of billions of trial cells in a time frame that would be
>     impractical on the CPU.
> </p>
> <p>
>     On a recent GPU, the throughput can exceed 10<sup>8</sup> trials per second, depending on hardware.
>     For maximum speed, keep the Brutus tab visible and active; background tabs may be throttled
>     by the browser.
> </p>

---

<a id="evaluating-solutions"></a>
## Evaluating Solutions

The indexing search usually produces several candidate cells. Brutus keeps at most the best 50 during the search. Selecting the correct one requires interpreting figures of merit and checking the refined fit.

### de Wolff Figure of Merit: M(20)

The primary ranking indicator is the **de Wolff Figure of Merit, M(20)**, calculated from the first 20 observed reflections. It combines both positional accuracy and completeness.

| M(20) value | Interpretation                                                    |
| :---------- | :---------------------------------------------------------------- |
| > 20        | Very likely correct.                                              |
| > 10        | Likely correct, provided the cell volume is chemically plausible. |
| 5–10        | Plausible; requires further inspection.                           |
| < 5         | Probably spurious; treat with caution.                            |

### F(N) Figure of Merit

As a complementary metric, Brutus computes the **F(N)** figure of merit, usually with N = 20 (F(20)). While M(20) emphasizes the completeness of indexing, F(N) focuses on the average positional accuracy.

$$F_N = \frac{N}{\langle |\Delta(2\theta)| \rangle \cdot N_{calc}}$$

* $N$ – number of observed lines used (e.g. 20).
* $\langle |\Delta(2\theta)| \rangle$ – mean absolute difference between observed and calculated $2\theta$ for those lines.
* $N_{calc}$ – number of theoretical reflections (observed or not) up to the $2\theta$ of the $N$-th observed line.

A high F(20) indicates a precise fit with low average error. A solution with both high M(20) and high F(20) is usually very reliable.

### Least-Squares and Zero-Point Refinement

For each promising candidate, Brutus performs a two-stage refinement.

#### Stage 1: Internal zero-point correction

First, a constrained refinement is carried out including a zero-point error parameter. This internal zero correction is limited by the user-defined `2θ Error`, which allows the algorithm to compensate for modest alignment errors without overfitting. The corrected peak positions are then used to refine a stable baseline cell.

#### Stage 2: Final refinement (optional)

If the Stage-1 cell achieves a sufficiently high M(20) and the `Refine Zero-Point Error` option is enabled, a second, full refinement (with an unconstrained zero-point) is performed. This yields the final reported parameters and standard deviations. If the option is disabled, the Stage-1 cell parameters are reported directly.

---

<a id="space-group-analysis"></a>
## Space Group Analysis

After a high-quality unit cell is obtained, Brutus can suggest likely space groups based on systematic absences. This serves as input for subsequent structure solution or Rietveld refinement.

### Method

1.  **Generate unique reflections.**
    Using the refined cell, Brutus computes a complete list of theoretical reflections and keeps only crystallographically unique ones (e.g. it includes (100) but omits equivalent (−100)).
2.  **Index observed peaks.**
    All peaks in the curated list are indexed against the theoretical pattern.
3.  **Build a high-confidence subset.**
    To avoid ambiguity due to overlapping lines, Brutus retains only reflections for which no other theoretical $(hkl)$ lies within the `2θ Error` window of the observed peak.
4.  **Determine centering and extinctions.**
    This subset of unambiguous $(hkl)$ indices is compared with the extinction rules of candidate lattices (centerings and glide/screw symmetries). A rule is considered violated only if an *observed unambiguous* peak contradicts it.
5.  **Rank space groups.**
    Based on the crystal system and plausible centerings, the internal database is filtered. Each space group is assigned a violation count and ranked accordingly.

### Interpreting the Output

* **0 violations:**
    Ideal case. No unambiguous reflection breaks the extinction rules; these are the strongest candidates.
* **1–2 violations:**
    Still plausible; minor violations can result from weak forbidden lines or experimental artifacts.
* **Ambiguous peaks in italics:**
    In the PDF report, peaks classified as ambiguous (and therefore excluded from extinction analysis) are printed in italics to distinguish them from the high-confidence subset.

---

<a id="advanced-topics"></a>
## Advanced Topics: Enhanced Search and Sieving

Beyond the core indexing routine, Brutus applies several “fishing” strategies and reduction steps to improve robustness and simplify the set of final solutions.

### 1. Swap Fishing for Ambiguity

For each promising solution, Brutus re-examines the indexing of the first four low-angle peaks. It identifies the two peaks closest in angle (a common source of mis-assignment) and creates a new hypothesis by swapping their HKL labels. The resulting cell is fully refined and rescored. If this swap resolves an ambiguity, the new solution will usually have a much higher M(20).

### 2. Matrix-Based Cell Transformations

Candidate cells are transformed using crystallographic matrices to test for related primitive or higher-symmetry descriptions. For example, a body-centered (I) cell can be mapped to an equivalent primitive (P) cell and evaluated.

### 3. HKL Divisor Analysis

The list of indexed HKLs is inspected for common divisors. If all indices along one axis (e.g. all *h* values) share a common factor, Brutus tests a cell with the corresponding axis shortened accordingly (sub-cell).

### 4. Orthorhombic–Hexagonal Relationship

A hexagonal lattice can sometimes be indexed as C-centered orthorhombic with $b/a \approx \sqrt{3}$. All orthorhombic solutions are checked against this condition, and potential hexagonal equivalents are generated and evaluated.

### 5. Niggli Cell Standardization

For each high-ranking solution, Brutus computes the **Niggli reduced cell**, i.e. the standardized primitive cell describing the same lattice.

The conventional (possibly centered) cell identified by the program is first converted to a primitive cell using the detected centering (e.g. I, F, C). A reduction algorithm is then applied to obtain the most compact set of basis vectors ($a, b, c$) and angles ($\alpha, \beta, \gamma$) satisfying the Niggli conditions.

The Niggli cell is useful for:

* **Database searching:** Different conventional cells that represent the same lattice reduce to the same Niggli cell, which can be used as a canonical key.
* **Symmetry analysis:** Niggli parameters correlate directly with Bravais lattices and crystal systems.
* **Standardized reporting:** It removes ambiguity when comparing unit cells between sources.

Niggli cells are included in the detailed section of the PDF report for each major candidate.

### 6. Final Sieving

After all transformations and refinements, Brutus applies a final de-duplication step. If two solutions have volumes within 1% of each other, the one with higher symmetry is preferred.

If their symmetry is identical (e.g. two monoclinic cells), M(20) is compared. Cells with M(20) values within a small tolerance (e.g. ΔM(20) < 0.05) are considered equivalent in quality and additional rules are applied:

* **Monoclinic:**
    If volumes differ by less than 2% and M(20) is effectively equal, Brutus prefers the cell whose β angle is closer to 90° (e.g. 106° is preferred over 147°).
* **Other systems:**
    If a tie persists, the first solution found is retained.

---

<a id="troubleshooting"></a>
## Troubleshooting & FAQ

### Why were no solutions found?

* **Poor peak list:**
    This is the most common cause. Ensure the list is accurate, impurity peaks are removed, and positions are refined. A minimum of 15–20 clean peaks is recommended. For low symmetry, the first 10 peaks (largest d-spacings) must be especially reliable.
* **Incorrect parameters:**
    Check the wavelength, the selected `Radiation Preset`, and the `Max Volume` value.
* **Large zero-point error:**
    Brutus can correct moderate misalignments, but very large zero errors may still prevent indexing.
* **Sample is a mixture:**
    Successful indexing requires peaks from a single crystalline phase.

### Why is M(20) low although the fit looks good?

* **Sub-multiple or super-multiple cell:**
    The found cell may be a multiple or sub-cell of the true one. It can index only a subset of peaks and is penalized for being too sparse or too dense in calculated reflections.
* **High error / low resolution:**
    The `2θ Error` setting may be too strict for broad peaks. Try slightly increasing it.
* **Spurious solution:**
    Random solutions can fit a few peaks by chance. Always check whether major observed peaks are reproduced by the calculated pattern. If not, the solution is incorrect regardless of M(20).

### Test Files

* The GitHub repository contains several example data sets:
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
    Laboratory Cu Kα samples representative of orthorhombic and hexagonal structures.
* **P-1\_sim\_5\_6\_7\_86\_91\_96.txt** –
    Simulated powder pattern for a P−1 lattice, λ = 1.7 Å,
    a = 5.0 Å, b = 6.0 Å, c = 7.0 Å, α = 86.0°, β = 91.0°, γ = 96.0°.
    On a Windows PC with Chrome and an NVIDIA T1000 8 GB GPU, Brutus typically finds a solution with
    M(20) ≈ 86.9 (or an equivalent alternative cell) in under three minutes.
    On the CPU-only version (see `combs.html`), the same search will take 10 to 20 hours.

---

<a id="references"></a>
## References

<p>
    Brutus was developed by Nita Dragoe at Université Paris-Saclay (2024–2025) as a successor to the earlier
    program <em>Powder</em> (1999–2000). If you use Brutus in your work, please cite:
    <br>
    <a href="https://doi.org/10.13140/RG.2.2.13443.57126">
        https://doi.org/10.13140/RG.2.2.13443.57126
    </a>
</p>
<p>
    For further background on the methodology, the following references are recommended:
</p>

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
*Help guide generated with the assistance of an AI, last updated 15 November 2025.*