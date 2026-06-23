# Paper draft: New methodology subsections 3.3.5 & 3.3.6

> Working draft for two new subsections of the paper Methodology, capturing the
> recently added pipeline functionality (LSF, spatial binning, integrated dΣ/dv,
> three σ-extraction methods, phase-decomposed mass-weighted σ_v).
>
> We polish here in Markdown, then convert to LaTeX for `main.tex`.

---

## 3.3.5 Modeling Instrument Response

The spectral cube built in Section 3.3.4 is what an *ideal* spectrograph with
infinite resolution would record: each cell's emission lands in the channel
matching its Doppler-shifted frequency, broadened only by the cell's own thermal
motion. A real telescope however imposes two further effects on this signal:
its spectrograph has a finite spectral resolving power
$R = \lambda/\Delta\lambda$, and its detector has a finite spatial pixel size
covering many simulation cells. We model both with simple linear operations
applied to the cube of Section 3.3.4: convolution with a line spread function
(LSF) along the frequency axis, followed by spatial binning of cells into
instrument pixels.

### Spectral Resolution

A real spectrograph cannot reproduce a delta-function input perfectly:
diffraction at the entrance slit, optical aberrations, and the finite slit
width smear any input line into a finite-width pattern on the detector. We
characterise the instrument by its *resolving power* $R = \lambda/\Delta\lambda$,
defined as the ratio of the operating wavelength to the minimum wavelength
separation at which two adjacent lines remain distinguishable.

**From wavelength to velocity.**
For the line widths considered here $v \ll c$, and the non-relativistic
Doppler relation gives

$$
\lambda(v) = \lambda_{0}\!\left(1 + \frac{v}{c}\right)
\quad\Longrightarrow\quad
\frac{d\lambda}{\lambda_{0}} = \frac{dv}{c},
$$

so to first order $\Delta\lambda/\lambda_{0} = \Delta v/c$. Substituting
into the definition of $R$,

$$
R \;=\; \frac{\lambda_{0}}{\Delta\lambda} \;=\; \frac{c}{\Delta v}
\quad\Longrightarrow\quad
\boxed{\;\Delta v = \frac{c}{R}.\;}
$$

The wavelength resolution element $\Delta\lambda(\lambda) = \lambda/R$ thus
scales with $\lambda$, while the corresponding velocity element
$\Delta v = c/R$ is *constant*, independent of position on the spectrum.
(Equivalently $\Delta\nu_{\rm FWHM} = \nu/R$ also scales with the axis
variable; only $v$ is shift-invariant.) We therefore carry out all
subsequent broadening operations on a uniform velocity grid.

**The line spread function and the convolution form.**
A spectrograph is a *linear* instrument: doubling the input doubles the
output, and superpositions map to superpositions. Let $\mathcal{L}$ denote
the linear operator that takes a model spectrum $I(v)$ to its observed
counterpart $O(v)$. Using the sifting property of the Dirac delta to write
$I(v) = \int I(v')\,\delta(v - v')\,dv'$,

$$
O(v) \;=\; \mathcal{L}[I](v)
     \;=\; \int I(v')\,\mathcal{L}[\delta(v - v')]\,dv'
     \;\equiv\; \int I(v')\,G(v - v')\,dv',
$$

where $G(v - v') \equiv \mathcal{L}[\delta(v - v')]$ is, by definition, the
instrument's response to a delta-function input centred at $v'$ — the
*line spread function* (LSF). The action of the spectrograph is therefore
a convolution of the model with $G$, which we model as a unit-area Gaussian

$$
G(v;\sigma_{\rm LSF}) \;=\;
\frac{1}{\sigma_{\rm LSF}\sqrt{2\pi}}\,
\exp\!\left(-\frac{v^{2}}{2\sigma_{\rm LSF}^{2}}\right),
$$

the unit-area normalisation guaranteeing that the convolution preserves the
integrated flux in each channel.

**Connecting $\sigma_{\rm LSF}$ to $R$.**
What remains is to relate the kernel width $\sigma_{\rm LSF}$ to the
operationally measured $R$. We adopt the Gaussian form of the Rayleigh
criterion: two equal-amplitude Gaussian profiles are *just resolved* when
their separation equals the FWHM of one profile (the analogue, for a kernel
without diffraction zeros, of "principal maximum coinciding with the first
zero"). The minimum resolvable velocity separation derived above is
therefore identified with the FWHM of $G$:

$$
\Delta v \;=\; \mathrm{FWHM}(G) \;=\; \frac{c}{R}.
$$

The Gaussian FWHM follows from $G(x_{1/2}) = \tfrac{1}{2}G(0)$:

$$
\exp\!\left(-\frac{x_{1/2}^{2}}{2\sigma_{\rm LSF}^{2}}\right) = \frac{1}{2}
\;\Longrightarrow\;
x_{1/2} = \sigma_{\rm LSF}\sqrt{2\ln 2},
\qquad
\mathrm{FWHM} = 2\,x_{1/2} = 2\sqrt{2\ln 2}\,\sigma_{\rm LSF}.
$$

Combining,

$$
\sigma_{\rm LSF} \;=\; \frac{c}{R\,\cdot\,2\sqrt{2\ln 2}}
                \;\approx\; \frac{c}{2.355\,R}.
$$

The default resolving power $R = 10^{6}$ gives
$\sigma_{\rm LSF} \approx 0.13\,\mathrm{km\,s^{-1}}$, representative of
high-resolution heterodyne facilities such as ALMA. In practice the
convolution is applied channel-by-channel along the velocity axis using
`scipy.ndimage.gaussian_filter1d`.

Because thermal broadening (already encoded in the cube of Section 3.3.4) and
instrumental LSF broadening are statistically independent Gaussian processes,
their variances add. The total apparent linewidth measured along a single
sightline is therefore, to leading order,

$$
\sigma_{\rm obs}^{2} \approx
\sigma_{\rm thermal}^{2} + \sigma_{\rm LSF}^{2} + \sigma_{\rm bulk}^{2},
$$

where $\sigma_{\rm bulk}$ is the dispersion of cell-level bulk velocities within
the resolution element.

### Spatial pixel binning

A real instrument pixel covers an area larger than a single simulation cell.
Let $b$ denote the binning factor, so that one instrument pixel encompasses
$b \times b$ simulation cells. We construct the pixel cube by summing (rather
than averaging) the contributions of the constituent cells,

$$
\mathrm{cube}_{\rm pix}[\nu, I, J] =
\sum_{i = Ib}^{(I+1)b - 1}\,
\sum_{j = Jb}^{(J+1)b - 1}\,
\mathrm{cube}_{\rm cell}[\nu, i, j].
$$

Summation, rather than averaging, is the physically correct reduction because a
CCD pixel collects every photon that falls within its area; the spatially
integrated flux $\iint S\, dy\, dz$ is therefore automatically conserved.
Averaging would suppress each pixel by a factor $b^{2}$ and require post-hoc
rescaling. When $(n_y, n_z)$ is not divisible by $b$, the trailing rows and
columns are trimmed. The default binning factor is $b = 4$.

### Combined application

The LSF and spatial binning act on independent axes—frequency and spatial,
respectively—and therefore commute. We apply them in the order
LSF $\to$ spatial bin to mirror the physical sequence "light through the
spectrograph, then onto the detector". The resulting instrument-pixel cube has
shape $(n_{\nu},\, n_y/b,\, n_z/b)$ and represents what a finite-resolution
instrument would record.

Figure~\ref{fig:binned_pixel_grid} shows the resulting mock observations for
CO, C\textsuperscript{+}, and HCO\textsuperscript{+}. Each panel displays the
LSF-convolved, $b \times b$-binned spectrum of a single instrument pixel,
exhibiting visibly broadened, smoother profiles than the per-cell spectra of
Figures~\ref{fig:co_spectral_grid}–\ref{fig:hcoplus_spectral_grid}.

---

## 3.3.6 Velocity Dispersion Diagnostics

Velocity dispersion $\sigma_{v}$ is one of the most direct kinematic
diagnostics of the ISM, simultaneously encoding turbulent motions, thermal
broadening, ordered bulk flows, and feedback-driven outflows. Our pipeline
supports two complementary measurements: from the *synthesised line profile*
(the observer's perspective, including thermal and instrumental broadening) and
from the *underlying cell-level velocity field* (the simulation's intrinsic
bulk-flow dispersion, free of broadening contributions). Comparing the two
quantifies how strongly thermal and instrumental effects inflate the observed
linewidth.

### Spatially integrated line profile

We construct the spatially integrated surface-brightness spectrum from the
LSF-convolved cube by summing over the spatial dimensions and normalising by
the projected area,

$$
\frac{d\Sigma}{dv}(v) = \frac{1}{N_{y} N_{z}\, \Delta y\, \Delta z}\,
\sum_{I,J}\, \mathrm{cube}_{\rm obs}\bigl[\nu(v),\, I,\, J\bigr],
$$

yielding units of $\mathrm{erg\,s^{-1}\,Hz^{-1}\,cm^{-2}}$. The profile is
computed for two complementary lines of sight—(i) along the $x$-axis
(integrating over the $y$-$z$ plane) and (ii) along the $y$-axis (over the
$x$-$z$ plane)—allowing kinematic anisotropy to be assessed.

### Three methods to extract $\sigma_{v}$ from the line profile

From the integrated profile we extract $\sigma_{v}$ using three complementary
methods, each carrying different assumptions about the line shape.

**(i) Gaussian fit.** We fit

$$
S(v) = A\, \exp\!\left[-\frac{(v - v_{0})^{2}}{2\sigma^{2}}\right]
$$

by bounded non-linear least squares, returning the dispersion parameter
$\sigma$ together with the line centroid $v_{0}$. This method assumes a
Gaussian profile and is most informative for a single, isolated, symmetric
component; multi-peak or strongly asymmetric profiles cause the fit to fail or
yield biased values.

**(ii) Spectral second moment.** Without assuming any line shape we compute the
intensity-weighted moments,

$$
\langle v\rangle = \sum_{k} w_{k} v_{k},
\qquad \sigma^{2} = \sum_{k} w_{k}\, (v_{k} - \langle v\rangle)^{2},
\qquad w_{k} = \frac{S_{k}}{\sum_{j} S_{j}},
$$

where $\{v_{k}, S_{k}\}$ are the channel velocities and intensities of the
integrated profile. The moment method is robust to non-Gaussian shapes but
sensitive to flux in the line wings—a few low-amplitude tail channels can
substantially inflate $\sigma$.

**(iii) Mass-weighted dispersion from cell data.** Operating directly on the
simulation cells rather than on the synthesised spectrum, we compute the
density-weighted velocity moments,

$$
\langle v\rangle = \frac{\sum_{i} \rho_{i} v_{i}}{\sum_{i} \rho_{i}},
\qquad \sigma^{2} = \frac{\sum_{i} \rho_{i}\,
(v_{i} - \langle v\rangle)^{2}}{\sum_{i} \rho_{i}},
$$

where $\rho_{i}$ is the cell density and $v_{i}$ the line-of-sight component of
the cell's bulk velocity. Because $v_{i}$ is a bulk (rather than per-particle)
velocity, this dispersion contains *neither* thermal broadening *nor* the
instrumental LSF; it is the simulation's intrinsic kinematic dispersion and
provides a clean reference against which the spectral measurements can be
calibrated. To leading order,
$\sigma_{\rm spec}^{2} \approx \sigma_{\rm cell}^{2} + \sigma_{\rm thermal}^{2} + \sigma_{\rm LSF}^{2}$,
isolating each contribution.

### Phase-decomposed mass-weighted dispersion

A single $\sigma_{v}$ averaged over the entire simulation conflates the
kinematic signatures of multiple thermal phases. The ISM is intrinsically
multiphase—cold molecular and atomic gas ($\lesssim 10^{4}\,$K) coexists with
warm ionised media ($\sim 10^{4}$–$10^{6}\,$K) and hot supernova-driven
outflows ($\gtrsim 10^{6}\,$K)—and each phase exhibits markedly different
turbulence and bulk-flow properties. To isolate phase-specific kinematics we
partition cells by their corrected temperature $T$ (Section 3.3.2) into three
regimes:

| Phase | Temperature range | Physical interpretation |
|-------|-------------------|------------------------|
| Cool | $T < 2\times 10^{4}\,$K | cold neutral / molecular disk |
| Warm | $2\times 10^{4} \leq T \leq 10^{6}\,$K | warm neutral / ionised medium |
| Hot  | $T > 10^{6}\,$K | hot wind / coronal halo |

The lower threshold corresponds approximately to the H{\sc i}/H{\sc ii}
recombination boundary; the upper threshold marks the onset of the
bremsstrahlung-dominated cooling regime characteristic of X-ray-emitting
outflows. Within each phase we apply the mass-weighted moments,

$$
\langle v\rangle_{\rm ph} =
\frac{\sum_{i \in {\rm ph}} \rho_{i} v_{i}}{\sum_{i \in {\rm ph}} \rho_{i}},
\qquad \sigma_{\rm ph}^{2} =
\frac{\sum_{i \in {\rm ph}} \rho_{i}\,
(v_{i} - \langle v\rangle_{\rm ph})^{2}}{\sum_{i \in {\rm ph}} \rho_{i}},
$$

yielding $\langle v\rangle$, $\sigma$, mass fraction, and cell-count fraction
for each phase.

We note two conceptual points concerning bulk-motion subtraction. First, the
per-phase mean $\langle v\rangle_{\rm ph}$ is subtracted *automatically* by the
dispersion formula, so the resulting $\sigma_{\rm ph}$ is intrinsically
referred to each phase's own rest frame. Second, this is sufficient when the
simulation contains no organised large-scale flow such as global rotation. Our
QUOKKA stratified-box setup (Section 3.1) follows the TIGRESS configuration
with periodic horizontal boundaries and no bulk rotation, hence no rotational
contamination of the dispersion. For a globally rotating disk simulation,
$\sigma$ would instead be dominated by the rotation curve—e.g. a system with
$v_\phi \sim 220\,\mathrm{km\,s^{-1}}$ and intrinsic turbulence
$\sim 10\,\mathrm{km\,s^{-1}}$ would produce a global
$\sigma \sim 200\,\mathrm{km\,s^{-1}}$ unless the cells are first binned by
$(R,z)$ and the local $\langle v_{\phi}\rangle(R,z)$ subtracted before
evaluating $\sigma_{\rm ph}$.

The phase-decomposed dispersions are summarised in
Figure~\ref{fig:phase_sigmaV_bar}, with the corresponding mass-weighted
velocity distributions per phase and LOS in Figure~\ref{fig:phase_sigmaV_hist}.

---

## Companion figures (to be added)

| Figure ref | Source file | Description |
|------------|-------------|-------------|
| `\ref{fig:integrated_spectrum}` | `IntegratedSpectrumTask` → `IntegratedSpectrum_*.png` | dΣ/dv for each species, two panels (LOS=x, LOS=y), intrinsic vs LSF-convolved, with three σ annotations (Gaussian, moment, data) |
| `\ref{fig:binned_pixel_grid}` | `BinnedPixelGridTask` → `*_BinnedPixelGrid_b4.png` | Mock observations after LSF + b=4 binning. One figure per species, or three side-by-side. |
| `\ref{fig:phase_sigmaV_bar}` | `PhaseSigmaVTask` → `PhaseSigmaV_bar.png` | σ_ph (cool/warm/hot) × LOS (x, y) grouped bar chart |
| `\ref{fig:phase_sigmaV_hist}` | `PhaseSigmaVTask` → `PhaseSigmaV_hist.png` | 3 × 2 grid of mass-weighted velocity distributions per phase per LOS |

---

## Open questions / TODOs before LaTeX conversion

- [ ] Decide whether to cite specific instrument resolution references (e.g. ALMA technical handbook) for $R = 10^{6}$ context, or leave it as a generic example.
- [ ] Decide on phase-threshold citations (Draine 2011 already in bibliography; consider adding Wolfire+ 2003 or McKee & Ostriker 1977 for the three-phase ISM concept).
- [ ] Confirm whether to add a sentence in 3.3.6 §iv referring back to the warm/hot σ values from the simulation as physical sanity (pending decision on whether numbers go into Results section instead).
- [ ] Confirm figure placement order in Results section.
