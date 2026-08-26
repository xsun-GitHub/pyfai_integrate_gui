# pyFAI Integrate Viewer

A PyQt/silx GUI for viewing 2-D diffraction images and performing 1-D
azimuthal integration with pyFAI. This project contains three generations; a
later version includes the applicable features of the versions before it.

## Versions

| Version | Program | Main additions |
| --- | --- | --- |
| Original | `pyfai_integrate_gui.py` | Standard integration, subtraction, plots, and export |
| v2 | `pyfai_integrate_gui_v2.py` | ROI Sum, unified Status, Stop, and improved caching |
| v3 | `pyfai_integrate_gui_v3.py` | p62 NeXus modes and integrated PyAnomScat ASAXS analysis |

The original and v2 programs remain available independently.

## Requirements and installation

Python 3.10 or newer is recommended. Core dependencies are NumPy, PyQt6,
pyFAI, silx, Fabio, h5py, qtawesome, imageio, and imageio-ffmpeg. The v3 ASAXS
window additionally uses pyqtgraph, scanf, and xraydb.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the required generation:

```powershell
python pyfai_integrate_gui.py
python pyfai_integrate_gui_v2.py
python pyfai_integrate_gui_v3.py
```

## Original application

### Main functions

- Opens one or multiple EDF, CBF, TIFF, IMG, MarCCD, HDF5, or NumPy images and
  expands multi-frame files.
- Loads PONI geometry and displays detector, pixel, distance, and wavelength
  information.
- Applies detector dead areas/module gaps and an optional user mask.
- Integrates full-resolution data while using a lower-memory display preview.
- Supports integration units, point count, radial/azimuthal ranges, Log X, and
  Log Y.
- Loads optional Empty and Background images with independent display,
  subtraction, and factor settings.
- Displays Data, Empty, Background, and Subtracted curves.
- Calculates a masked detector intensity sum using an `int64` accumulator.
- Optionally displays a pyFAI azimuthal-versus-radial Cake plot.
- Saves current plots, tab-separated ASCII data, and grouped MP4 videos.
- Keeps separate persistent Input and Export paths under **File -> Path**.
- **Options -> Plot NeXus** browses numeric datasets, assigns X/Y, plots
  matching arrays, and optionally displays `dY/dX`.
- **Options -> Plot ASCII** selects X/Y columns from common numeric text files.

Raw Data, Empty, and Background detector intensities must use a supported
integer NumPy type. Invalid shapes and data types are reported to the user.

### Standard workflow

1. Select one or more diffraction files using **Image**.
2. Load the corresponding **PONI** geometry and optionally a **Mask**.
3. Configure the number of points, integration unit, and optional ranges.
4. Optionally select Empty/Background data and configure **Show**, **Subtract**,
   and **Factor**.
5. Start integration and navigate files/frames with **Previous/Next**.
6. Export the current plot, all ASCII data, grouped video, or Data and Plots.

A single-frame reference is reused for every data frame. In the original
application, a multi-frame reference is paired frame-by-frame and must have the
same frame count as the sample.

### ASCII output

One `.dat` file is saved for each image/frame. Comment lines begin with `#` and
record metadata, filenames, factors, and column names. Depending on the loaded
references, columns include the radial coordinate, `Intensity`, `Subtracted`,
`Empty`, and `Background`. Reference columns contain scaled intensities;
`Subtracted` removes every reference whose **Subtract** option is selected.

```text
# Empty file: empty.tif; factor: 1.00
# Background file: background.tif; factor: 0.50
# q_A^-1    Intensity    Subtracted    Empty    Background
1.0000000000e-03    2.5000000000e+04    2.4100000000e+04    7.0000000000e+02    2.0000000000e+02
```

## Version 2 additions

v2 preserves the original integration workflow and adds the features below.

### Detector Sum and ROI Sum

- Detector Sum runs only from its own **Calculate** button; **Integrate** never
  calculates it implicitly.
- The Source label displays a cached value for the current frame or
  `not calculated`.
- **Select ROI** in the Source toolbar draws a rectangular detector region.
  Drawing a new rectangle replaces the previous ROI.
- ROI Sum calculates masked integer intensity inside the ROI for every selected
  image/frame. Missing or invalid ROI selection produces a clear error.
- Detector and user-mask pixels contribute zero. ROI coordinates always refer
  to the full-resolution detector image.
- Detector Sum and ROI Sum remain independent and retain completed curves
  during recalculation, after Stop, and after ROI/PONI/Mask changes. Selecting
  a new image list clears both caches.

### Status, Stop, caching, and export

- **Start Integration** becomes **Integrate**.
- A unified **Status** area reports loading, integration, sum calculations, and
  exports. Integrate is disabled while required input is loading.
- **Stop** cancels at safe file/frame boundaries. An active pyFAI call is
  allowed to return safely and its result is discarded.
- Compatible Cake data, combined masks, reference integrations, and static
  detector masks are reused. Dynamic dummy-pixel masks are recalculated per
  frame.
- **Save -> Data and Plots** writes one `.dat` per image/frame. A one-frame
  filename group produces a PNG; a multi-frame group uses grouped MP4 export.
  The selected Integration, Detector Sum, or ROI Sum view is included, together
  with Cake when enabled.

## Version 3 additions

v3 preserves v2 and adds direct p62 NeXus loading, energy-aware reference
matching, multi-curve ASCII plotting, and integrated ASAXS analysis.

### p62 beamline modes

Choose **File -> Beamline -> p62**. Four buttons appear above **Input files**:

| Mode | Image dataset | Energy/wavelength behaviour |
| --- | --- | --- |
| SAXS | `/scan/data/saxs_raw` | PONI wavelength remains unchanged |
| WAXS | `/scan/data/waxs_raw` | PONI wavelength remains unchanged |
| ASAXS | `/scan/data/saxs_raw` | Energy is read and wavelength is updated per image |
| AWAXS | `/scan/data/waxs_raw` | Energy is read and wavelength is updated per image |

Energy comes from `/scan/data/energy` as `float64`, is validated as finite and
positive, and may be displayed as integer eV. ASAXS/AWAXS calculate wavelength
in metres using `lambda = hc/E` and assign it to the integrator for the
corresponding 2-D image. The PONI file on disk is never overwritten.

The image count may be an integer multiple of the energy count. Energies are
then assigned to consecutive image groups: 18 images and 6 energy values means
3 images per energy. Incompatible dimensions produce an input error.

### p62 Empty and Background subtraction

Empty and Background accept one or multiple `.nxs` files. SAXS/ASAXS use
`/scan/data/saxs_raw`; WAXS/AWAXS use `/scan/data/waxs_raw`.

- One reference image is integrated once and reused for all sample images.
- If sample and reference totals are equal, frames are paired in order and
  their energies must match.
- If totals differ, references are matched by energy. Repeated sample images at
  one energy reuse that energy's reference integration.
- Multi-energy reference data must contain compatible energy values.
- If q ranges differ, subtraction uses the common q interval and interpolates
  the reference onto the sample q points.

### Status, saving, and memory

The fixed Status section shows overall saving progress, file-series progress,
frame progress, and only the enabled Empty/Background subtraction sources. The
scrolling section shows filenames only. Cake data is saved only when Cake is
selected. Video export integrates and writes one frame at a time, then releases
the frame payload and Cake cache to reduce peak memory use.

### ASAXS ASCII output

ASAXS/AWAXS filenames include energy using `_E<energy>`:

```text
sample_frame_0001_E12000.dat
```

The ASCII header also records energy and wavelength. PyAnomScat can therefore
recover energy from the header when the filename does not use the preferred
form.

### Integrated PyAnomScat window

Choose **Options -> ASAXS...** to open the PyAnomScat Stuhrmann-method GUI as a
child window. **Import ASCII** starts in the main application's current Input
path. v3 loads only these local project resources and does not call the original
external ASAXS directory:

- `pyAnomScat_stuhrmann_method_v3.py`
- `pyAnomScat_stuhrmann_method.ui`

The `.ui` file defines the PyAnomScat window layout and is required. The old
bundled example `data/` directory and dark `StyleSheet/` are not needed by the
integrated workflow and have been removed. PyAnomScat uses the normal light
application palette. **Import ASCII** is available from the File menu; redundant
Quit controls were removed because the window's close button has the same role.
The legacy, unimplemented **Import HDF/Nexus** entry is also located in File and
remains disabled rather than incorrectly treating HDF data as ASCII. The first
Control Panel row contains Element, anomalous-factor, monochromator-shift, and
chemical-shift controls from left to right; the second row contains Stuhrmann
and Export. Table columns use practical defaults matching the compact data view
and remain manually resizable from the header. The Color table column is hidden;
**Plot -> Color** selects either the original default color list or Origin
Color4Line and immediately recolors loaded curves. New imports inherit the
current mono/chemical shifts, plots automatically scale both axes, and curve
export proposes the first imported filename by default.

The anomalous-factor operation checks the imported energies against the xraydb
range and displays a warning for missing, invalid, or unsupported values instead
of raising `ValueError: max() iterable argument is empty`. **Element** opens an
interactive periodic table; selecting a symbol updates both its symbol and
atomic number in the main ASAXS window. The Control Panel table includes a
clickable **Del** column and supports row drag-and-drop while keeping curve and
energy data aligned. Imported locations are split at the final path separator:
**Path** displays the directory and **File name** displays only the `.dat`
filename. All PyAnomScat plots use a light background; the input
curve plot and Stuhrmann-result plot include legends.

When importing a curve, PyAnomScat reads the comment immediately above the
first numeric row. A first column named `q_A^-1` changes the input and result X
axes to `A^-1`; `q_nm^-1` keeps `nm^-1`. Curve export adds the first two fields
of that source comment (for example `# q_A^-1 Intensity`) to its header.

### Multi-file Plot ASCII

**Options -> Plot ASCII** uses one mode for both single- and multi-file input.
Multiple curves can be loaded together, and every curve can be shown or hidden
independently in an Origin-like curve list. **Log X** and **Log Y** above the
plot are provided with the other plot-toolbar buttons and independently switch
either axis between linear and logarithmic scale. The right-side curve list
shows a color/line-style sample beside every filename; no separate legend is
drawn over or beside the plot.
