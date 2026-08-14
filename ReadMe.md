# pyFAI Integrate Viewer

A PyQt/silx desktop GUI for viewing two-dimensional diffraction data and
performing one-dimensional azimuthal integration with pyFAI.

## Features

- Opens one or multiple EDF, CBF, TIFF, IMG, MarCCD, HDF5, or NumPy images.
- Expands multi-frame HDF5 files and navigates frames with Previous/Next.
- Loads PONI geometry and displays detector, pixel, distance, and wavelength data.
- Applies detector dead areas/module gaps and an optional user Mask.
- Requires raw Data, Empty, and Background detector intensities to use an
  integer NumPy dtype; non-integer detector images are rejected with an error.
- Displays the current frame's total intensity after detector and user masks,
  using masked pixels as zero and an int64 accumulator.
- Provides a Detector Sum tab plotting the masked detector sum for every
  selected image/frame against its sequence index.
- Calculates the all-frame Detector Sum only on the first integration, then
  reuses it until the image list, PONI geometry, or mask changes.
- Recomputes detector dummy-pixel masks per frame while safely caching static masks.
- Displays masks as a pink overlay on a `viridis` detector image.
- Integrates in a worker thread and reuses pyFAI/HDF5 caches for better performance.
- Caches the combined detector/user Mask instead of rebuilding it on cursor movement.
- Reuses unchanged Empty/Background 1-D integrations for Update, video, and ASCII export.
- Supports integration units, point count, radial range, Log X, and Log Y.
- Loads optional Empty and Background images with format validation.
- Reuses a single-frame Empty/Background for every data frame, or matches
  multi-frame references frame-by-frame when their frame counts are equal.
- Rejects multi-frame Data/Empty/Background combinations with unequal frame counts.
- Displays `Data`, scaled `Empty`/`Background`, and `Subtracted data` curves.
- Optionally calculates and displays a pyFAI azimuthal-versus-radial Cake plot above the 1-D plot; it is disabled by default, calculated after all 1-D work, and cached once per source image.
- Uses logarithmic Viridis intensity mapping by default for Source and Cake images; Cake provides the same native silx colormap tool as Source without recalculating integration.
- Applies independent Empty/Background subtraction factors.
- Saves the currently visible plot canvases, grouped batch MP4 videos, and ASCII `.dat` data.
- Provides **Data and Plots**: one source exports its ASCII data and current
  plot, while multiple images/frames export all ASCII data and grouped videos.
- Provides separate persistent Input and Export paths under **File > Path**;
  Input applies to Image/Empty/Background and Export applies to all saves.
- ASCII export includes reference filenames, factors, and reference/subtracted columns.
- **Options > Plot NeXus** first opens one NeXus/HDF5 file, then provides one
  silx file tree where selected numeric 1-D datasets can be assigned with
  **Set as X** and **Set as Y**. The selected dataset's index and raw values
  appear in a scrollable table below the tree. The X/Y data are plotted when
  their lengths match, with editable axis names. The resulting plot window has
  a checkable **Derivative** option in the native plot toolbar that adds or
  removes a `dY/dX` curve. Rectangle zoom is not constrained to the data bounds.
- **Options > Plot ASCII** selects the X and Y columns from a whitespace-,
  tab-, or comma-separated numeric table. A matching comment/header line
  immediately above the data is shown beside its corresponding column.
- Uses a low-memory display preview while integrating full-resolution data.

## Requirements

- Python 3.10 or newer
- NumPy
- PyQt6
- pyFAI
- silx
- Fabio
- h5py
- qtawesome
- imageio and imageio-ffmpeg (for MP4 export)

## Installation

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Running the Application

Run the standard version:

```powershell
python pyfai_integrate_gui.py
```

Run version 2 with rectangular ROI analysis:

```powershell
python pyfai_integrate_gui_v2.py
```

## Version 2: ROI Sum

`pyfai_integrate_gui_v2.py` contains all standard-version features and adds a
rectangular ROI workflow:

- **Select ROI** is available in the Source image toolbar. Click it and drag a
  rectangle over the required detector area. Drawing a new rectangle replaces
  the previous ROI.
- **ROI Sum** appears beside **Integration** and **Detector Sum** in the top
  menu-bar row.
- The ROI Sum X axis is the sequence index of every selected image/frame. Its Y
  axis is the total integer detector intensity inside the selected ROI for the
  corresponding frame.
- Detector-mask and user-mask pixels inside the ROI contribute zero.
- ROI coordinates refer to the full-resolution detector data, even when the
  displayed Source preview is downsampled.
- On the first integration, Detector Sum and ROI Sum are calculated together
  while each image/frame is read only once. The values are cached until the ROI,
  image list, PONI geometry, or mask changes.
- When **ROI Sum** is the selected view, plot and video export include Source
  plus the ROI Sum plot.

The standard `pyfai_integrate_gui.py` remains available without ROI controls.

## Usage

1. Click **Browse** next to **Image** and select one or more diffraction files.
   Multi-frame HDF5 files are expanded automatically.
2. Load the matching **PONI** geometry and optionally load a **Mask**.
3. Set the number of points and X-axis unit. The optional **Radial range** uses
   the selected X-axis unit; the optional **Azimuthal range** uses degrees.
   Enter a range in one field as `minimum, maximum`, or leave it blank at
   **Auto** to use the full range.
4. Optionally load **Empty** and/or **Background** reference images. Their shape
   and data type must match the sample image. A single-frame reference is reused
   for all data frames. A multi-frame reference is matched frame-by-frame and
   must have the same frame count as each multi-frame data file.
5. Click **Start Integration**.
   The first integration also calculates the Detector Sum series in the
   background. Later integrations reuse it while images, PONI, and mask remain
   unchanged.
6. Use **Show** to immediately show/hide a reference curve. Set **Subtract** and
   **Factor**, then click **Update** to recalculate the subtracted result.
7. Use **Previous/Next** to navigate selected files or HDF5 frames.
   The value below Source shows the current masked detector sum; open the
   **Detector Sum** tab to compare this value across all selected frames.
8. Use **File > Save** to export:
   - **Plot (current view)**: source and 1-D plots, plus Cake when it is visible.
   - **Batch Video**: one sorted MP4 per filename-prefix group using the same currently visible plots.
   - **ASCII (all integrated data)**: one `.dat` file per image/frame.

The image title shows the source filename. Display downsampling does not affect
the integration because pyFAI always receives the full-resolution detector data.

## Exported ASCII Data Format

**ASCII (all integrated data)** creates one tab-separated `.dat` file for each
source image or HDF5 frame. Numeric values use scientific notation with ten
digits after the decimal point. Lines beginning with `#` are comments. The last
comment line contains the column names. Without an Empty or Background image,
the file contains two columns: the selected radial coordinate (for example
`q_A^-1`) and `Intensity`. When references are loaded, preceding comment lines
record each reference filename and factor, and the columns are ordered as the
radial coordinate, `Intensity`, `Subtracted`, `Empty`, and `Background`, omitting
any reference that was not loaded. `Empty` and `Background` contain the scaled
reference intensities (`factor × integrated reference`). `Subtracted` contains
the sample intensity minus each scaled reference whose **Subtract** option was
selected. All rows share the same radial bins and selected integration unit.

Example:

```text
# Empty file: empty.tif; factor: 1.00
# Background file: background.tif; factor: 0.50
# q_A^-1    Intensity    Subtracted    Empty    Background
1.0000000000e-03    2.5000000000e+04    2.4100000000e+04    7.0000000000e+02    2.0000000000e+02
```
