# pyFAI Integrate Viewer v2

Version 2 preserves the integration, Cake, mask, Empty/Background, Detector
Sum, ASCII, plot, and video features from `pyfai_integrate_gui.py`, and adds
rectangular ROI analysis, unified operation status, and cancellable background
work.

## Run

```powershell
python pyfai_integrate_gui_v2.py
```

Load one or more Data images/frames, a matching PONI file, and optional Mask,
Empty, and Background files. Configure the integration parameters and click
**Integrate**.

## Detector Sum

Open **Detector Sum** and click its **Calculate** button to calculate the masked
integer intensity sum for every selected image/frame.

- Detector Sum is never calculated by **Integrate**.
- The value below the Source image displays only a previously calculated,
  cached Detector Sum for the current image/frame.
- Before Detector Sum has been calculated, the Source label displays
  `Detector sum intensity: not calculated` and performs no hidden summation.
- Static detector masks are reused. Dynamic dummy-pixel masks are recalculated
  per frame because they depend on that frame's pixel values.

## ROI Sum

1. Click **Select ROI** in the Source image toolbar.
2. Drag a rectangle over the required detector region. Drawing another
   rectangle replaces the previous ROI.
3. Open **ROI Sum** and click its own **Calculate** button.

ROI Sum calculates the masked integer intensity inside the rectangle for every
selected image/frame. If no valid rectangle is selected, **Calculate** reports
`ROI Required` and does not start a background task.

Detector-mask and user-mask pixels inside the ROI contribute zero. ROI
coordinates refer to the full-resolution detector data even when the displayed
Source preview is downsampled.

Detector Sum and ROI Sum are independent: calculating one does not calculate
the other. Existing curves remain visible while recalculating, after stopping,
when the ROI moves, and when PONI or Mask settings change. Both cached series
are cleared only after a new Image selection successfully replaces the current
image list.

## Status and Stop

The **Status** panel reports image/frame loading, integration, Detector/ROI Sum,
ASCII export, and video export progress. While background data is still being
loaded, **Integrate** is disabled.

**Stop** requests cancellation of the active load, mask preparation,
Detector/ROI Sum, integration, ASCII export, or video export. File/frame loops
stop at the next safe boundary. A pyFAI call already executing cannot be killed
safely; its result is discarded and the task finishes as soon as that call
returns. Previously completed Detector Sum and ROI Sum curves remain available.

## Calculation reuse

- **Integrate** performs only pyFAI 1-D integration plus requested Cake and
  Empty/Background work; it never calculates Detector Sum or ROI Sum.
- Cake, combined masks, reference integrations, and static detector masks are
  cached using the inputs that affect their numerical results.
- Batch ASCII export reads and integrates a matching Empty/Background reference
  only on a cache miss, then reuses it for compatible sample frames.
- Clicking **Integrate** again intentionally recalculates the current sample.
  Batch and video export necessarily process each requested image/frame.

## Export

Plot and video export capture the Source image plus the currently selected
right-side view. When **Detector Sum** or **ROI Sum** is selected, its cached
curve is included. ASCII export writes one integrated `.dat` file per selected
image/frame.

The original `pyfai_integrate_gui.py` remains the non-ROI version.
