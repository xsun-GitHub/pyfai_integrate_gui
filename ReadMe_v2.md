# pyFAI Integrate Viewer v2

Version 2 preserves the existing integration, Cake, mask, Empty/Background,
Detector Sum, ASCII, plot, and video features from `pyfai_integrate_gui.py` and
adds rectangular ROI intensity analysis.

## ROI Sum

1. Run `python pyfai_integrate_gui_v2.py` and load the Data, PONI, and optional
   Mask as usual.
2. Click **Select ROI** in the Source image toolbar, then drag a rectangle over
   the required detector region. Drawing another rectangle replaces the old ROI.
3. Click **Start Integration**. During the first integration, v2 reads each
   selected image/frame once and calculates both Detector Sum and ROI Sum.
4. Select **ROI Sum** beside **Integration** and **Detector Sum** in the top
   menu-bar row. X is the selected image/frame sequence index and Y is the total
   integer intensity inside the ROI for that frame.

Detector and user mask pixels inside the ROI are treated as zero. The ROI uses
full-resolution detector coordinates even when the Source preview is sampled.
ROI results are cached until the ROI, image list, PONI, or mask changes. Plot and
video export capture Source plus whichever right-side view is currently selected.

The original `pyfai_integrate_gui.py` remains the non-ROI version.
