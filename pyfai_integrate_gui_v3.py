"""pyFAI Integrate Viewer v3 with p62 beamline/ASAXS support.

Features
--------
- View single images, multiple files, and every frame of multi-frame HDF5 files.
- Load a pyFAI PONI geometry and combine its detector mask with a user mask.
- Perform responsive 1-D azimuthal integration with selectable units and range.
- Integrate, display, scale, and optionally subtract Empty/Background references.
- Browse images with Previous/Next and automatically update integrated curves.
- Cache combined masks, reference integrations, and per-image Cake results.
- Save the current plots, grouped batch videos/PNG plots, and integrated ASCII
  ``.dat`` files.
- Plot selected 1-D NeXus datasets or selected ASCII columns in separate windows.

Version 2 additions
-------------------
- Optional cached Cake integration with shared 1-D controls and native colormap.
- Single- or multi-frame Empty/Background references with frame matching,
  scaling, display, subtraction, and Clear controls.
- Detector Sum and rectangular ROI Sum are calculated only from their own
  ``Calculate`` buttons and cached per image/frame; ``Integrate`` never computes
  either sum, and the source label only displays an existing Detector Sum cache.
- Multi-frame HDF5 navigation, automatic reintegration, and safer serialized
  background file processing on Windows.
- Persistent export path, current-view plot export, grouped MP4/PNG export, and
  extended ASCII output containing reference and subtracted 1-D columns.
- Lower-memory previews, Cython Cake histogram integration, and reusable mask,
  reference, Cake, and detector-sum caches.

Version 3 additions
-------------------
- ``File > Beamline > p62`` with SAXS, WAXS, ASAXS, and AWAXS NeXus modes.
- Direct ``/scan/data/saxs_raw`` or ``/scan/data/waxs_raw`` image-stack reads.
- Per-energy wavelength updates for anomalous modes using
  ``/scan/data/energy`` and ``wavelength = hc / energy``.
- Single shared or energy-matched Empty/Background subtraction, including
  common-q interpolation and one-to-one matching for equal-length series.
- Unified fixed/scrolling Status display and lower-memory sequential video
  export for large detector stacks.
- ``Options > ASAXS`` opens the project-local PyAnomScat Stuhrmann GUI. ASAXS
  ASCII exports include ``_E<energy>`` filenames for direct import.

Basic use
---------
1. Select one or more detector images, then load the matching PONI file.
2. Optionally load a mask and Empty/Background reference images.
3. Configure integration parameters and click ``Integrate``.
4. Use ``Show`` for immediate reference visibility; use ``Update`` after changing
   Subtract or Factor settings.
5. Export results from ``File > Save``.
6. For ASAXS analysis, open ``Options > ASAXS`` and import the exported
   ``*_E<energy>.dat`` curves.
"""

from __future__ import annotations

import os
import importlib.util
import re
import sys
import threading
import traceback
import warnings
import zlib
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import fabio
import h5py
import numpy as np
import pyFAI
from silx.gui import qt
from silx.gui.colors import Colormap
from silx.gui.hdf5 import Hdf5TreeModel, Hdf5TreeView
from silx.gui.plot import Plot1D, Plot2D
from silx.gui.plot.items.roi import RectangleROI
from silx.gui.plot.tools.roi import RegionOfInterestManager


warnings.filterwarnings(
    "ignore",
    message="Ignoring fixed y limits to fulfill fixed data aspect.*",
    category=UserWarning,
)


class _MatplotlibHeavyWeightFilter(logging.Filter):
    """Hide only Qt's harmless ``heavy`` -> 700 Matplotlib font fallback."""

    def filter(self, record):
        message = record.getMessage()
        return not (
            message.startswith("findfont: Failed to find font weight heavy")
            and "now using 700" in message
        )


# silx maps a Qt heavy font to Matplotlib's equivalent numeric weight 700.
# Keep all other font-manager warnings visible so real missing fonts are reported.
logging.getLogger("matplotlib.font_manager").addFilter(
    _MatplotlibHeavyWeightFilter()
)


IMAGE_FILTER = (
    "Detector images (*.edf *.edf.gz *.cbf *.tif *.tiff *.img *.mccd *.mar3450 "
    "*.h5 *.hdf5 *.npy);;All files (*)"
)
P62_IMAGE_FILTER = "NeXus files (*.nxs);;All files (*)"
EV_TO_METRE = 1.2398419843320026e-6
# Keep the integrated analysis tool self-contained: never depend on the
# original development directory under D:\Program.
PYANOMSCAT_SCRIPT = Path(__file__).with_name(
    "pyAnomScat_stuhrmann_method_v3.py"
)
MAX_DISPLAY_DIMENSION = 1200
# pyFAI's sparse CSR lookup can require hundreds of MB for large detectors,
# especially for a 2-D cake. Keep the established 1-D method unchanged, while
# explicitly selecting the no-split Cython histogram for Cake: in pyFAI 2026,
# the bare string "histogram" resolves to the NumPy/Python implementation,
# which allocates detector-sized float64 temporary arrays.
INTEGRATION_METHOD_1D = "histogram"
INTEGRATION_METHOD_1D_ERROR = ("no", "histogram", "cython")
INTEGRATION_METHOD_2D = ("no", "histogram", "cython")


def wavelength_from_energy(energy_ev):
    """Convert photon energy in eV to wavelength in metres."""
    return EV_TO_METRE / float(energy_ev)


def read_ascii_columns(filename):
    """Return numeric ASCII columns and optional names from the preceding comment."""
    lines = Path(filename).read_text(encoding="utf-8-sig", errors="replace").splitlines()
    rows = []
    first_data_line = None
    column_count = None
    for line_number, line in enumerate(lines):
        text = line.strip()
        if not text or text.startswith(("#", ";", "%")):
            continue
        fields = [item for item in re.split(r"[\s,]+", text) if item]
        try:
            values = [float(item) for item in fields]
        except ValueError:
            continue
        if len(values) < 2:
            continue
        if column_count is None:
            column_count = len(values)
            first_data_line = line_number
        if len(values) != column_count:
            raise ValueError(
                f"Line {line_number + 1} has {len(values)} columns; "
                f"expected {column_count}."
            )
        rows.append(values)
    if not rows:
        raise ValueError("No numeric table with at least two columns was found.")

    names = None
    if first_data_line:
        header = lines[first_data_line - 1].strip()
        header = re.sub(r"^[#;%]+\s*", "", header)
        fields = [item for item in re.split(r"[\s,]+", header) if item]
        if len(fields) == column_count:
            names = fields
    return np.asarray(rows, dtype=np.float64), names


@contextmanager
def filter_libpng_iccp_warnings():
    """Hide only libpng's bad-profile noise while forwarding real stderr."""
    stderr_fd = 2
    saved_fd = os.dup(stderr_fd)
    read_fd, write_fd = os.pipe()

    def forward_stderr():
        warning = b"libpng warning: iCCP: known incorrect sRGB profile"
        with os.fdopen(read_fd, "rb", buffering=0) as stream:
            for line in iter(stream.readline, b""):
                if warning not in line:
                    os.write(saved_fd, line)

    thread = threading.Thread(target=forward_stderr, daemon=True)
    thread.start()
    try:
        os.dup2(write_fd, stderr_fd)
        os.close(write_fd)
        yield
    finally:
        try:
            sys.stderr.flush()
        except Exception:
            pass
        os.dup2(saved_fd, stderr_fd)
        thread.join(timeout=1.0)
        os.close(saved_fd)


@dataclass(frozen=True)
class ImageSource:
    path: str
    frame: int | None = None
    frame_count: int = 1
    dataset_path: str | None = None
    energy_ev: int | None = None

    @property
    def title(self):
        name = Path(self.path).name
        dataset = "" if self.dataset_path is None else f" [{Path(self.dataset_path).name}]"
        if self.frame is None:
            return f"{name}{dataset}"
        return f"{name}{dataset} [Frame {self.frame + 1}/{self.frame_count}]"

    @property
    def data_filename(self):
        path = Path(self.path)
        if self.frame is None:
            return path.with_suffix(".dat").name
        return f"{path.stem}_frame_{self.frame + 1:04d}.dat"


def read_p62_energy_values(filename):
    """Return all finite numeric /scan/data/energy values converted to integer eV."""
    with h5py.File(filename, "r") as nexus:
        dataset = nexus.get("/scan/data/energy")
        if not isinstance(dataset, h5py.Dataset):
            raise KeyError("Missing NeXus dataset /scan/data/energy")
        values = np.asarray(dataset[()])
    if values.size < 1 or not np.issubdtype(values.dtype, np.number):
        raise TypeError("/scan/data/energy must contain numeric energy values")
    numeric = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(numeric)):
        raise ValueError("/scan/data/energy must contain only finite energy values")
    result = [int(value) for value in numeric]
    if any(value <= 0 for value in result):
        raise ValueError("/scan/data/energy values must be greater than zero")
    return result


def expand_image_file(filename: str, dataset_path=None, include_energy=False) -> list[ImageSource]:
    """Expand a multi-frame HDF5 file into individually selectable frames."""
    if dataset_path is not None:
        with h5py.File(filename, "r") as nexus:
            dataset = nexus.get(dataset_path)
            if not isinstance(dataset, h5py.Dataset):
                raise KeyError(f"Missing NeXus image dataset {dataset_path}")
            if dataset.ndim == 2:
                count = 1
                is_stack = False
            elif dataset.ndim != 3:
                raise ValueError(
                    f"{dataset_path} must be a 2-D image or 3-D image stack; "
                    f"received shape {dataset.shape}"
                )
            else:
                count = int(dataset.shape[0])
                is_stack = True
        if count < 1:
            raise ValueError(f"{dataset_path} contains no image frames")
        energies = read_p62_energy_values(filename) if include_energy else [None] * count
        if include_energy and len(energies) != count:
            if count % len(energies) != 0:
                raise ValueError(
                    f"{Path(filename).name}: {dataset_path} contains {count} image(s), "
                    f"but /scan/data/energy contains {len(energies)} value(s); "
                    "the image count must be a multiple of the energy count"
                )
            repeats_per_energy = count // len(energies)
            energies = [
                energy
                for energy in energies
                for _ in range(repeats_per_energy)
            ]
        if count == 1:
            return [ImageSource(
                filename,
                frame=0 if is_stack else None,
                frame_count=1,
                dataset_path=dataset_path,
                energy_ev=energies[0],
            )]
        return [
            ImageSource(filename, frame, count, dataset_path, energies[frame])
            for frame in range(count)
        ]
    if Path(filename).suffix.lower() not in (".h5", ".hdf5", ".nxs"):
        return [ImageSource(filename)]
    image = fabio.open(filename)
    try:
        count = max(1, int(image.nframes))
    finally:
        image.close()
    if count == 1:
        return [ImageSource(filename)]
    return [ImageSource(filename, frame, count) for frame in range(count)]


def read_image(source: str | ImageSource) -> np.ndarray:
    """Read a detector image using Fabio, with a small convenience for NPY."""
    if isinstance(source, ImageSource):
        filename, frame, dataset_path = source.path, source.frame, source.dataset_path
    else:
        filename, frame, dataset_path = source, None, None
    try:
        if dataset_path is not None:
            with h5py.File(filename, "r") as nexus:
                dataset = nexus[dataset_path]
                data = np.asarray(dataset[()] if frame is None else dataset[frame]).copy()
        elif filename.lower().endswith(".npy"):
            data = np.load(filename)
        else:
            image = fabio.open(filename)
            try:
                if frame is None:
                    data = np.asarray(image.data).copy()
                else:
                    data = np.asarray(image.getframe(frame).data).copy()
            finally:
                image.close()
    except Exception as exc:
        frame_text = "" if frame is None else f", frame {frame + 1}"
        raise OSError(
            f"Unable to read detector image '{filename}'{frame_text}: {exc}"
        ) from exc
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(f"A 2-D image is required; the data shape is {data.shape}")
    return data


def require_integer_detector_image(image, label="Detector image"):
    """Reject non-integer raw detector data before display or integration."""
    if not np.issubdtype(image.dtype, np.integer):
        raise TypeError(
            f"{label} must contain integer detector intensity; "
            f"received data type {image.dtype}"
        )


def matching_reference_source(
    reference_sources, sample_source, sample_sources=None, sample_index=None
):
    """Match references one-to-one for equal lengths, otherwise by energy."""
    if len(reference_sources) == 1:
        return reference_sources[0]
    if sample_sources is not None and len(reference_sources) == len(sample_sources):
        if sample_index is None:
            sample_index = sample_sources.index(sample_source)
        reference = reference_sources[sample_index]
        if (
            sample_source.energy_ev is not None
            and reference.energy_ev is not None
            and sample_source.energy_ev != reference.energy_ev
        ):
            raise ValueError(
                f"Energy mismatch at image {sample_index + 1}: sample is "
                f"{sample_source.energy_ev} eV, reference is {reference.energy_ev} eV"
            )
        return reference
    if isinstance(sample_source, ImageSource) and sample_source.energy_ev is not None:
        matches = [
            source for source in reference_sources
            if source.energy_ev == sample_source.energy_ev
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one reference image at {sample_source.energy_ev} eV; "
                f"found {len(matches)}"
            )
        return matches[0]
    sample_count = (
        sample_source.frame_count
        if isinstance(sample_source, ImageSource) and sample_source.frame is not None
        else 1
    )
    if sample_count != len(reference_sources):
        raise ValueError(
            f"Data has {sample_count} frame(s), but the reference has "
            f"{len(reference_sources)} frames"
        )
    return reference_sources[sample_source.frame]


def detector_accepts_shape(detector, image_shape) -> bool:
    """Return whether a shape is the full detector or an integer-binned form."""
    shape = tuple(int(value) for value in image_shape[:2])
    detector_shape = getattr(detector, "shape", None)
    if detector_shape is not None and tuple(detector_shape) == shape:
        return True
    max_shape = getattr(detector, "max_shape", None)
    if max_shape is None:
        return True  # Generic detectors have no fixed pixel layout to validate
    max_shape = tuple(int(value) for value in max_shape)
    if max_shape == shape:
        return True
    if any(value <= 0 for value in shape):
        return False
    return all(maximum % requested == 0 for maximum, requested in zip(max_shape, shape))


def detector_mask_for_image(detector, image):
    """Return static + per-image dummy mask without mutating detector.mask."""
    detector.guess_binning(image)
    static_mask = detector.mask
    dummy = getattr(detector, "dummy", None)
    if dummy is None:
        return None if static_mask is None else np.asarray(static_mask, dtype=bool)
    actual_dummy = np.dtype(image.dtype).type(np.int64(dummy))
    delta_dummy = getattr(detector, "delta_dummy", None)
    if delta_dummy is None:
        dummy_mask = image == actual_dummy
    else:
        dummy_mask = np.abs(float(actual_dummy) - image) < delta_dummy
    if static_mask is None:
        return np.asarray(dummy_mask, dtype=bool)
    return np.logical_or(np.asarray(static_mask, dtype=bool), dummy_mask)


def mask_checksum(mask):
    """Return a stable checksum without materializing a bytes copy."""
    if mask is None:
        return None
    contiguous = np.ascontiguousarray(mask, dtype=np.bool_)
    return zlib.crc32(memoryview(contiguous))


def masked_intensity_sum(image, mask=None):
    """Sum unmasked integer detector pixels with an int64 accumulator."""
    require_integer_detector_image(image)
    if mask is None:
        return int(np.sum(image, dtype=np.int64))
    if mask.shape != image.shape:
        raise ValueError(
            f"Mask shape {mask.shape} does not match image shape {image.shape}"
        )
    valid = np.logical_not(mask)
    return int(np.sum(image, where=valid, dtype=np.int64))


def group_batch_video_sources(sources):
    """Group by the stem before the final _number and sort each group numerically."""
    groups = {}
    for source in sources:
        path = source.path if isinstance(source, ImageSource) else str(source)
        stem = Path(path).stem
        match = re.fullmatch(r"(.+)_([0-9]+)", stem)
        prefix, number = (match.group(1), int(match.group(2))) if match else (stem, 0)
        groups.setdefault(prefix, []).append((number, source))
    return [
        (prefix, [item[1] for item in sorted(groups[prefix], key=lambda item: item[0])])
        for prefix in sorted(groups, key=str.casefold)
    ]


class IntegrationWorker(qt.QObject):
    finished = qt.Signal(object)
    failed = qt.Signal(str)
    cancelled = qt.Signal()

    def __init__(
        self, image, integrator, mask, points, unit, radial_range,
        azimuth_range, error_model, references, calculate_cake, cached_cake,
        cake_cache_key,
    ):
        super().__init__()
        self.image = image
        self.integrator = integrator
        self.mask = mask
        self.points = points
        self.unit = unit
        self.radial_range = radial_range
        self.azimuth_range = azimuth_range
        self.error_model = error_model
        self.references = references
        self.calculate_cake = calculate_cake
        self.cached_cake = cached_cake
        self.cake_cache_key = cake_cache_key
        self._cancel_requested = threading.Event()

    def cancel(self):
        self._cancel_requested.set()

    def _check_cancelled(self):
        if self._cancel_requested.is_set():
            raise InterruptedError

    @qt.Slot()
    def run(self):
        try:
            self._check_cancelled()
            result = self.integrator.integrate1d(
                self.image,
                self.points,
                mask=self.mask,
                unit=self.unit,
                radial_range=self.radial_range,
                azimuth_range=self.azimuth_range,
                method=(
                    INTEGRATION_METHOD_1D_ERROR
                    if self.error_model is not None else INTEGRATION_METHOD_1D
                ),
                correctSolidAngle=True,
                error_model=self.error_model,
            )
            payload = {
                "radial": np.asarray(result.radial),
                "sample": np.asarray(result.intensity),
                "sigma": (
                    None if self.error_model is None or result.sigma is None
                    else np.asarray(result.sigma)
                ),
                "unit": str(result.unit),
                "references": {},
                "reference_cache_keys": {},
                "reference_cache_values": {},
                "cake": self.cached_cake,
                "cake_cache_key": self.cake_cache_key,
            }
            reference_results = []
            for name, data, show, subtract, factor, cache_key, cached in self.references:
                self._check_cancelled()
                if cached is None:
                    ref_result = self.integrator.integrate1d(
                        data, self.points, mask=self.mask, unit=self.unit,
                        radial_range=self.radial_range,
                        method=(
                            INTEGRATION_METHOD_1D_ERROR
                            if self.error_model is not None else INTEGRATION_METHOD_1D
                        ),
                        azimuth_range=self.azimuth_range,
                        correctSolidAngle=True,
                        error_model=self.error_model,
                    )
                    ref_radial = np.asarray(ref_result.radial)
                    intensity = np.asarray(ref_result.intensity)
                    sigma = (
                        None if self.error_model is None or ref_result.sigma is None
                        else np.asarray(ref_result.sigma)
                    )
                else:
                    if isinstance(cached, tuple) and len(cached) == 3:
                        ref_radial, intensity, sigma = cached
                    elif isinstance(cached, tuple) and len(cached) == 2:
                        ref_radial, intensity = cached
                        sigma = None
                    else:
                        ref_radial, intensity = payload["radial"], cached
                        sigma = None
                    ref_radial = np.asarray(ref_radial)
                    intensity = np.asarray(intensity)
                reference_results.append((
                    name, ref_radial, intensity, sigma, show, subtract, factor,
                    cache_key
                ))
                payload["reference_cache_keys"][name] = cache_key
                payload["reference_cache_values"][name] = (
                    ref_radial, intensity, sigma
                )
            common = np.ones(payload["radial"].shape, dtype=bool)
            for _name, ref_radial, _intensity, _sigma, _show, subtract, _factor, _key in reference_results:
                if subtract:
                    common &= (
                        (payload["radial"] >= np.min(ref_radial))
                        & (payload["radial"] <= np.max(ref_radial))
                    )
            if not np.any(common):
                raise ValueError("Sample and reference curves have no common q range")
            payload["radial"] = payload["radial"][common]
            payload["sample"] = payload["sample"][common]
            if payload["sigma"] is not None:
                payload["sigma"] = payload["sigma"][common]
            corrected = payload["sample"].copy()
            corrected_variance = (
                None if payload["sigma"] is None else payload["sigma"] ** 2
            )
            for name, ref_radial, intensity, sigma, show, subtract, factor, _key in reference_results:
                if ref_radial[0] > ref_radial[-1]:
                    ref_radial, intensity = ref_radial[::-1], intensity[::-1]
                    if sigma is not None:
                        sigma = sigma[::-1]
                aligned = np.interp(payload["radial"], ref_radial, intensity)
                payload["references"][name] = (aligned, show, subtract, factor)
                if subtract:
                    corrected -= factor * aligned
                    if corrected_variance is not None and sigma is not None:
                        aligned_sigma = np.interp(
                            payload["radial"], ref_radial, sigma
                        )
                        corrected_variance += (factor * aligned_sigma) ** 2
            payload["corrected"] = corrected
            payload["corrected_sigma"] = (
                None if corrected_variance is None
                else np.sqrt(corrected_variance)
            )
            # Cake is deliberately last: all 1-D and subtraction work finishes
            # before the higher-memory 2-D integration starts.
            if self.calculate_cake and payload["cake"] is None:
                self._check_cancelled()
                cake = self.integrator.integrate2d(
                    self.image, self.points, 360, mask=self.mask, unit=self.unit,
                    radial_range=self.radial_range,
                    azimuth_range=self.azimuth_range,
                    method=INTEGRATION_METHOD_2D, correctSolidAngle=True,
                )
                payload["cake"] = {
                    "intensity": np.asarray(cake.intensity),
                    "radial": np.asarray(cake.radial),
                    "azimuthal": np.asarray(cake.azimuthal),
                }
            self._check_cancelled()
            self.finished.emit(payload)
        except InterruptedError:
            self.cancelled.emit()
        except Exception:
            self.failed.emit(traceback.format_exc())


class DetectorMaskWorker(qt.QObject):
    finished = qt.Signal(object, int, str)
    cancelled = qt.Signal()

    def __init__(self, detector, image, generation):
        super().__init__()
        self.detector = detector
        self.image = image
        self.generation = generation
        self._cancel_requested = threading.Event()

    def cancel(self):
        self._cancel_requested.set()

    @qt.Slot()
    def run(self):
        try:
            if self._cancel_requested.is_set():
                self.cancelled.emit()
                return
            mask = detector_mask_for_image(self.detector, self.image)
            if self._cancel_requested.is_set():
                self.cancelled.emit()
                return
            if mask is not None:
                mask = np.asarray(mask, dtype=bool)
                if mask.shape != self.image.shape:
                    mask = None
            self.finished.emit(mask, self.generation, "")
        except Exception:
            self.finished.emit(None, self.generation, traceback.format_exc())


class DetectorSumWorker(qt.QObject):
    progress = qt.Signal(int, object, object, int)
    finished = qt.Signal(object, int)
    failed = qt.Signal(str, int)
    cancelled = qt.Signal()

    def __init__(
        self, sources, poni, user_mask, generation, roi_bounds=None,
        mode="detector", detector_mask_cache=None,
    ):
        super().__init__()
        self.sources = sources
        self.poni = poni
        self.user_mask = user_mask
        self.generation = generation
        self.roi_bounds = roi_bounds
        self.mode = mode
        # Reuse static masks already prepared by the GUI. Dynamic dummy masks
        # are intentionally recalculated per frame because their pixels differ.
        self.detector_mask_cache = dict(detector_mask_cache or {})
        self._cancel_requested = threading.Event()

    def cancel(self):
        self._cancel_requested.set()

    @qt.Slot()
    def run(self):
        try:
            integrator = pyFAI.load(self.poni) if self.poni else None
            static_mask_cache = dict(self.detector_mask_cache)
            values = []
            roi_values = []
            for index, source in enumerate(self.sources):
                if self._cancel_requested.is_set():
                    self.cancelled.emit()
                    return
                image = read_image(source)
                require_integer_detector_image(image, source.title)
                detector_mask = None
                if integrator is not None:
                    detector = integrator.detector
                    dynamic = getattr(detector, "dummy", None) is not None
                    shape_key = tuple(image.shape)
                    if dynamic or shape_key not in static_mask_cache:
                        detector_mask = detector_mask_for_image(detector, image)
                        if detector_mask is not None:
                            detector_mask = np.asarray(detector_mask, dtype=bool)
                        if not dynamic:
                            static_mask_cache[shape_key] = detector_mask
                    else:
                        detector_mask = static_mask_cache[shape_key]
                mask = detector_mask
                if self.user_mask is not None:
                    if self.user_mask.shape != image.shape:
                        raise ValueError(
                            f"{source.title}: user mask shape {self.user_mask.shape} "
                            f"does not match image shape {image.shape}"
                        )
                    mask = (
                        self.user_mask if mask is None
                        else np.logical_or(mask, self.user_mask)
                    )
                detector_value = None
                roi_value = None
                if self.mode == "detector":
                    detector_value = masked_intensity_sum(image, mask)
                    values.append(detector_value)
                elif self.mode == "roi" and self.roi_bounds is not None:
                    left, top, right, bottom = self.roi_bounds
                    roi_mask = None if mask is None else mask[top:bottom, left:right]
                    roi_value = masked_intensity_sum(
                        image[top:bottom, left:right], roi_mask
                    )
                    roi_values.append(roi_value)
                self.progress.emit(
                    index, detector_value, roi_value, self.generation
                )
                del image, mask, detector_mask
            self.finished.emit(
                {
                    "mode": self.mode,
                    "values": np.asarray(
                        values if self.mode == "detector" else roi_values,
                        dtype=np.int64,
                    ),
                },
                self.generation,
            )
        except Exception:
            self.failed.emit(traceback.format_exc(), self.generation)


class BatchIntegrationWorker(qt.QObject):
    progress = qt.Signal(int, int, str)
    finished = qt.Signal(str, int)
    failed = qt.Signal(str)
    cancelled = qt.Signal()

    def __init__(
        self, paths, output_dir, poni, user_mask, points, unit, radial_range,
        azimuth_range, error_model,
        references=None, use_source_energy=False,
        include_energy_in_filename=False,
    ):
        super().__init__()
        self.paths = paths
        self.output_dir = output_dir
        self.poni = poni
        self.user_mask = user_mask
        self.points = points
        self.unit = unit
        self.radial_range = radial_range
        self.azimuth_range = azimuth_range
        self.error_model = error_model
        self.references = references or []
        self.use_source_energy = use_source_energy
        self.include_energy_in_filename = include_energy_in_filename
        self._cancel_requested = threading.Event()

    def cancel(self):
        self._cancel_requested.set()

    @qt.Slot()
    def run(self):
        readers = {}
        try:
            integrator = pyFAI.load(self.poni)
            mask_cache = {}
            reference_cache = {}
            total = len(self.paths)
            for index, path in enumerate(self.paths, 1):
                if self._cancel_requested.is_set():
                    self.cancelled.emit()
                    return
                if (
                    isinstance(path, ImageSource)
                    and path.frame is not None
                    and path.dataset_path is None
                ):
                    reader = readers.get(path.path)
                    if reader is None:
                        reader = fabio.open(path.path)
                        readers[path.path] = reader
                    image = np.asarray(reader.getframe(path.frame).data)
                else:
                    image = read_image(path)
                source_name = path.title if isinstance(path, ImageSource) else Path(path).name
                energy_ev = (
                    path.energy_ev
                    if self.use_source_energy and isinstance(path, ImageSource)
                    else None
                )
                if energy_ev is not None:
                    integrator.wavelength = wavelength_from_energy(energy_ev)
                require_integer_detector_image(image, source_name)
                detector = integrator.detector
                if not detector_accepts_shape(detector, image.shape):
                    raise ValueError(
                        f"{source_name}: image shape {image.shape} is incompatible "
                        f"with detector shape {detector.max_shape}"
                    )
                shape_key = tuple(image.shape)
                dynamic_detector_mask = getattr(detector, "dummy", None) is not None
                if dynamic_detector_mask or shape_key not in mask_cache:
                    detector_mask = detector_mask_for_image(detector, image)
                    mask = None if detector_mask is None else np.asarray(detector_mask, dtype=bool)
                    if self.user_mask is not None:
                        if self.user_mask.shape != image.shape:
                            raise ValueError(
                                f"{source_name}: user mask shape {self.user_mask.shape} "
                                f"does not match image shape {image.shape}"
                            )
                        mask = self.user_mask if mask is None else np.logical_or(mask, self.user_mask)
                    if not dynamic_detector_mask:
                        mask_cache[shape_key] = mask
                else:
                    mask = mask_cache[shape_key]
                mask_key = (shape_key, mask_checksum(mask))
                result = integrator.integrate1d(
                    image, self.points, mask=mask, unit=self.unit,
                    radial_range=self.radial_range,
                    method=(
                        INTEGRATION_METHOD_1D_ERROR
                        if self.error_model is not None else INTEGRATION_METHOD_1D
                    ),
                    azimuth_range=self.azimuth_range,
                    correctSolidAngle=True,
                    error_model=self.error_model,
                )
                columns = [result.radial, result.intensity]
                column_names = [str(result.unit), "Intensity"]
                if self.error_model is not None and result.sigma is not None:
                    columns.append(result.sigma)
                    column_names.append("Sigma")
                comments = []
                if isinstance(path, ImageSource) and path.energy_ev is not None:
                    comments.append(f"Energy: {path.energy_ev} eV")
                    comments.append(
                        f"Wavelength used: {integrator.wavelength:.12g} m"
                    )
                corrected = np.asarray(result.intensity).copy()
                reference_columns = []
                common_q = np.ones(np.asarray(result.radial).shape, dtype=bool)
                for name, ref_sources, _ref_file, factor, subtract in self.references:
                    if self._cancel_requested.is_set():
                        self.cancelled.emit()
                        return
                    ref_source = matching_reference_source(
                        ref_sources, path, self.paths, index - 1
                    )
                    # Key before reading: a cache hit avoids both reopening the
                    # reference frame and repeating its pyFAI integration.
                    cache_key = (
                        name,
                        os.path.abspath(ref_source.path),
                        ref_source.frame,
                        mask_key,
                        integrator.wavelength,
                    )
                    reference_curve = reference_cache.get(cache_key)
                    if reference_curve is None:
                        if ref_source.frame is not None and ref_source.dataset_path is None:
                            ref_reader = readers.get(ref_source.path)
                            if ref_reader is None:
                                ref_reader = fabio.open(ref_source.path)
                                readers[ref_source.path] = ref_reader
                            ref_data = np.asarray(
                                ref_reader.getframe(ref_source.frame).data
                            )
                        else:
                            ref_data = read_image(ref_source)
                        if (
                            ref_data.shape != image.shape
                            or ref_data.dtype != image.dtype
                        ):
                            raise ValueError(
                                f"{name} format does not match {source_name}"
                            )
                        ref_result = integrator.integrate1d(
                            ref_data, self.points, mask=mask, unit=self.unit,
                            radial_range=self.radial_range,
                            method=(
                                INTEGRATION_METHOD_1D_ERROR
                                if self.error_model is not None
                                else INTEGRATION_METHOD_1D
                            ),
                            azimuth_range=self.azimuth_range,
                            correctSolidAngle=True,
                            error_model=self.error_model,
                        )
                        reference_curve = (
                            np.asarray(ref_result.radial),
                            np.asarray(ref_result.intensity),
                        )
                        reference_cache[cache_key] = reference_curve
                    ref_radial, reference_intensity = reference_curve
                    if ref_radial[0] > ref_radial[-1]:
                        ref_radial = ref_radial[::-1]
                        reference_intensity = reference_intensity[::-1]
                    if subtract:
                        common_q &= (
                            (result.radial >= np.min(ref_radial))
                            & (result.radial <= np.max(ref_radial))
                        )
                    aligned = np.interp(result.radial, ref_radial, reference_intensity)
                    scaled = factor * aligned
                    if subtract:
                        corrected -= scaled
                    reference_columns.append((name, scaled))
                    comments.append(
                        f"{name} file: {ref_source.title}; factor: {factor:.2f}"
                    )
                if reference_columns:
                    if not np.any(common_q):
                        raise ValueError(
                            f"{source_name}: sample and reference curves have no common q range"
                        )
                    columns = [np.asarray(column)[common_q] for column in columns]
                    corrected = corrected[common_q]
                    reference_columns = [
                        (name, values[common_q])
                        for name, values in reference_columns
                    ]
                    columns.append(corrected)
                    column_names.append("Subtracted")
                    for name, scaled in reference_columns:
                        columns.append(scaled)
                        column_names.append(name)
                output_name = (
                    path.data_filename if isinstance(path, ImageSource)
                    else Path(path).with_suffix(".dat").name
                )
                if (
                    self.include_energy_in_filename
                    and isinstance(path, ImageSource)
                    and path.energy_ev is not None
                ):
                    # PyAnomScat's importer recognizes the trailing _E<float>
                    # token and uses it as the photon energy in eV.
                    output_path_name = Path(output_name)
                    output_name = (
                        f"{output_path_name.stem}_E{path.energy_ev}"
                        f"{output_path_name.suffix}"
                    )
                output_path = Path(self.output_dir) / output_name
                np.savetxt(
                    output_path,
                    np.column_stack(columns),
                    fmt="%.10e",
                    delimiter="\t",
                    header="\n".join(comments + ["\t".join(column_names)]),
                )
                self.progress.emit(index, total, output_name)
            self.finished.emit(self.output_dir, total)
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            for reader in readers.values():
                reader.close()


class ImageLoadWorker(qt.QObject):
    progress = qt.Signal(int, int, str)
    finished = qt.Signal(object)
    failed = qt.Signal(str)
    cancelled = qt.Signal()

    def __init__(self, filenames, dataset_path=None, include_energy=False):
        super().__init__()
        self.filenames = list(filenames)
        self.dataset_path = dataset_path
        self.include_energy = include_energy
        self._cancel_requested = threading.Event()

    def cancel(self):
        self._cancel_requested.set()

    @qt.Slot()
    def run(self):
        try:
            sources = []
            total = len(self.filenames)
            for index, filename in enumerate(self.filenames, 1):
                if self._cancel_requested.is_set():
                    self.cancelled.emit()
                    return
                sources.extend(expand_image_file(
                    filename, self.dataset_path, self.include_energy
                ))
                self.progress.emit(index, total, Path(filename).name)
            if self._cancel_requested.is_set():
                self.cancelled.emit()
            else:
                self.finished.emit(sources)
        except Exception:
            self.failed.emit(traceback.format_exc())


class RawArrayTableModel(qt.QAbstractTableModel):
    """Lightweight table model for browsing a potentially long 1-D array."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._values = np.empty(0)

    def setValues(self, values):
        self.beginResetModel()
        self._values = np.asarray(values)
        self.endResetModel()

    def rowCount(self, parent=qt.QModelIndex()):
        return 0 if parent.isValid() else self._values.size

    def columnCount(self, parent=qt.QModelIndex()):
        return 0 if parent.isValid() else 2

    def data(self, index, role=qt.Qt.DisplayRole):
        if not index.isValid() or role != qt.Qt.DisplayRole:
            return None
        row = index.row()
        if index.column() == 0:
            return str(row)
        return str(self._values[row])

    def headerData(self, section, orientation, role=qt.Qt.DisplayRole):
        if role != qt.Qt.DisplayRole:
            return None
        if orientation == qt.Qt.Horizontal:
            return ("Index", "Raw value")[section]
        return str(section)


class NexusDatasetDialog(qt.QDialog):
    """Select X and Y datasets from one NeXus/HDF5 file tree."""

    def __init__(self, filename, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Plot NeXus Data — {Path(filename).name}")
        self.resize(760, 620)
        self.filename = filename
        self.x_selection = None
        self.y_selection = None
        layout = qt.QVBoxLayout(self)
        layout.addWidget(qt.QLabel(
            "Select a numeric 1-D dataset in the tree, then assign it to X or Y."
        ))
        splitter = qt.QSplitter(qt.Qt.Vertical)
        self.tree = Hdf5TreeView(self)
        self.model = Hdf5TreeModel(self)
        self.tree.setModel(self.model)
        self.model.appendFile(filename)
        splitter.addWidget(self.tree)
        preview_container = qt.QWidget(self)
        preview_layout = qt.QVBoxLayout(preview_container)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_label = qt.QLabel("Select a numeric 1-D dataset to preview its raw values.")
        self.raw_table = qt.QTableView(preview_container)
        self.raw_table_model = RawArrayTableModel(self.raw_table)
        self.raw_table.setModel(self.raw_table_model)
        self.raw_table.setAlternatingRowColors(True)
        self.raw_table.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.raw_table.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        self.raw_table.horizontalHeader().setStretchLastSection(True)
        preview_layout.addWidget(self.preview_label)
        preview_layout.addWidget(self.raw_table)
        splitter.addWidget(preview_container)
        splitter.setSizes([350, 230])
        layout.addWidget(splitter)
        self.tree.selectionModel().selectionChanged.connect(
            self._preview_current_selection
        )
        assignment = qt.QGridLayout()
        self.x_path = qt.QLineEdit()
        self.y_path = qt.QLineEdit()
        self.x_path.setReadOnly(True)
        self.y_path.setReadOnly(True)
        set_x = qt.QPushButton("Set as X")
        set_y = qt.QPushButton("Set as Y")
        assignment.addWidget(qt.QLabel("X data"), 0, 0)
        assignment.addWidget(self.x_path, 0, 1)
        assignment.addWidget(set_x, 0, 2)
        assignment.addWidget(qt.QLabel("Y data"), 1, 0)
        assignment.addWidget(self.y_path, 1, 1)
        assignment.addWidget(set_y, 1, 2)
        self.x_label_edit = qt.QLineEdit("x")
        self.y_label_edit = qt.QLineEdit("y")
        assignment.addWidget(qt.QLabel("X-axis name"), 2, 0)
        assignment.addWidget(self.x_label_edit, 2, 1, 1, 2)
        assignment.addWidget(qt.QLabel("Y-axis name"), 3, 0)
        assignment.addWidget(self.y_label_edit, 3, 1, 1, 2)
        layout.addLayout(assignment)
        set_x.clicked.connect(lambda: self._assign_dataset("X"))
        set_y.clicked.connect(lambda: self._assign_dataset("Y"))
        buttons = qt.QDialogButtonBox(
            qt.QDialogButtonBox.Ok | qt.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @qt.Slot()
    def _preview_current_selection(self, *_args):
        self.raw_table_model.setValues(np.empty(0))
        nodes = list(self.tree.selectedH5Nodes())
        if len(nodes) != 1:
            self.preview_label.setText("Select one dataset.")
            return
        obj = nodes[0].h5py_object
        if not isinstance(obj, h5py.Dataset):
            self.preview_label.setText("The selected item is not a dataset.")
            return
        if obj.ndim != 1 or not np.issubdtype(obj.dtype, np.number):
            self.preview_label.setText(
                f"{nodes[0].physical_name}: preview requires numeric 1-D data."
            )
            return
        values = np.asarray(obj[()])
        self.preview_label.setText(
            f"{nodes[0].physical_name} — raw values, {values.size:,} points, "
            f"dtype {values.dtype}"
        )
        self.raw_table_model.setValues(values)

    def _current_dataset(self):
        nodes = list(self.tree.selectedH5Nodes())
        if len(nodes) != 1:
            qt.QMessageBox.warning(self, "Select Dataset", "Select one dataset.")
            return None
        obj = nodes[0].h5py_object
        if not isinstance(obj, h5py.Dataset):
            qt.QMessageBox.warning(self, "Select Dataset", "The selected item is not a dataset.")
            return None
        if obj.ndim != 1 or not np.issubdtype(obj.dtype, np.number):
            qt.QMessageBox.warning(
                self, "Invalid Dataset",
                "Select a numeric one-dimensional dataset.",
            )
            return None
        return (
            nodes[0].physical_filename,
            nodes[0].physical_name,
            np.asarray(obj[()]),
        )

    def _assign_dataset(self, role):
        selection = self._current_dataset()
        if selection is None:
            return
        if role == "X":
            self.x_selection = selection
            self.x_path.setText(selection[1])
        else:
            self.y_selection = selection
            self.y_path.setText(selection[1])

    def accept(self):
        if self.x_selection is None or self.y_selection is None:
            qt.QMessageBox.warning(
                self, "Missing Dataset", "Assign both an X dataset and a Y dataset."
            )
            return
        if self.x_selection[2].size != self.y_selection[2].size:
            qt.QMessageBox.critical(
                self, "Dataset Length Mismatch",
                f"X contains {self.x_selection[2].size} values, but Y contains "
                f"{self.y_selection[2].size} values.",
            )
            return
        super().accept()

    def selection(self):
        return (
            self.x_selection,
            self.y_selection,
            self.x_label_edit.text().strip() or "x",
            self.y_label_edit.text().strip() or "y",
        )

    def done(self, result):
        # Hdf5TreeModel owns its opened file. Close it immediately so Windows
        # does not keep the selected NeXus file locked after this dialog.
        self.model.clear()
        super().done(result)


class XYSelectionDialog(qt.QDialog):
    """Select X/Y inputs and editable axis labels."""

    def __init__(self, title, choices, parent=None, default_y=1):
        super().__init__(parent)
        self.setWindowTitle(title)
        layout = qt.QFormLayout(self)
        self.x_combo = qt.QComboBox()
        self.y_combo = qt.QComboBox()
        self.x_combo.addItems(choices)
        self.y_combo.addItems(choices)
        self.x_combo.setCurrentIndex(0)
        self.y_combo.setCurrentIndex(min(default_y, len(choices) - 1))
        self.x_label_edit = qt.QLineEdit("x")
        self.y_label_edit = qt.QLineEdit("y")
        layout.addRow("X data", self.x_combo)
        layout.addRow("Y data", self.y_combo)
        layout.addRow("X-axis name", self.x_label_edit)
        layout.addRow("Y-axis name", self.y_label_edit)
        buttons = qt.QDialogButtonBox(
            qt.QDialogButtonBox.Ok | qt.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def selection(self):
        return (
            self.x_combo.currentIndex(),
            self.y_combo.currentIndex(),
            self.x_label_edit.text().strip() or "x",
            self.y_label_edit.text().strip() or "y",
        )


class ExternalCurveWindow(qt.QMainWindow):
    """Independent silx window used for Options > Plot commands."""

    def __init__(
        self, x, y, x_label, y_label, title, parent=None,
    ):
        super().__init__(parent)
        self.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.setWindowTitle(title)
        self.resize(900, 650)
        self.x_values = np.asarray(x)
        self.y_values = np.asarray(y)
        self.x_label = x_label
        self.y_label = y_label
        self.plot = Plot1D(self)
        self.setCentralWidget(self.plot)
        self.plot.setGraphTitle(title)
        self.plot.setGraphXLabel(x_label)
        self.plot.setGraphYLabel(y_label)
        # Leave space outside the data and keep the view unconstrained so the
        # native silx rectangle-zoom can start/end beyond the data bounds.
        self.plot.setDataMargins(0.05, 0.05, 0.05, 0.05)
        self.plot.addCurve(
            self.x_values, self.y_values, legend="Data", resetzoom=True
        )
        self.derivative_action = self.plot.toolBar().addAction("Derivative")
        self.derivative_action.setCheckable(True)
        self.derivative_action.setToolTip("Show or hide the numerical dY/dX curve")
        self.derivative_action.toggled.connect(self._toggle_derivative)

    @qt.Slot(bool)
    def _toggle_derivative(self, enabled):
        legend = f"d({self.y_label})/d({self.x_label})"
        if not enabled:
            self.plot.removeCurve(legend)
            return
        x_values = np.asarray(self.x_values, dtype=np.float64)
        y_values = np.asarray(self.y_values, dtype=np.float64)
        if x_values.size < 2 or np.any(np.diff(x_values) == 0):
            self.derivative_action.blockSignals(True)
            self.derivative_action.setChecked(False)
            self.derivative_action.blockSignals(False)
            qt.QMessageBox.critical(
                self, "Cannot Calculate Derivative",
                "Derivative requires at least two points and no repeated "
                "adjacent X values.",
            )
            return
        self.plot.addCurve(
            x_values, np.gradient(y_values, x_values),
            legend=legend, resetzoom=False,
        )
        self.plot.resetZoom()


class MultiAsciiCurveWindow(qt.QMainWindow):
    """Plot ASCII curves with visibility controls and independent log axes."""

    def __init__(self, curves, x_label, y_label, parent=None):
        super().__init__(parent)
        self.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.setWindowTitle("Multiple ASCII Data")
        self.resize(1050, 700)
        self.plot = Plot1D(self)
        self.setCentralWidget(self.plot)
        self.plot.setGraphTitle("Multiple ASCII Data")
        self.plot.setGraphXLabel(x_label)
        self.plot.setGraphYLabel(y_label)
        self.plot.setDataMargins(0.05, 0.05, 0.05, 0.05)
        self._curve_data = {}
        self._curve_items = {}

        # Expose silx's native logarithmic-axis actions in Plot1D's main
        # toolbar, exactly where the main integration window shows its plot
        # controls. The interactive-mode toolbar can be hidden by silx.
        plot_toolbar = self.plot.toolBar()
        plot_toolbar.addSeparator()
        self.log_x_action = self.plot.getXAxisLogarithmicAction()
        self.log_y_action = self.plot.getYAxisLogarithmicAction()
        self.log_x_action.setText("Log X")
        self.log_y_action.setText("Log Y")
        self.log_x_action.setToolTip("Toggle logarithmic X axis")
        self.log_y_action.setToolTip("Toggle logarithmic Y axis")
        self.log_x_action.setVisible(True)
        self.log_y_action.setVisible(True)
        plot_toolbar.addAction(self.log_x_action)
        plot_toolbar.addAction(self.log_y_action)
        self.log_x_action.toggled.connect(self._axis_scale_changed)
        self.log_y_action.toggled.connect(self._axis_scale_changed)

        visibility_dock = qt.QDockWidget("Curves", self)
        visibility_dock.setObjectName("multipleAsciiCurvesDock")
        visibility_dock.setFeatures(
            qt.QDockWidget.DockWidgetMovable | qt.QDockWidget.DockWidgetFloatable
        )
        curve_list = qt.QListWidget(visibility_dock)
        curve_list.setAlternatingRowColors(True)
        curve_list.setIconSize(qt.QSize(48, 16))
        visibility_dock.setWidget(curve_list)
        self.addDockWidget(qt.Qt.RightDockWidgetArea, visibility_dock)
        self.curve_list = curve_list

        for legend, x_values, y_values in curves:
            self._curve_data[legend] = (
                np.asarray(x_values), np.asarray(y_values)
            )
            curve = self.plot.addCurve(
                x_values, y_values, legend=legend, resetzoom=False
            )
            item = qt.QListWidgetItem(legend)
            item.setIcon(self._curve_style_icon(curve))
            item.setToolTip(
                f"{legend}\nColor: {curve.getColor()}\n"
                f"Line style: {curve.getLineStyle()}"
            )
            item.setFlags(item.flags() | qt.Qt.ItemIsUserCheckable)
            item.setCheckState(qt.Qt.Checked)
            curve_list.addItem(item)
            self._curve_items[legend] = item
        curve_list.itemChanged.connect(self._curve_visibility_changed)
        self.plot.resetZoom()

    @staticmethod
    def _curve_style_icon(curve):
        """Render a curve's color and line style for the right-side list."""
        color = curve.getColor()
        if isinstance(color, str):
            qcolor = qt.QColor(color)
        else:
            values = np.asarray(color, dtype=float).ravel()
            if values.size >= 3 and np.nanmax(values[:3]) <= 1.0:
                alpha = values[3] if values.size > 3 else 1.0
                qcolor = qt.QColor.fromRgbF(
                    values[0], values[1], values[2], alpha
                )
            else:
                alpha = int(values[3]) if values.size > 3 else 255
                qcolor = qt.QColor(
                    int(values[0]), int(values[1]), int(values[2]), alpha
                )

        pixmap = qt.QPixmap(48, 16)
        pixmap.fill(qt.Qt.transparent)
        painter = qt.QPainter(pixmap)
        pen = qt.QPen(qcolor)
        pen.setWidth(max(2, int(round(curve.getLineWidth()))))
        pen.setStyle({
            "--": qt.Qt.DashLine,
            "-.": qt.Qt.DashDotLine,
            ":": qt.Qt.DotLine,
        }.get(curve.getLineStyle(), qt.Qt.SolidLine))
        painter.setPen(pen)
        painter.drawLine(3, 8, 45, 8)
        painter.end()
        return qt.QIcon(pixmap)

    @qt.Slot(bool)
    def _axis_scale_changed(self, _checked=False):
        """Apply the visible toolbar actions to the actual plot axes."""
        self.plot.getXAxis().setScale(
            "log" if self.log_x_action.isChecked() else "linear"
        )
        self.plot.getYAxis().setScale(
            "log" if self.log_y_action.isChecked() else "linear"
        )
        self.plot.resetZoom()

    @qt.Slot(qt.QListWidgetItem)
    def _curve_visibility_changed(self, item):
        curve = self.plot.getCurve(item.text())
        if curve is not None:
            curve.setVisible(item.checkState() == qt.Qt.Checked)
            self.plot.replot()


class StatusPanelProxy:
    """Route legacy status-label writes into the unified Status panel."""

    def __init__(self, update_callback):
        self._update_callback = update_callback
        self._text = ""

    def setText(self, text):
        self._text = str(text)
        self._update_callback("operation", "Status", self._text)

    def text(self):
        return self._text


class MainWindow(qt.QMainWindow):
    def __init__(self):
        super().__init__()
        qt.QLocale.setDefault(qt.QLocale("en_US"))
        self.setWindowTitle("pyFAI Integrate Viewer v3")
        self.resize(1900, 850)
        self._settings = qt.QSettings(
            "pyFAI Integrate Viewer", "pyFAI Integrate Viewer"
        )
        saved_export_path = self._settings.value("exportPath", "", type=str)
        self.export_path = (
            saved_export_path
            if saved_export_path and Path(saved_export_path).is_dir()
            else str(Path.home())
        )
        saved_input_path = self._settings.value("inputPath", "", type=str)
        self.input_path = (
            saved_input_path
            if saved_input_path and Path(saved_input_path).is_dir()
            else str(Path.home())
        )
        self.image_data: np.ndarray | None = None
        self.image_paths: list[ImageSource] = []
        self.image_index = -1
        self._beamline = None
        self._measurement_mode = None
        self.mask_data: np.ndarray | None = None
        self.empty_data: np.ndarray | None = None
        self.background_data: np.ndarray | None = None
        self._reference_sources = {"empty": [], "background": []}
        self.detector_mask: np.ndarray | None = None
        self._effective_mask_cache = None
        self._effective_mask_crc = None
        self._effective_mask_cache_valid = False
        self.integrator = None
        self._poni_wavelength = None
        self._thread = None
        self._worker = None
        self._load_thread = None
        self._load_worker = None
        self._operation_cancel_requested = False
        self._integration_worker_payload = None
        self._integration_worker_error = None
        self._mask_generation = 0
        self._mask_jobs = []
        self._detector_sum_generation = 0
        self._detector_sum_jobs = []
        self._detector_sum_restart_pending = False
        self._detector_sum_valid = False
        self._roi_sum_valid = False
        self._sum_calculation_mode = None
        self._detector_sum_values = np.empty(0, dtype=np.int64)
        self._detector_sum_by_source = {}
        self._sum_job_sources = []
        self._roi_sum_values = np.empty(0, dtype=np.int64)
        self._roi_bounds = None
        self._detector_mask_cache = {}
        self._image_readers = {}
        self._auto_integrate_images = False
        self._pending_auto_integration = False
        self._video_writer = None
        self._video_path = ""
        self._video_original_index = -1
        self._video_original_auto = False
        self._exporting_video = False
        self._batch_video_queue = []
        self._batch_video_original_paths = None
        self._batch_video_original_index = -1
        self._batch_video_original_auto = False
        self._video_progress_key = ""
        self._progress_blocks = {}
        self._fixed_status_blocks = {}
        self._batch_thread = None
        self._batch_worker = None
        self._batch_worker_result = None
        self._batch_worker_error = None
        self._combined_export_video_dir = None
        self._combined_export_plot_path = None
        self._combined_export_plot_queue = []
        self._combined_export_video_groups = []
        self._plot_export_original_paths = None
        self._plot_export_original_index = -1
        self._plot_export_original_auto = False
        self._plot_export_current_filename = ""
        self._exporting_plot = False
        self._batch_plot_only = False
        self._batch_plot_output_dir = ""
        self._batch_plot_total = 0
        self._batch_plot_current = 0
        self._last_integration_payload = None
        self._cake_cache = {}
        self._cake_log_mesh = None
        self._cake_colormap_signal_source = None
        self._reference_curve_cache = {}
        self._syncing_integration_x = False
        self._result_y_autoscale_pending = False
        self._external_plot_windows = []
        self._asaxs_windows = []
        self._asaxs_module = None
        self._build_ui()
        self._build_menu_bar()

    def _build_menu_bar(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")
        path_menu = file_menu.addMenu("Path")
        self.input_path_action = path_menu.addAction("Input Path...")
        self.export_path_action = path_menu.addAction("Export Path...")
        self.input_path_action.triggered.connect(self.choose_input_path)
        self.export_path_action.triggered.connect(self.choose_export_path)
        beamline_menu = file_menu.addMenu("Beamline")
        self.p62_action = beamline_menu.addAction("p62")
        self.p62_action.setCheckable(True)
        self.p62_action.toggled.connect(self._set_p62_enabled)
        file_menu.addSeparator()
        save_menu = file_menu.addMenu("Save")
        self.save_current_action = save_menu.addAction("Plot (current view)")
        self.save_batch_plots_action = save_menu.addAction("Batch Plots")
        self.save_batch_video_action = save_menu.addAction("Batch Video")
        self.save_data_action = save_menu.addAction("ASCII (all integrated data)")
        self.save_ascii_video_action = save_menu.addAction("Data and Plots")
        self.save_current_action.triggered.connect(self.save_current_view)
        self.save_batch_plots_action.triggered.connect(self.save_batch_plots)
        self.save_batch_video_action.triggered.connect(self.save_batch_videos)
        self.save_data_action.triggered.connect(self.save_all_integrated_data)
        self.save_ascii_video_action.triggered.connect(self.save_data_and_plots)
        options_menu = menu_bar.addMenu("Options")
        self.plot_nexus_action = options_menu.addAction("Plot NeXus...")
        self.plot_ascii_action = options_menu.addAction("Plot ASCII...")
        self.asaxs_action = options_menu.addAction("ASAXS...")
        self.plot_nexus_action.triggered.connect(self.plot_nexus_data)
        self.plot_ascii_action.triggered.connect(self.plot_ascii_data)
        self.asaxs_action.triggered.connect(self.open_asaxs_window)
        # Keep the view selector in the menu-bar row so it does not push the
        # integration toolbar below the Source image toolbar.
        menu_bar.setCornerWidget(self.right_tab_bar, qt.Qt.TopRightCorner)

    def _show_external_curve(self, x, y, x_label, y_label, filename):
        if x.shape != y.shape:
            raise ValueError(
                f"X and Y lengths differ ({x.size} and {y.size})."
            )
        window = ExternalCurveWindow(
            x, y, x_label, y_label, Path(filename).name, self,
        )
        self._external_plot_windows.append(window)
        window.destroyed.connect(
            lambda _=None, window=window: self._remove_external_plot(window)
        )
        window.show()

    def _remove_external_plot(self, window):
        if window in self._external_plot_windows:
            self._external_plot_windows.remove(window)

    @qt.Slot()
    def open_asaxs_window(self):
        """Open the PyAnomScat Stuhrmann GUI as a child application window."""
        try:
            if not PYANOMSCAT_SCRIPT.is_file():
                raise FileNotFoundError(f"ASAXS GUI not found: {PYANOMSCAT_SCRIPT}")
            if self._asaxs_module is None:
                module_name = "integrated_pyanomscat_stuhrmann_v3"
                spec = importlib.util.spec_from_file_location(
                    module_name, PYANOMSCAT_SCRIPT
                )
                if spec is None or spec.loader is None:
                    raise ImportError(f"Cannot load {PYANOMSCAT_SCRIPT}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                self._asaxs_module = module
            previous_directory = os.getcwd()
            try:
                os.chdir(PYANOMSCAT_SCRIPT.parent)
                window = self._asaxs_module.MainWindow(parent=self)
            finally:
                os.chdir(previous_directory)
            window.appImportPath = self.input_path
            if hasattr(window, "statusbar"):
                window.statusbar.showMessage(window.appImportPath)
            window.setAttribute(qt.Qt.WA_DeleteOnClose)
            self._asaxs_windows.append(window)
            window.destroyed.connect(
                lambda _=None, window=window: self._remove_asaxs_window(window)
            )
            window.show()
        except Exception as error:
            self.show_error("Cannot Open ASAXS", traceback.format_exc())

    def _remove_asaxs_window(self, window):
        if window in self._asaxs_windows:
            self._asaxs_windows.remove(window)

    def _show_external_curves(self, curves, x_label, y_label):
        window = MultiAsciiCurveWindow(curves, x_label, y_label, self)
        self._external_plot_windows.append(window)
        window.destroyed.connect(
            lambda _=None, window=window: self._remove_external_plot(window)
        )
        window.show()

    @qt.Slot()
    def plot_nexus_data(self):
        filename, _ = qt.QFileDialog.getOpenFileName(
            self, "Select NeXus File", self.input_path,
            "NeXus/HDF5 files (*.nxs *.h5 *.hdf5);;All files (*)",
        )
        if not filename:
            return
        self._set_input_path(str(Path(filename).parent))
        dialog = NexusDatasetDialog(filename, self)
        if dialog.exec() != qt.QDialog.Accepted:
            return
        try:
            x_selection, y_selection, x_label, y_label = dialog.selection()
            self._show_external_curve(
                x_selection[2], y_selection[2], x_label, y_label, filename,
            )
        except Exception as error:
            qt.QMessageBox.critical(self, "Cannot Plot NeXus Data", str(error))

    @qt.Slot()
    def plot_ascii_data(self):
        self.plot_multiple_ascii_data()

    @qt.Slot()
    def plot_multiple_ascii_data(self):
        filenames, _ = qt.QFileDialog.getOpenFileNames(
            self, "Select One or More ASCII Files", self.input_path,
            "ASCII data (*.dat *.txt *.csv *.asc);;All files (*)",
        )
        if not filenames:
            return
        self._set_input_path(str(Path(filenames[0]).parent))
        try:
            tables = []
            for filename in filenames:
                data, names = read_ascii_columns(filename)
                tables.append((filename, data, names))
            first_data, first_names = tables[0][1], tables[0][2]
            choices = [
                f"Column {index + 1}: {first_names[index]}" if first_names
                else f"Column {index + 1}"
                for index in range(first_data.shape[1])
            ]
            dialog = XYSelectionDialog(
                "Select X and Y Columns for All ASCII Files", choices, self
            )
            if dialog.exec() != qt.QDialog.Accepted:
                return
            x_index, y_index, x_label, y_label = dialog.selection()
            curves = []
            used_legends = {}
            for filename, data, _names in tables:
                if max(x_index, y_index) >= data.shape[1]:
                    raise ValueError(
                        f"{Path(filename).name} has only {data.shape[1]} columns; "
                        f"the selected columns are unavailable."
                    )
                base_legend = Path(filename).name
                used_legends[base_legend] = used_legends.get(base_legend, 0) + 1
                count = used_legends[base_legend]
                legend = base_legend if count == 1 else f"{base_legend} ({count})"
                curves.append((legend, data[:, x_index], data[:, y_index]))
            self._show_external_curves(curves, x_label, y_label)
        except Exception as error:
            qt.QMessageBox.critical(
                self, "Cannot Plot Multiple ASCII Data", str(error)
            )

    def _set_save_actions_enabled(self, enabled):
        for action in (
            self.save_current_action,
            self.save_batch_plots_action,
            self.save_batch_video_action,
            self.save_data_action,
            self.save_ascii_video_action,
        ):
            action.setEnabled(enabled)

    @qt.Slot()
    def choose_input_path(self):
        """Choose the default directory for Image/Empty/Background dialogs."""
        directory = qt.QFileDialog.getExistingDirectory(
            self, "Select Default Input Path", self.input_path
        )
        if directory:
            self._set_input_path(directory)
            self.status_label.setText(f"Input path: {self.input_path}")

    def _set_input_path(self, directory):
        directory = str(Path(directory).resolve())
        if Path(directory).is_dir():
            self.input_path = directory
            self._settings.setValue("inputPath", directory)

    @qt.Slot()
    def choose_export_path(self):
        """Choose and persist the initial directory for every export dialog."""
        directory = qt.QFileDialog.getExistingDirectory(
            self, "Select Default Export Path", self.export_path
        )
        if directory:
            self._set_export_path(directory)
            self.status_label.setText(f"Export path: {self.export_path}")

    def _set_export_path(self, directory):
        directory = str(Path(directory).resolve())
        if Path(directory).is_dir():
            self.export_path = directory
            self._settings.setValue("exportPath", directory)

    def _build_ui(self):
        root = qt.QWidget(self)
        self.setCentralWidget(root)
        layout = qt.QHBoxLayout(root)

        controls = qt.QWidget()
        form_box = qt.QVBoxLayout(controls)
        controls.setMaximumWidth(380)

        self.p62_mode_widget = qt.QWidget()
        p62_mode_layout = qt.QHBoxLayout(self.p62_mode_widget)
        p62_mode_layout.setContentsMargins(0, 0, 0, 0)
        self.p62_mode_group = qt.QButtonGroup(self)
        self.p62_mode_group.setExclusive(True)
        for mode in ("saxs", "waxs", "asaxs", "awaxs"):
            button = qt.QPushButton(mode)
            button.setCheckable(True)
            button.clicked.connect(
                lambda checked=False, selected=mode: self._select_p62_mode(selected)
            )
            self.p62_mode_group.addButton(button)
            p62_mode_layout.addWidget(button)
        self.p62_mode_widget.setVisible(False)
        form_box.addWidget(self.p62_mode_widget)

        files = qt.QGroupBox("Input Files")
        grid = qt.QGridLayout(files)
        self.image_edit = qt.QLineEdit()
        self.poni_edit = qt.QLineEdit()
        self.mask_edit = qt.QLineEdit()
        self.empty_edit = qt.QLineEdit()
        self.background_edit = qt.QLineEdit()
        for row, (name, edit, slot) in enumerate(
             (("Image", self.image_edit, self.choose_image),
             ("PONI", self.poni_edit, self.choose_poni),
             ("Mask", self.mask_edit, self.choose_mask))
        ):
            button = qt.QPushButton("Browse…")
            button.clicked.connect(slot)
            grid.addWidget(qt.QLabel(name), row, 0)
            grid.addWidget(edit, row, 1)
            grid.addWidget(button, row, 2)
        clear_mask = qt.QPushButton("Clear Mask")
        clear_mask.clicked.connect(self.clear_mask)
        grid.addWidget(clear_mask, 3, 2)
        navigation = qt.QWidget()
        navigation_layout = qt.QHBoxLayout(navigation)
        navigation_layout.setContentsMargins(0, 0, 0, 0)
        self.previous_image_button = qt.QPushButton("< Previous")
        self.next_image_button = qt.QPushButton("Next >")
        self.previous_image_button.clicked.connect(self.show_previous_image)
        self.next_image_button.clicked.connect(self.show_next_image)
        navigation_layout.addWidget(self.previous_image_button)
        navigation_layout.addWidget(self.next_image_button)
        grid.addWidget(navigation, 4, 1, 1, 2)
        self._update_navigation_buttons()
        form_box.addWidget(files)

        detector_box = qt.QGroupBox("Detector Information")
        detector_layout = qt.QVBoxLayout(detector_box)
        self.detector_label = qt.QLabel("Load a PONI file to read detector geometry")
        self.detector_label.setWordWrap(True)
        detector_layout.addWidget(self.detector_label)
        form_box.addWidget(detector_box)

        options = qt.QGroupBox("Integration Parameters")
        form = qt.QFormLayout(options)
        self.points_spin = qt.QSpinBox()
        self.points_spin.setRange(10, 100000)
        self.points_spin.setValue(1000)
        self.unit_combo = qt.QComboBox()
        self.unit_combo.addItems(["q_A^-1", "q_nm^-1", "2th_deg", "2th_rad", "r_mm"])
        self.error_model_combo = qt.QComboBox()
        self.error_model_combo.addItem("No error", None)
        self.error_model_combo.addItem("Poisson", "poisson")
        self.error_model_combo.addItem("Azimuthal", "azimuthal")
        self.error_model_combo.addItem("Hybrid", "hybrid")
        self.error_model_combo.setToolTip(
            "pyFAI error model. When enabled, integrated ASCII files contain "
            "a third column named Sigma."
        )
        self.radial_range_edit = qt.QLineEdit()
        self.azimuth_range_edit = qt.QLineEdit()
        for edit in (self.radial_range_edit, self.azimuth_range_edit):
            edit.setPlaceholderText("Auto")
            edit.setToolTip("minimum, maximum")
        form.addRow("Number of points", self.points_spin)
        form.addRow("X-axis unit", self.unit_combo)
        form.addRow("Error model", self.error_model_combo)
        form.addRow("Radial range", self.radial_range_edit)
        form.addRow("Azimuthal range (deg)", self.azimuth_range_edit)
        self.cake_check = qt.QCheckBox("Cake")
        self.cake_check.setToolTip("Calculate and display the 2-D cake integration")
        form.addRow(self.cake_check)
        self.show_empty_check = qt.QCheckBox("Show")
        self.subtract_empty_check = qt.QCheckBox("Subtract")
        self.empty_factor = qt.QDoubleSpinBox()
        self.show_background_check = qt.QCheckBox("Show")
        self.subtract_background_check = qt.QCheckBox("Subtract")
        self.background_factor = qt.QDoubleSpinBox()
        self.show_empty_check.toggled.connect(self._show_reference_changed)
        self.show_background_check.toggled.connect(self._show_reference_changed)
        for factor in (self.empty_factor, self.background_factor):
            factor.setRange(-1e6, 1e6)
            factor.setDecimals(2)
            factor.setSingleStep(0.1)
            factor.setLocale(qt.QLocale("en_US"))
            factor.setValue(1.0)
        form.addRow(
            "Empty", self._reference_file_controls(
                self.empty_edit, self.choose_empty, self.clear_empty
            )
        )
        form.addRow(
            self._reference_controls(
                self.show_empty_check, self.subtract_empty_check, self.empty_factor
            )
        )
        form.addRow(
            "Background", self._reference_file_controls(
                self.background_edit, self.choose_background,
                self.clear_background
            )
        )
        form.addRow(
            self._reference_controls(
                self.show_background_check, self.subtract_background_check,
                self.background_factor
            )
        )
        self.update_references_button = qt.QPushButton("Update")
        self.update_references_button.clicked.connect(self.update_reference_integration)
        form.addRow(self.update_references_button)
        form_box.addWidget(options)

        operation_buttons = qt.QWidget()
        operation_layout = qt.QHBoxLayout(operation_buttons)
        operation_layout.setContentsMargins(0, 0, 0, 0)
        self.integrate_button = qt.QPushButton("Integrate")
        self.integrate_button.setMinimumHeight(42)
        self.integrate_button.setEnabled(False)
        self.integrate_button.clicked.connect(self.start_integration)
        self.stop_button = qt.QPushButton("Stop")
        self.stop_button.setMinimumHeight(42)
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_current_operation)
        operation_layout.addWidget(self.integrate_button, 1)
        operation_layout.addWidget(self.stop_button)
        form_box.addWidget(operation_buttons)
        progress_box = qt.QGroupBox("Status")
        progress_layout = qt.QVBoxLayout(progress_box)
        self.status_summary = qt.QPlainTextEdit()
        self.status_summary.setReadOnly(True)
        self.status_summary.setMinimumHeight(150)
        self.status_summary.setMaximumHeight(190)
        self.status_summary.setVerticalScrollBarPolicy(qt.Qt.ScrollBarAlwaysOff)
        self.status_summary.setHorizontalScrollBarPolicy(qt.Qt.ScrollBarAlwaysOff)
        self.export_progress = qt.QPlainTextEdit()
        self.export_progress.setReadOnly(True)
        self.export_progress.setPlaceholderText(
            "Loading, integration, and export status appears here"
        )
        self.export_progress.setMinimumHeight(65)
        self.export_progress.setMaximumHeight(90)
        self.export_progress.setMaximumBlockCount(1000)
        progress_layout.addWidget(self.status_summary)
        progress_layout.addWidget(self.export_progress, 1)
        form_box.addWidget(progress_box)
        self.status_label = StatusPanelProxy(self._update_export_progress_block)
        self.status_label.setText(
            "Select a 2-D diffraction image and a PONI file"
        )
        form_box.addStretch(1)

        splitter = qt.QSplitter(qt.Qt.Horizontal)
        self.plot_splitter = splitter
        self.image_plot = Plot2D()
        self.image_plot.setGraphTitle("No image loaded")
        self.image_plot.setKeepDataAspectRatio(True)
        self.image_plot.setDataMargins(0.0, 0.0, 0.0, 0.0)
        self.image_plot.setDefaultColormap(
            Colormap(name="viridis", normalization=Colormap.LOGARITHM)
        )
        self.roi_manager = RegionOfInterestManager(self.image_plot)
        self.roi_action = self.roi_manager.getInteractionModeAction(RectangleROI)
        self.roi_action.setText("Select ROI")
        self.roi_action.setToolTip(
            "Draw a rectangular ROI for frame-by-frame intensity sums"
        )
        self.image_plot.toolBar().addAction(self.roi_action)
        self.roi_manager.sigRoiAdded.connect(self._roi_added)
        if hasattr(self.roi_manager, "sigRoiRemoved"):
            self.roi_manager.sigRoiRemoved.connect(self._roi_removed)
        source_position = self.image_plot.getPositionInfoWidget()
        if source_position is not None and len(source_position._fields) >= 3:
            value_label, _name, _converter = source_position._fields[2]
            source_position._fields[2] = (
                value_label, "Intensity", self._source_intensity_at
            )
            source_position.layout().itemAt(4).widget().setText(
                "<b>Intensity:</b>"
            )
        source_colorbar = self.image_plot.getColorBarWidget()
        source_colorbar.setMinimumWidth(70)
        source_colorbar.setMaximumWidth(70)
        if source_colorbar.layout() is not None:
            source_colorbar.layout().setContentsMargins(2, 4, 2, 4)
        self.source_sum_label = qt.QLabel("Detector sum intensity: no image loaded")
        self.source_sum_label.setAlignment(qt.Qt.AlignCenter)
        self.source_sum_label.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)
        self.result_plot = Plot1D()
        self.result_plot.setGraphTitle("1D integration")
        self.result_plot.setGraphYLabel("Intensity")
        self.cake_plot = Plot2D()
        self.cake_plot.setGraphTitle("Cake integration")
        self.cake_plot.setGraphYLabel("Azimuthal angle (°)")
        self.cake_plot.setDefaultColormap(
            Colormap(name="viridis", normalization=Colormap.LOGARITHM)
        )
        self.cake_plot.getColorBarWidget().setVisible(False)
        self.cake_plot.getColorBarAction().setVisible(False)
        # Use identical plot-area margins so both X axes line up exactly.
        axes_margins = (0.15, 0.10, 0.05, 0.15)
        self.cake_plot.setAxesMargins(*axes_margins)
        self.result_plot.setAxesMargins(*axes_margins)
        self.result_log_x_action = self.result_plot.getXAxisLogarithmicAction()
        self.result_log_y_action = self.result_plot.getYAxisLogarithmicAction()
        self.cake_log_x_action = self.cake_plot.getXAxisLogarithmicAction()
        self.cake_log_y_action = self.cake_plot.getYAxisLogarithmicAction()
        self.result_log_x_action.setVisible(True)
        self.result_log_y_action.setVisible(True)
        self.cake_log_x_action.setVisible(True)
        self.cake_log_y_action.setVisible(False)
        self.cake_log_x_action.toggled.connect(self._shared_log_x_changed)
        self.result_log_x_action.toggled.connect(self._shared_log_x_changed)
        self.result_plot.getYAxis().sigScaleChanged.connect(
            self._result_y_scale_changed
        )
        self.result_plot.getXAxis().sigLimitsChanged.connect(
            self._result_x_limits_changed
        )
        self.cake_plot.getXAxis().sigLimitsChanged.connect(
            self._cake_x_limits_changed
        )
        result_toolbar = self.result_plot.toolBar()
        cake_toolbar = self.cake_plot.toolBar()
        # Detach the plot toolbars and expose one shared toolbar above both
        # plots, like pyFAI-calib2. X-limit changes are synchronized below.
        self.result_plot.removeToolBar(result_toolbar)
        self.cake_plot.removeToolBar(cake_toolbar)
        # Plot1D/Plot2D create several independent QToolBars (interactive mode,
        # output, options and profile tools). Merge the complete Plot1D set into
        # the shared bar so Qt cannot wrap them onto a second toolbar row.
        existing_actions = set(result_toolbar.actions())
        interactive_toolbar = self.result_plot.getInteractiveModeToolBar()
        first_action = result_toolbar.actions()[0] if result_toolbar.actions() else None
        interactive_actions = [
            action for action in interactive_toolbar.actions() if action.isVisible()
        ]
        if first_action is not None:
            for action in interactive_actions:
                result_toolbar.insertAction(first_action, action)
                existing_actions.add(action)
            interactive_separator = result_toolbar.insertSeparator(first_action)
        else:
            result_toolbar.addActions(interactive_actions)
            existing_actions.update(interactive_actions)
            interactive_separator = result_toolbar.addSeparator()
        for toolbar in self.result_plot.findChildren(qt.QToolBar):
            if toolbar is result_toolbar:
                continue
            visible_actions = [action for action in toolbar.actions() if action.isVisible()]
            if visible_actions:
                result_toolbar.addSeparator()
            for action in visible_actions:
                if action not in existing_actions:
                    result_toolbar.addAction(action)
                    existing_actions.add(action)
            self.result_plot.removeToolBar(toolbar)
            toolbar.setVisible(False)
        # Use Cake's native silx colormap dialog, identical to Source image,
        # from the single toolbar shared by the integration plots. Keep it
        # immediately after Zoom/Pan so Qt does not move it into the overflow
        # menu when the toolbar is narrower than all available actions.
        self.cake_colormap_action = self.cake_plot.getColormapAction()
        result_toolbar.insertAction(
            interactive_separator, self.cake_colormap_action
        )
        self.cake_colormap_action.setVisible(False)
        # The native silx CopyAction renders through an in-memory PNG path that
        # can terminate Qt on Windows (0xC0000409). Replace it with a direct,
        # ownership-safe QImage copy of the same visible plots used by exports.
        native_copy_action = self.result_plot.getOutputToolBar().getCopyAction()
        toolbar_actions = result_toolbar.actions()
        copy_index = toolbar_actions.index(native_copy_action)
        next_action = (
            toolbar_actions[copy_index + 1]
            if copy_index + 1 < len(toolbar_actions) else None
        )
        result_toolbar.removeAction(native_copy_action)
        native_copy_action.setShortcut(qt.QKeySequence())
        native_copy_action.setEnabled(False)
        self.copy_visible_plots_action = qt.QAction(
            native_copy_action.icon(), "Copy visible plots", self
        )
        self.copy_visible_plots_action.setToolTip(
            "Copy the currently visible plot canvases to the clipboard"
        )
        self.copy_visible_plots_action.setShortcut(qt.QKeySequence.Copy)
        self.copy_visible_plots_action.setShortcutContext(
            qt.Qt.WidgetWithChildrenShortcut
        )
        self.copy_visible_plots_action.triggered.connect(self.copy_visible_plots)
        if next_action is None:
            result_toolbar.addAction(self.copy_visible_plots_action)
        else:
            result_toolbar.insertAction(next_action, self.copy_visible_plots_action)
        # A profile ROI only applies to a 2-D image. Expose Cake's profile tools
        # in the shared toolbar while Cake is enabled; Plot1D is already itself
        # a one-dimensional profile and has no equivalent ProfileToolBar.
        cake_profile_toolbar = self.cake_plot.profile
        self.cake_profile_actions = [
            action for action in cake_profile_toolbar.actions()
            if action.isVisible()
            and action.text().strip().casefold() != "clear profile"
        ]
        self.cake_profile_separator = None
        if self.cake_profile_actions:
            self.cake_profile_separator = result_toolbar.addSeparator()
            self.cake_profile_separator.setVisible(False)
            for action in self.cake_profile_actions:
                result_toolbar.addAction(action)
                action.setVisible(False)
        # Cake is operated by the same shared controls; hide every toolbar it
        # created internally so none can appear between Cake and 1-D.
        for toolbar in self.cake_plot.findChildren(qt.QToolBar):
            self.cake_plot.removeToolBar(toolbar)
            toolbar.setVisible(False)
        result_toolbar.setMovable(False)
        result_toolbar.setFloatable(False)
        result_toolbar.setVisible(True)
        # All useful actions are placed directly on this toolbar. Qt still
        # creates an overflow button for hidden actions/separators, yielding an
        # empty "..." menu, so remove that non-functional button.
        toolbar_extension = result_toolbar.findChild(
            qt.QToolButton, "qt_toolbar_ext_button"
        )
        if toolbar_extension is not None:
            toolbar_extension.setFixedSize(0, 0)
            toolbar_extension.setVisible(False)
        cake_toolbar.setVisible(False)
        self.result_plot.resetZoomAction.triggered.connect(
            self._reset_integration_plots
        )
        integration_panel = qt.QSplitter(qt.Qt.Vertical)
        integration_panel.addWidget(self.cake_plot)
        integration_panel.addWidget(self.result_plot)
        integration_panel.setStretchFactor(0, 1)
        integration_panel.setStretchFactor(1, 1)
        integration_panel.setSizes([400, 400])
        self.integration_splitter = integration_panel
        self.cake_plot.setVisible(False)
        self.cake_check.toggled.connect(self._cake_toggled)
        integration_container = qt.QWidget()
        integration_layout = qt.QVBoxLayout(integration_container)
        integration_layout.setContentsMargins(0, 0, 0, 0)
        integration_layout.setSpacing(0)
        integration_layout.addWidget(result_toolbar)
        integration_layout.addWidget(integration_panel, 1)
        self._build_shared_position_bar(integration_layout)
        self.cake_toolbar = cake_toolbar
        self.result_toolbar = result_toolbar
        source_container = qt.QWidget()
        source_layout = qt.QVBoxLayout(source_container)
        source_layout.setContentsMargins(0, 0, 0, 0)
        source_layout.setSpacing(2)
        source_layout.addWidget(self.image_plot, 1)
        source_layout.addWidget(self.source_sum_label)
        self.source_container = source_container

        self.detector_sum_plot = Plot1D()
        self.detector_sum_plot.setGraphTitle("Detector sum intensity")
        self.detector_sum_plot.setGraphXLabel("Image / frame index")
        self.detector_sum_plot.setGraphYLabel("Detector sum intensity")
        detector_sum_container = qt.QWidget()
        self.detector_sum_container = detector_sum_container
        detector_sum_layout = qt.QVBoxLayout(detector_sum_container)
        detector_sum_layout.setContentsMargins(0, 0, 0, 0)
        self.detector_sum_calculate_button = qt.QPushButton("Calculate")
        self.detector_sum_calculate_button.setEnabled(False)
        self.detector_sum_calculate_button.clicked.connect(
            lambda: self._start_detector_sum_update("detector")
        )
        detector_sum_layout.addWidget(self.detector_sum_calculate_button)
        detector_sum_layout.addWidget(self.detector_sum_plot, 1)
        self.roi_sum_plot = Plot1D()
        self.roi_sum_plot.setGraphTitle("ROI sum intensity")
        self.roi_sum_plot.setGraphXLabel("Image / frame index")
        self.roi_sum_plot.setGraphYLabel("ROI sum intensity")
        roi_sum_container = qt.QWidget()
        self.roi_sum_container = roi_sum_container
        roi_sum_layout = qt.QVBoxLayout(roi_sum_container)
        roi_sum_layout.setContentsMargins(0, 0, 0, 0)
        self.roi_sum_calculate_button = qt.QPushButton("Calculate")
        self.roi_sum_calculate_button.setEnabled(False)
        self.roi_sum_calculate_button.clicked.connect(
            lambda: self._start_detector_sum_update("roi")
        )
        roi_sum_layout.addWidget(self.roi_sum_calculate_button)
        roi_sum_layout.addWidget(self.roi_sum_plot, 1)
        self.right_stack = qt.QStackedWidget()
        self.right_stack.addWidget(integration_container)
        self.right_stack.addWidget(detector_sum_container)
        self.right_stack.addWidget(roi_sum_container)
        self.right_tab_bar = qt.QTabBar()
        self.right_tab_bar.addTab("Integration")
        self.right_tab_bar.addTab("Detector Sum")
        self.right_tab_bar.addTab("ROI Sum")
        self.right_tab_bar.setExpanding(False)
        self.right_tab_bar.setDrawBase(False)
        self.right_tab_bar.setDocumentMode(True)
        self.right_tab_bar.currentChanged.connect(
            self.right_stack.setCurrentIndex
        )

        splitter.addWidget(source_container)
        splitter.addWidget(self.right_stack)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([800, 500])
        layout.addWidget(controls)
        layout.addWidget(splitter, 1)

        qt.QTimer.singleShot(0, self._update_plot_sizes)

    def _build_shared_position_bar(self, parent_layout):
        """Place Cake and 1-D cursor information in one shared bottom row."""
        cake_info = self.cake_plot.getPositionInfoWidget()
        result_info = self.result_plot.getPositionInfoWidget()
        shared_bar = qt.QWidget()
        shared_layout = qt.QHBoxLayout(shared_bar)
        shared_layout.setContentsMargins(6, 1, 6, 1)
        shared_layout.setSpacing(8)
        shared_layout.addWidget(cake_info)
        shared_layout.addWidget(result_info)
        parent_layout.addWidget(shared_bar)

        # Cake Plot2D fields are X, Y, Data and Dims. Keep X as the single
        # shared radial coordinate and give the remaining fields explicit names.
        cake_layout = cake_info.layout()
        cake_layout.itemAt(0).widget().setText("<b>X:</b>")
        cake_layout.itemAt(2).widget().setText("<b>Cake Y:</b>")
        cake_layout.itemAt(4).widget().setText("<b>Cake Data:</b>")
        # Plot1D has X and Y: hide its duplicate X and name Y unambiguously.
        result_layout = result_info.layout()
        result_layout.itemAt(0).widget().setVisible(False)
        result_layout.itemAt(1).widget().setVisible(False)
        result_layout.itemAt(2).widget().setText("<b>1D Intensity:</b>")

        self.shared_position_bar = shared_bar
        self._cake_position_widgets = [
            cake_layout.itemAt(index).widget() for index in range(2, 8)
            if cake_layout.itemAt(index) is not None
            and cake_layout.itemAt(index).widget() is not None
        ]
        self._shared_x_value_label = cake_info._fields[0][0]
        self._shared_1d_value_label = result_info._fields[1][0]
        self.result_plot.sigPlotSignal.connect(self._update_shared_x_from_1d)
        self.cake_plot.sigPlotSignal.connect(self._update_shared_1d_from_cake)
        for widget in self._cake_position_widgets:
            widget.setVisible(False)

    @qt.Slot(object)
    def _update_shared_x_from_1d(self, event):
        """Update the shared radial X field while the cursor is over 1-D."""
        if event.get("event") == "mouseMoved":
            self._shared_x_value_label.setText(f"{event['x']:.7g}")

    @qt.Slot(object)
    def _update_shared_1d_from_cake(self, event):
        """Show the 1-D intensity at the Cake cursor's shared radial X."""
        if event.get("event") != "mouseMoved" or self._last_integration_payload is None:
            return
        radial = self._last_integration_payload.get("radial")
        intensity = self._last_integration_payload.get("sample")
        if radial is None or intensity is None or len(radial) == 0:
            return
        value = np.interp(event["x"], radial, intensity, left=np.nan, right=np.nan)
        self._shared_1d_value_label.setText(
            "------" if not np.isfinite(value) else f"{value:.7g}"
        )

    def _reference_controls(self, show, subtract, factor):
        widget = qt.QWidget()
        layout = qt.QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(show)
        layout.addWidget(subtract)
        layout.addWidget(qt.QLabel("Factor"))
        layout.addWidget(factor)
        return widget

    @qt.Slot(bool)
    def _cake_toggled(self, checked):
        """Show Cake only when requested and calculate it on the next integration."""
        self.cake_plot.setVisible(checked)
        self.cake_colormap_action.setVisible(checked)
        self.result_log_y_action.setVisible(True)
        self.cake_log_x_action.setVisible(True)
        for widget in self._cake_position_widgets:
            widget.setVisible(checked)
        for action in self.cake_profile_actions:
            action.setVisible(checked)
        if self.cake_profile_separator is not None:
            self.cake_profile_separator.setVisible(checked)
        if checked:
            self.cake_log_x_action.setChecked(
                self.result_plot.getXAxis().getScale() == "log"
            )
            self.integration_splitter.setSizes([400, 400])
        else:
            self.integration_splitter.setSizes([0, 800])

    @qt.Slot(bool)
    def _shared_log_x_changed(self, checked):
        """Apply the Cake toolbar's Log X action to both integration plots."""
        scale = "log" if checked else "linear"
        self.cake_plot.getXAxis().setScale(scale)
        self.result_plot.getXAxis().setScale(scale)
        # An image whose left extent is exactly zero is not drawable on a log
        # axis. Re-add Cake with only its non-positive edge columns omitted.
        # This changes display only; the cached integration remains untouched.
        if (self.cake_check.isChecked()
                and self._last_integration_payload is not None
                and self._last_integration_payload.get("cake") is not None):
            self._render_cake_image(
                self._last_integration_payload,
                resetzoom=False,
            )
        if checked:
            positive_x = []
            for curve in self.result_plot.getAllCurves():
                values = np.asarray(curve.getXData(copy=False))
                valid = values[np.isfinite(values) & (values > 0)]
                if valid.size:
                    positive_x.append(valid)
            if positive_x:
                minimum = min(float(values.min()) for values in positive_x)
                maximum = max(float(values.max()) for values in positive_x)
                self.result_plot.getXAxis().setLimits(minimum, maximum)
        self.result_plot.replot()
        self.cake_plot.replot()

    @qt.Slot(str)
    def _result_y_scale_changed(self, _scale):
        """Autoscale 1-D Y after Linear/Log/Arcsinh changes, preserving X."""
        if self._result_y_autoscale_pending:
            return
        self._result_y_autoscale_pending = True
        qt.QTimer.singleShot(0, self._autoscale_result_y)

    @qt.Slot()
    def _autoscale_result_y(self):
        """Autoscale both axes after silx has fully applied the Y scale."""
        self._result_y_autoscale_pending = False
        x_axis = self.result_plot.getXAxis()
        y_axis = self.result_plot.getYAxis()
        x_axis.setAutoScale(True)
        y_axis.setAutoScale(True)
        self.result_plot.resetZoom()
        # Explicit limits make this reliable with the matplotlib backend, whose
        # scale-change action can otherwise restore the previous linear limits.
        scale = y_axis.getScale()
        y_min = np.inf
        y_max = -np.inf
        for curve in self.result_plot.getAllCurves():
            if not curve.isVisible():
                continue
            values = np.asarray(curve.getYData(copy=False))
            valid = np.isfinite(values)
            if scale == "log":
                valid &= values > 0
            if np.any(valid):
                y_min = min(y_min, float(np.min(values[valid])))
                y_max = max(y_max, float(np.max(values[valid])))
        if np.isfinite(y_min) and np.isfinite(y_max):
            if scale == "log":
                if y_min == y_max:
                    y_min, y_max = y_min / 10.0, y_max * 10.0
                else:
                    margin = (y_max / y_min) ** 0.05
                    y_min, y_max = y_min / margin, y_max * margin
            else:
                span = y_max - y_min
                margin = 0.05 * span if span > 0 else max(abs(y_min) * 0.05, 1.0)
                y_min, y_max = y_min - margin, y_max + margin
            y_axis.setLimits(y_min, y_max)
        self.result_plot.replot()

    @qt.Slot()
    def _reset_integration_plots(self):
        """Reset both plots from the single shared integration toolbar."""
        if self.cake_check.isChecked():
            self.cake_plot.resetZoom()

    @qt.Slot(float, float)
    def _result_x_limits_changed(self, minimum, maximum):
        if self._syncing_integration_x or not self.cake_check.isChecked():
            return
        self._syncing_integration_x = True
        try:
            self.cake_plot.getXAxis().setLimits(minimum, maximum)
        finally:
            self._syncing_integration_x = False

    @qt.Slot(float, float)
    def _cake_x_limits_changed(self, minimum, maximum):
        if self._syncing_integration_x or not self.cake_check.isChecked():
            return
        self._syncing_integration_x = True
        try:
            self.result_plot.getXAxis().setLimits(minimum, maximum)
        finally:
            self._syncing_integration_x = False

    def _reference_file_controls(self, edit, browse_slot, clear_slot):
        widget = qt.QWidget()
        layout = qt.QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(edit, 1)
        browse_button = qt.QPushButton("Browse...")
        browse_button.clicked.connect(browse_slot)
        layout.addWidget(browse_button)
        clear_button = qt.QPushButton("Clear")
        clear_button.clicked.connect(clear_slot)
        layout.addWidget(clear_button)
        return widget

    @qt.Slot()
    def update_reference_integration(self):
        """Recalculate sample and selected Empty/Background reference curves."""
        if self._thread is not None:
            self.status_label.setText("Integration is already running; try Update again when complete")
            return
        required = {
            name for name, data, show, subtract in (
                ("Empty", self.empty_data, self.show_empty_check.isChecked(),
                 self.subtract_empty_check.isChecked()),
                ("Background", self.background_data,
                 self.show_background_check.isChecked(),
                 self.subtract_background_check.isChecked()),
            ) if data is not None and (show or subtract)
        }
        cached = (
            set() if self._last_integration_payload is None
            else set(self._last_integration_payload["references"])
        )
        try:
            ranges = self._integration_ranges()
            current_key = self._cake_cache_key(*ranges)
        except ValueError:
            current_key = None
        cache_context_matches = (
            self._last_integration_payload is not None
            and self._last_integration_payload.get("cake_cache_key") == current_key
        )
        if cache_context_matches and required <= cached:
            payload = self._last_integration_payload
            corrected = payload["sample"].copy()
            for name, values in list(payload["references"].items()):
                intensity = values[0]
                if name == "Empty":
                    show = self.show_empty_check.isChecked()
                    subtract = self.subtract_empty_check.isChecked()
                    factor = self.empty_factor.value()
                else:
                    show = self.show_background_check.isChecked()
                    subtract = self.subtract_background_check.isChecked()
                    factor = self.background_factor.value()
                payload["references"][name] = (
                    intensity, show, subtract, factor
                )
                if subtract:
                    corrected -= factor * intensity
            payload["corrected"] = corrected
            self._render_integration_payload(payload)
            self.result_plot.getXAxis().setAutoScale(True)
            self.result_plot.getYAxis().setAutoScale(True)
            self.result_plot.resetZoom()
            self.status_label.setText("Reference factors and subtraction updated")
            return
        self.start_integration()

    @qt.Slot(bool)
    def _show_reference_changed(self, checked):
        """Show/hide cached references immediately; integrate once if not cached."""
        required = []
        if self.show_empty_check.isChecked():
            required.append("Empty")
        if self.show_background_check.isChecked():
            required.append("Background")
        cached = (
            set() if self._last_integration_payload is None
            else set(self._last_integration_payload["references"])
        )
        try:
            current_key = self._cake_cache_key(*self._integration_ranges())
        except ValueError:
            current_key = None
        if (
            self._last_integration_payload is None
            or self._last_integration_payload.get("cake_cache_key") != current_key
        ):
            cached.clear()
        if all(name in cached for name in required):
            if self._last_integration_payload is not None:
                x_limits = self.result_plot.getXAxis().getLimits()
                y_limits = self.result_plot.getYAxis().getLimits()
                self._render_integration_payload(self._last_integration_payload)
                if checked:
                    self.result_plot.getXAxis().setAutoScale(True)
                    self.result_plot.getYAxis().setAutoScale(True)
                    self.result_plot.resetZoom()
                else:
                    self.result_plot.getXAxis().setLimits(*x_limits)
                    self.result_plot.getYAxis().setLimits(*y_limits)
            return
        if (
            self._thread is None and self.image_data is not None
            and self.integrator is not None and os.path.isfile(self.poni_edit.text().strip())
        ):
            self.start_integration()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        qt.QTimer.singleShot(0, self._update_plot_sizes)

    def closeEvent(self, event):
        # Avoid "QThread: Destroyed while thread is still running" crashes.
        running = [
            thread for thread in (
                self._thread,
                self._load_thread,
                self._batch_thread,
                *(job[0] for job in self._mask_jobs),
                *(job[0] for job in self._detector_sum_jobs),
            )
            if thread is not None and thread.isRunning()
        ]
        if running:
            self.status_label.setText(
                "Please wait for the active integration task before closing"
            )
            event.ignore()
            return
        for reader in self._image_readers.values():
            reader.close()
        self._image_readers.clear()
        super().closeEvent(event)

    def _update_plot_sizes(self):
        """Keep the source-image widget square; the result uses remaining width."""
        if not hasattr(self, "plot_splitter"):
            return
        plot_height = self.plot_splitter.contentsRect().height()
        if plot_height <= 0:
            return
        available_width = self.plot_splitter.contentsRect().width()
        handle_width = self.plot_splitter.handleWidth()
        target_result_width = 700
        result_width = min(
            target_result_width, max(100, available_width - handle_width - 100)
        )
        source_width = max(100, available_width - result_width - handle_width)
        self.source_container.setMinimumWidth(source_width)
        self.source_container.setMaximumWidth(source_width)
        self.plot_splitter.setSizes([source_width, result_width])

    def _pick(self, title, file_filter, use_input_path=False):
        initial_path = self.input_path if use_input_path else ""
        filename, _ = qt.QFileDialog.getOpenFileName(
            self, title, initial_path, file_filter
        )
        if filename and use_input_path:
            self._set_input_path(Path(filename).parent)
        return filename

    def _update_navigation_buttons(self):
        count = len(self.image_paths)
        idle = not self._has_active_operation()
        self.previous_image_button.setEnabled(idle and count > 1 and self.image_index > 0)
        self.next_image_button.setEnabled(idle and count > 1 and self.image_index < count - 1)
        if hasattr(self, "integrate_button"):
            can_integrate = (
                idle and self.image_data is not None and self.integrator is not None
            )
            self.integrate_button.setEnabled(can_integrate)
        if hasattr(self, "stop_button"):
            self.stop_button.setEnabled(not idle)
        if hasattr(self, "detector_sum_calculate_button"):
            sums_available = idle and bool(self.image_paths)
            self.detector_sum_calculate_button.setEnabled(sums_available)
            self.roi_sum_calculate_button.setEnabled(sums_available)
        if hasattr(self, "save_current_action"):
            self._set_save_actions_enabled(idle)

    def _has_active_operation(self):
        return bool(
            self._load_thread is not None
            or self._thread is not None
            or self._batch_thread is not None
            or self._exporting_video
            or self._exporting_plot
            or self._mask_jobs
            or self._detector_sum_jobs
        )

    @qt.Slot()
    def stop_current_operation(self):
        """Request cancellation for every active loading, calculation, or export."""
        if not self._has_active_operation():
            return
        self._operation_cancel_requested = True
        self._pending_auto_integration = False
        self._auto_integrate_images = False
        self._detector_sum_restart_pending = False
        self._combined_export_video_dir = None
        self._combined_export_plot_path = None
        self._combined_export_plot_queue.clear()
        self._combined_export_video_groups.clear()
        self._batch_video_queue.clear()
        for worker in (
            self._load_worker,
            self._worker,
            self._batch_worker,
            *(job[1] for job in self._mask_jobs),
            *(job[1] for job in self._detector_sum_jobs),
        ):
            if worker is not None and hasattr(worker, "cancel"):
                worker.cancel()
        self.status_label.setText("Stopping current operation…")
        self._update_export_progress_block(
            "operation", "Operation", "stopping…"
        )
        self._update_navigation_buttons()
        if self._exporting_video:
            self._finish_video_export(False, cancelled=True)
            return
        if self._exporting_plot:
            self._exporting_plot = False
            self._batch_plot_only = False
            self._restore_plot_export_state()

    def _operation_stopped(self):
        if self._has_active_operation():
            return
        self._operation_cancel_requested = False
        self.status_label.setText("Operation stopped")
        self._update_export_progress_block("operation", "Operation", "stopped")
        self._set_save_actions_enabled(True)
        self._update_navigation_buttons()

    def _image_position_text(self):
        if not self.image_paths or self.image_index < 0:
            return ""
        return f"Image {self.image_index + 1} of {len(self.image_paths)}"

    def _subtraction_reference_text(self, sample_index=None, sample_sources=None):
        """Describe the exact Empty/Background sources used for subtraction."""
        sources = self.image_paths if sample_sources is None else sample_sources
        index = self.image_index if sample_index is None else sample_index
        if not (0 <= index < len(sources)):
            return ""
        parts = []
        sample_source = sources[index]
        for kind in ("empty", "background"):
            subtract = getattr(self, f"subtract_{kind}_check").isChecked()
            references = self._reference_sources[kind]
            if not subtract or not references:
                continue
            try:
                reference = matching_reference_source(
                    references, sample_source, sources, index
                )
            except ValueError as error:
                parts.append(f"{kind.title()}: {error}")
            else:
                parts.append(f"{kind.title()} subtraction: {reference.title}")
        return "; ".join(parts)

    @staticmethod
    def _source_file_frame_progress(sources, index):
        """Return global file progress plus frame progress for one source."""
        if not (0 <= index < len(sources)):
            return ""
        source = sources[index]
        source_path = source.path if isinstance(source, ImageSource) else str(source)
        grouped = group_batch_video_sources(sources)
        source_group = next(
            (ordered for _prefix, ordered in grouped if source in ordered),
            list(sources),
        )
        ordered_files = list(dict.fromkeys(
            item.path if isinstance(item, ImageSource) else str(item)
            for item in source_group
        ))
        file_index = ordered_files.index(source_path) + 1
        text = f"file [{file_index}/{len(ordered_files)}]"
        if isinstance(source, ImageSource) and source.frame is not None:
            text += f"; frame [{source.frame + 1}/{source.frame_count}]"
        else:
            text += "; frame [1/1]"
        return text

    def _video_saving_position(self, source):
        """Return position across every selected image/frame, not one video group."""
        all_sources = (
            self._batch_video_original_paths
            if self._batch_video_original_paths is not None
            else self.image_paths
        )
        try:
            current = all_sources.index(source) + 1
        except ValueError:
            current = self.image_index + 1
        return current, len(all_sources)

    def _current_source_key(self):
        if not (0 <= self.image_index < len(self.image_paths)):
            return None
        source = self.image_paths[self.image_index]
        return (os.path.abspath(source.path), source.frame)

    def _cake_cache_key(self, radial_range, azimuth_range):
        """Key Cake by every input that can change its numerical result."""
        poni = self.poni_edit.text().strip()
        try:
            poni_mtime = os.path.getmtime(poni)
        except OSError:
            poni_mtime = None
        mask = self._effective_mask()
        mask_summary = (
            None if mask is None
            else (tuple(mask.shape), self._effective_mask_crc)
        )
        return (
            self._current_source_key(), os.path.abspath(poni), poni_mtime,
            None if self.integrator is None else self.integrator.wavelength,
            self.points_spin.value(), self.unit_combo.currentText(),
            radial_range, azimuth_range, mask_summary,
        )

    def _reference_cache_key(self, name, radial_range, azimuth_range):
        kind = name.casefold()
        ref_source = matching_reference_source(
            self._reference_sources[kind], self.image_paths[self.image_index],
            self.image_paths, self.image_index,
        )
        try:
            reference_mtime = os.path.getmtime(ref_source.path)
        except OSError:
            reference_mtime = None
        cake_context = self._cake_cache_key(radial_range, azimuth_range)
        # Drop the source-image identity: a reference curve is reusable across
        # all samples with the same geometry, shape, mask and integration setup.
        image_shape = None if self.image_data is None else tuple(self.image_data.shape)
        return (
            name, os.path.abspath(ref_source.path), ref_source.frame,
            reference_mtime, image_shape, self._error_model()
        ) + cake_context[1:]

    def _reset_export_progress(self):
        self._progress_blocks.clear()
        self.export_progress.clear()
        self._update_subtraction_status_block()

    def _append_export_progress(self, text):
        self.export_progress.appendPlainText(text)
        cursor = self.export_progress.textCursor()
        cursor.movePosition(qt.QTextCursor.End)
        self.export_progress.setTextCursor(cursor)
        self.export_progress.ensureCursorVisible()
        scroll_bar = self.export_progress.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())

    def _update_export_progress_block(self, key, label, progress):
        """Add a filename once, then update only its following progress line."""
        if key == "operation" and self._exporting_video:
            return
        if key in ("operation", "subtraction"):
            if key == "operation":
                label = "Status"
            self._fixed_status_blocks[key] = f"{label}:\n    {progress}"
            self.status_summary.setPlainText(
                "\n".join(self._fixed_status_blocks.values())
            )
            return
        self._progress_blocks[key] = (
            f"{label}:\n    {progress}" if progress else str(label)
        )
        self.export_progress.setPlainText("\n".join(self._progress_blocks.values()))
        scroll_bar = self.export_progress.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())

    def _update_subtraction_status_block(self, sample_index=None):
        """Show matched subtraction files inside the existing Status panel."""
        text = self._subtraction_reference_text(sample_index)
        if text:
            self._fixed_status_blocks["subtraction"] = (
                f"Subtraction:\n    {text}"
            )
        else:
            self._fixed_status_blocks.pop("subtraction", None)
        self.status_summary.setPlainText(
            "\n".join(self._fixed_status_blocks.values())
        )

    def _update_video_status(self, progress):
        """Update fixed video progress without allowing transient frame states."""
        self._fixed_status_blocks["operation"] = f"Status:\n    {progress}"
        self.status_summary.setPlainText(
            "\n".join(self._fixed_status_blocks.values())
        )

    def _update_scrolling_filename(self, source, prefix="file"):
        """Add only a source filename to the scrolling Status history."""
        path = source.path if isinstance(source, ImageSource) else str(source)
        key = f"{prefix}:{os.path.abspath(path)}"
        self._progress_blocks[key] = Path(path).name
        self.export_progress.setPlainText("\n".join(self._progress_blocks.values()))
        scroll_bar = self.export_progress.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())

    def _clear_incompatible_calibration(self):
        """Clear geometry and masks when an image does not fit the detector."""
        self._mask_generation += 1  # Ignore results from an older mask job
        self.integrator = None
        self._poni_wavelength = None
        self.detector_mask = None
        self._effective_mask_cache_valid = False
        self._detector_mask_cache.clear()
        self._cake_cache.clear()
        self._reference_curve_cache.clear()
        self.mask_data = None
        self.poni_edit.clear()
        self.mask_edit.clear()
        self._show_detector_information()
        self._auto_integrate_images = False
        self._pending_auto_integration = False
        self._invalidate_detector_sum()

    def _update_image_plot(self, resetzoom=True):
        """Display a memory-efficient preview while retaining full data for pyFAI."""
        if self.image_data is None:
            self.image_plot.clear()
            return 1

        height, width = self.image_data.shape
        step = max(1, int(np.ceil(max(height, width) / MAX_DISPLAY_DIMENSION)))
        preview = self.image_data[::step, ::step]
        effective_mask = self._effective_mask()
        preview_mask = None if effective_mask is None else effective_mask[::step, ::step]
        # The active source image is also used by silx line profiles. Replace
        # masked detector intensities with zero while retaining a separate pink
        # overlay; keep self.image_data unchanged for pyFAI integration.
        display_preview = (
            preview if preview_mask is None
            else np.where(preview_mask, 0, preview)
        )

        valid = np.isfinite(preview)
        if preview_mask is not None:
            valid &= ~preview_mask
        # Logarithmic intensity normalization uses only strictly positive data
        # to determine its display limits. Masked pixels remain zero.
        values = preview[valid & (preview > 0)]
        if values.size:
            vmin, vmax = np.percentile(values, (1.0, 99.5))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
                vmin, vmax = float(np.min(values)), float(np.max(values))
            if vmin >= vmax:
                vmax = vmin + 1.0
        else:
            vmin, vmax = 1.0, 10.0

        image_colormap = Colormap(
            name="viridis", normalization=Colormap.LOGARITHM,
            vmin=float(vmin), vmax=float(vmax)
        )

        self.image_plot.clear()
        self.image_plot.setGraphTitle(
            self.image_paths[self.image_index].title
            if 0 <= self.image_index < len(self.image_paths)
            else "No image loaded"
        )
        self.image_plot.addImage(
            display_preview,
            legend="detector image",
            scale=(step, step),
            colormap=image_colormap,
            resetzoom=resetzoom,
        )
        if preview_mask is not None and np.any(preview_mask):
            mask_colormap = Colormap(
                name=None,
                colors=((0.0, 0.0, 0.0, 0.0), (1.0, 0.4, 1.0, 0.8)),
                vmin=0,
                vmax=1,
            )
            self.image_plot.addImage(
                preview_mask.astype(np.uint8), legend="mask overlay",
                scale=(step, step), colormap=mask_colormap, z=10,
                resetzoom=False,
            )
        self.image_plot.setActiveImage("detector image")
        return step

    def _effective_mask(self):
        if self._effective_mask_cache_valid:
            return self._effective_mask_cache
        masks = [mask for mask in (self.detector_mask, self.mask_data) if mask is not None]
        if not masks:
            combined = None
        elif len(masks) == 1:
            combined = masks[0]
        else:
            combined = np.logical_or(masks[0], masks[1])
        self._effective_mask_cache = combined
        self._effective_mask_crc = mask_checksum(combined)
        self._effective_mask_cache_valid = True
        return combined

    def _source_intensity_at(self, x, y):
        """Return full-resolution detector intensity below the source cursor."""
        if self.image_data is None or not np.isfinite(x) or not np.isfinite(y):
            return "-"
        column = int(np.floor(x))
        row = int(np.floor(y))
        if not (
            0 <= row < self.image_data.shape[0]
            and 0 <= column < self.image_data.shape[1]
        ):
            return "-"
        effective_mask = self._effective_mask()
        if effective_mask is not None and bool(effective_mask[row, column]):
            return 0
        return self.image_data[row, column]

    def _update_source_sum_label(self):
        """Display only a cached Calculate result; never sum during Integrate."""
        if self.image_data is None:
            self.source_sum_label.setText(
                "Detector sum intensity: no image loaded"
            )
            return
        value = self._detector_sum_by_source.get(self._current_source_key())
        if value is None:
            self.source_sum_label.setText(
                "Detector sum intensity: not calculated"
            )
            return
        self.source_sum_label.setText(
            f"Detector sum intensity (calculated): {int(value):,d}"
        )

    def _roi_added(self, roi):
        """Keep one rectangle and cache its full-resolution pixel bounds."""
        if not isinstance(roi, RectangleROI):
            return
        for other in list(self.roi_manager.getRois()):
            if other is not roi:
                self.roi_manager.removeRoi(other)
        roi.sigRegionChanged.connect(self._roi_geometry_changed)
        self._roi_geometry_changed()

    def _roi_removed(self, _roi):
        if not self.roi_manager.getRois():
            self._roi_bounds = None
            self._invalidate_detector_sum(roi_only=True)

    @qt.Slot()
    def _roi_geometry_changed(self):
        rois = self.roi_manager.getRois()
        if not rois or self.image_data is None:
            self._roi_bounds = None
        else:
            origin = rois[-1].getOrigin()
            size = rois[-1].getSize()
            left = max(0, int(np.floor(origin[0])))
            top = max(0, int(np.floor(origin[1])))
            right = min(
                self.image_data.shape[1], int(np.ceil(origin[0] + size[0]))
            )
            bottom = min(
                self.image_data.shape[0], int(np.ceil(origin[1] + size[1]))
            )
            self._roi_bounds = (
                (left, top, right, bottom)
                if right > left and bottom > top else None
            )
        self._invalidate_detector_sum(roi_only=True)

    def _start_detector_sum_update(self, mode="detector"):
        """Calculate detector or ROI sums for every selected image/frame."""
        if mode == "roi" and self._roi_bounds is None:
            self.show_error(
                "ROI Required",
                "Select a rectangular ROI on the detector image before calculating ROI Sum.",
            )
            return
        if not self.image_paths:
            return
        if self._detector_sum_jobs:
            return
        self._detector_sum_generation += 1
        generation = self._detector_sum_generation
        self._sum_calculation_mode = mode
        self._sum_job_sources = list(self.image_paths)
        # Fabio/HDF5 readers are not reliably thread-safe on Windows. Release
        # GUI-owned persistent readers before the worker opens the same files,
        # and disable navigation until that worker has finished.
        for reader in self._image_readers.values():
            reader.close()
        self._image_readers.clear()
        poni = self.poni_edit.text().strip() if self.integrator is not None else ""
        thread = qt.QThread(self)
        worker = DetectorSumWorker(
            list(self.image_paths), poni, self.mask_data, generation,
            self._roi_bounds if mode == "roi" else None, mode,
            self._detector_mask_cache,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._detector_sum_progress)
        worker.finished.connect(self._detector_sum_finished)
        worker.failed.connect(self._detector_sum_failed)
        worker.cancelled.connect(thread.quit)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        worker.cancelled.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        job = (thread, worker)
        self._detector_sum_jobs.append(job)
        thread.finished.connect(
            lambda job=job: self._remove_detector_sum_job(job)
        )
        self._update_navigation_buttons()
        self.detector_sum_calculate_button.setEnabled(False)
        self.roi_sum_calculate_button.setEnabled(False)
        thread.start()

    def _invalidate_detector_sum(self, clear=False, roi_only=False):
        """Mark cached sums stale, preserving curves unless images are replaced."""
        if not roi_only:
            self._detector_sum_valid = False
        self._roi_sum_valid = False
        self._detector_sum_restart_pending = False
        if clear:
            self._detector_sum_generation += 1
            self._detector_sum_values = np.empty(0, dtype=np.int64)
            self._detector_sum_by_source.clear()
            self._sum_job_sources = []
            self._roi_sum_values = np.empty(0, dtype=np.int64)
            self.detector_sum_plot.clear()
            self.roi_sum_plot.clear()

    def _remove_detector_sum_job(self, job):
        if job in self._detector_sum_jobs:
            self._detector_sum_jobs.remove(job)
        self._update_navigation_buttons()
        if not self._detector_sum_jobs:
            self.detector_sum_calculate_button.setEnabled(True)
            self.roi_sum_calculate_button.setEnabled(True)
        if self._operation_cancel_requested:
            self._operation_stopped()

    @qt.Slot(int, object, object, int)
    def _detector_sum_progress(self, index, value, roi_value, generation):
        if generation != self._detector_sum_generation:
            return
        label = "ROI sum" if self._sum_calculation_mode == "roi" else "detector sum"
        self.status_label.setText(
            f"Calculating {label}: {index + 1}/{len(self.image_paths)}"
        )
        self._update_export_progress_block(
            "operation", f"Calculating {label}",
            f"frame [{index + 1}/{len(self.image_paths)}]",
        )

    @qt.Slot(object, int)
    def _detector_sum_finished(self, result, generation):
        if generation != self._detector_sum_generation:
            return
        mode = result["mode"]
        values = np.asarray(result["values"], dtype=np.int64)
        x = np.arange(1, values.size + 1, dtype=np.float64)
        if mode == "detector":
            self._detector_sum_values = values
            self._detector_sum_by_source = {
                (os.path.abspath(source.path), source.frame): int(value)
                for source, value in zip(self._sum_job_sources, values)
            }
            self._detector_sum_valid = True
            plot = self.detector_sum_plot
            legend = "Detector sum intensity"
        else:
            self._roi_sum_values = values
            self._roi_sum_valid = True
            plot = self.roi_sum_plot
            legend = "ROI sum intensity"
        plot.clear()
        if values.size:
            plot.addCurve(
                x, values, legend=legend, symbol="o", resetzoom=True,
            )
        plot.setGraphXLabel("Image / frame index")
        plot.setGraphYLabel(legend)
        self.status_label.setText(
            f"{legend} calculation complete: {values.size} frames"
        )
        self._update_export_progress_block(
            "operation", legend, f"complete [{values.size}/{values.size}]"
        )
        self._update_source_sum_label()

    @qt.Slot(str, int)
    def _detector_sum_failed(self, details, generation):
        if generation != self._detector_sum_generation:
            return
        self._update_source_sum_label()
        label = "ROI Sum" if self._sum_calculation_mode == "roi" else "Detector Sum"
        self.status_label.setText(
            f"{label} calculation failed; see error dialog"
        )
        self.show_error(f"{label} Failed", details)

    def _start_detector_mask_update(self):
        """Calculate pyFAI's dynamic detector mask without blocking the GUI."""
        self._mask_generation += 1
        generation = self._mask_generation
        self.detector_mask = None
        self._effective_mask_cache_valid = False
        if self.integrator is None or self.image_data is None:
            self._update_source_sum_label()
            return
        # Detector-mask preparation is independent of Detector Sum. Preserve an
        # existing cached reading (or "not calculated") while the mask is built.
        self._update_source_sum_label()
        shape_key = tuple(self.image_data.shape)
        detector_mask_is_static = getattr(self.integrator.detector, "dummy", None) is None
        if detector_mask_is_static and shape_key in self._detector_mask_cache:
            self._detector_mask_finished(
                self._detector_mask_cache[shape_key], generation, ""
            )
            return
        if self._exporting_video:
            try:
                mask = detector_mask_for_image(
                    self.integrator.detector, self.image_data
                )
                if mask is not None:
                    mask = np.asarray(mask, dtype=bool)
                    if mask.shape != self.image_data.shape:
                        mask = None
                error = ""
            except Exception:
                mask = None
                error = traceback.format_exc()
            self._detector_mask_finished(mask, generation, error)
            return
        thread = qt.QThread(self)
        worker = DetectorMaskWorker(self.integrator.detector, self.image_data, generation)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._detector_mask_finished)
        worker.cancelled.connect(thread.quit)
        worker.finished.connect(thread.quit)
        worker.cancelled.connect(worker.deleteLater)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        job = (thread, worker)
        self._mask_jobs.append(job)
        thread.finished.connect(
            lambda job=job: qt.QTimer.singleShot(
                50, lambda job=job: self._remove_mask_job(job)
            )
        )
        self.integrate_button.setEnabled(False)
        self.status_label.setText("PONI loaded; calculating detector mask in background...")
        thread.start()

    def _remove_mask_job(self, job):
        if job in self._mask_jobs:
            self._mask_jobs.remove(job)
        self._update_navigation_buttons()
        if self._operation_cancel_requested:
            self._operation_stopped()

    @qt.Slot(object, int, str)
    def _detector_mask_finished(self, mask, generation, error):
        if generation != self._mask_generation:
            return
        self.detector_mask = mask
        self._effective_mask_cache_valid = False
        if (
            mask is not None and self.image_data is not None
            and self.integrator is not None
            and getattr(self.integrator.detector, "dummy", None) is None
        ):
            self._detector_mask_cache[tuple(self.image_data.shape)] = mask
        self.integrate_button.setEnabled(True)
        if self.image_data is not None:
            self._update_image_plot(resetzoom=False)
            self._update_source_sum_label()
        count = 0 if mask is None else np.count_nonzero(mask)
        if error:
            self.status_label.setText(
                f"{self._image_position_text()}; detector mask calculation failed "
                "(integration is still available)"
            )
        else:
            self.status_label.setText(
                f"{self._image_position_text()}; detector mask excludes {count:,} pixels"
            )
        self._update_navigation_buttons()
        if self._pending_auto_integration and self.integrator is not None:
            self._pending_auto_integration = False
            self.start_integration()

    def _show_detector_information(self):
        if self.integrator is None:
            self.detector_label.setText("Load a PONI file to read detector geometry")
            return
        detector = self.integrator.detector
        name = getattr(detector, "name", detector.__class__.__name__)
        shape = getattr(detector, "shape", None) or getattr(detector, "max_shape", None)
        pixel1 = getattr(detector, "pixel1", None)
        pixel2 = getattr(detector, "pixel2", None)
        lines = [f"Detector: {name}    Shape: {shape or 'unknown'}"]
        if pixel1 is not None and pixel2 is not None:
            geometry_line = (
                f"Pixel: {pixel2 * 1e6:.3f} x {pixel1 * 1e6:.3f} um    "
                f"Distance: {self.integrator.dist * 1e3:.3f} mm"
            )
        else:
            geometry_line = f"Distance: {self.integrator.dist * 1e3:.3f} mm"
        lines.append(geometry_line)
        energy_ev = self._current_energy_ev()
        if energy_ev is not None:
            lines.append(
                f"Energy: {energy_ev} eV    "
                f"Wavelength: {wavelength_from_energy(energy_ev):.12g} m"
            )
        if self.integrator.wavelength is not None:
            if energy_ev is None:
                lines.append(f"Wavelength: {self.integrator.wavelength * 1e10:.6g} A")
        self.detector_label.setText("\n".join(lines))

    @qt.Slot(bool)
    def _set_p62_enabled(self, enabled):
        self._beamline = "p62" if enabled else None
        self.p62_mode_widget.setVisible(enabled)
        if not enabled:
            self._measurement_mode = None
            self._apply_current_energy()

    def _select_p62_mode(self, mode):
        self._measurement_mode = mode
        if mode in ("saxs", "asaxs"):
            # SAXS curves span several decades on both axes. Match the usual
            # beamline view as soon as either SAXS mode is selected.
            self.result_log_x_action.setChecked(True)
            self.result_log_y_action.setChecked(True)
        self.status_label.setText(f"p62 mode: {mode.upper()}")
        self._apply_current_energy()

    def _current_energy_ev(self):
        if (
            self._beamline != "p62"
            or self._measurement_mode not in ("asaxs", "awaxs")
        ):
            return None
        if not (0 <= self.image_index < len(self.image_paths)):
            return None
        return self.image_paths[self.image_index].energy_ev

    def _apply_current_energy(self):
        if self.integrator is None:
            return
        energy_ev = self._current_energy_ev()
        if energy_ev is not None:
            self.integrator.wavelength = wavelength_from_energy(energy_ev)
            self._cake_cache.clear()
            self._reference_curve_cache.clear()
        else:
            self.integrator.wavelength = self._poni_wavelength
        self._show_detector_information()

    @qt.Slot()
    def choose_image(self):
        if self._has_active_operation():
            return
        p62_mode = self._measurement_mode if self._beamline == "p62" else None
        if self._beamline == "p62" and p62_mode is None:
            self.show_error(
                "p62 Mode Required",
                "Select saxs, waxs, asaxs, or awaxs before loading images.",
            )
            return
        dataset_path = {
            "saxs": "/scan/data/saxs_raw",
            "asaxs": "/scan/data/saxs_raw",
            "waxs": "/scan/data/waxs_raw",
            "awaxs": "/scan/data/waxs_raw",
        }.get(p62_mode)
        filenames, _ = qt.QFileDialog.getOpenFileNames(
            self, "Select 2-D Diffraction Images", self.input_path,
            P62_IMAGE_FILTER if p62_mode else IMAGE_FILTER,
        )
        if not filenames:
            return
        if p62_mode:
            try:
                invalid = [name for name in filenames if Path(name).suffix.lower() != ".nxs"]
                if invalid:
                    raise ValueError(f"p62 {p62_mode.upper()} image loading accepts only .nxs files")
            except Exception as exc:
                self.show_error(f"Invalid p62 {p62_mode.upper()} NeXus File", str(exc))
                return
        self._set_input_path(Path(filenames[0]).parent)
        for reader in self._image_readers.values():
            reader.close()
        self._image_readers.clear()
        self._operation_cancel_requested = False
        self._load_thread = qt.QThread(self)
        self._load_worker = ImageLoadWorker(
            filenames, dataset_path, include_energy=bool(p62_mode)
        )
        self._load_worker.moveToThread(self._load_thread)
        self._load_thread.started.connect(self._load_worker.run)
        self._load_worker.progress.connect(self._image_load_progress)
        self._load_worker.finished.connect(self._image_load_finished)
        self._load_worker.failed.connect(self._image_load_failed)
        self._load_worker.cancelled.connect(self._load_thread.quit)
        self._load_worker.finished.connect(self._load_thread.quit)
        self._load_worker.failed.connect(self._load_thread.quit)
        self._load_worker.finished.connect(self._load_worker.deleteLater)
        self._load_worker.failed.connect(self._load_worker.deleteLater)
        self._load_worker.cancelled.connect(self._load_worker.deleteLater)
        self._load_thread.finished.connect(self._image_load_thread_finished)
        self._load_thread.finished.connect(self._load_thread.deleteLater)
        self.status_label.setText("Loading selected images…")
        self._update_export_progress_block(
            "operation", "Loading images", f"starting [0/{len(filenames)}]"
        )
        self._update_navigation_buttons()
        self._load_thread.start()

    @qt.Slot(int, int, str)
    def _image_load_progress(self, current, total, filename):
        self.status_label.setText(f"Loading {current} of {total}: {filename}")
        self._update_export_progress_block(
            "operation", "Loading images", f"file [{current}/{total}] {filename}"
        )

    @qt.Slot(object)
    def _image_load_finished(self, sources):
        if self._operation_cancel_requested:
            return
        self.image_paths = sources
        self._cake_cache.clear()
        self._invalidate_detector_sum(clear=True)
        self.image_index = 0
        self._load_current_image()
        self._update_export_progress_block(
            "operation", "Loading images", f"complete ({len(sources)} images/frames)"
        )

    @qt.Slot(str)
    def _image_load_failed(self, details):
        if not self._operation_cancel_requested:
            self.show_error("Unable to Read Image Frames", details)
            self.status_label.setText("Loading images failed")
            self._update_export_progress_block(
                "operation", "Loading images", "failed"
            )

    @qt.Slot()
    def _image_load_thread_finished(self):
        self._load_worker = None
        self._load_thread = None
        if self._operation_cancel_requested:
            self._operation_stopped()
        else:
            self._update_navigation_buttons()

    def _load_current_image(self):
        if not (0 <= self.image_index < len(self.image_paths)):
            return
        source = self.image_paths[self.image_index]
        try:
            # Keep only the current detector array in memory. read_image opens
            # and closes the file (or HDF5 frame) within this call.
            data = read_image(source)
            require_integer_detector_image(data, source.title)
        except Exception as exc:
            self.show_error("Unable to Read Image", str(exc))
            self._update_navigation_buttons()
            return
        self.image_data = data
        self._apply_current_energy()
        self.image_edit.setText(source.title)
        references_cleared = False
        for kind in ("empty", "background"):
            reference_sources = self._reference_sources[kind]
            # A single-frame reference is loaded once and reused unchanged for
            # every data frame. Only multi-frame references need re-reading.
            if len(reference_sources) > 1:
                try:
                    reference_source = matching_reference_source(
                        reference_sources, source, self.image_paths,
                        self.image_index,
                    )
                    setattr(self, f"{kind}_data", read_image(reference_source))
                except Exception as exc:
                    self._clear_reference(kind, update_plot=False)
                    references_cleared = True
                    self.show_error(
                        f"Invalid {kind.title()} Frames", str(exc)
                    )
                    continue
            reference = getattr(self, f"{kind}_data")
            if reference is not None and (
                reference.shape != data.shape or reference.dtype != data.dtype
            ):
                self._clear_reference(kind, update_plot=False)
                references_cleared = True
        calibration_cleared = False
        if self.integrator is not None:
            detector = self.integrator.detector
            detector_matches = detector_accepts_shape(detector, data.shape)
            if detector_matches:
                try:
                    detector_matches = bool(detector.guess_binning(data))
                except (ValueError, TypeError, AttributeError):
                    detector_matches = False
            if not detector_matches:
                self._clear_incompatible_calibration()
                calibration_cleared = True
        if not calibration_cleared and self.mask_data is not None and self.mask_data.shape != data.shape:
            self.mask_data = None
            self._effective_mask_cache_valid = False
            self.mask_edit.clear()
            self._cake_cache.clear()
            self._reference_curve_cache.clear()
        step = self._update_image_plot()
        if self.result_plot.getAllCurves():
            self.result_plot.getXAxis().setAutoScale(True)
            self.result_plot.getYAxis().setAutoScale(True)
            self.result_plot.resetZoom()
        preview_note = f"; preview sampled every {step} pixels" if step > 1 else ""
        status = (
            f"{self._image_position_text()}; loaded {data.shape[1]} x "
            f"{data.shape[0]} pixels{preview_note}"
        )
        if calibration_cleared:
            status += "; incompatible detector: PONI and mask cleared"
        if references_cleared:
            self._reference_curve_cache.clear()
            status += "; incompatible Empty/Background cleared"
        self.status_label.setText(status)
        self._update_subtraction_status_block()
        self._update_navigation_buttons()
        self._pending_auto_integration = (
            self._auto_integrate_images and self.integrator is not None
        )
        self._start_detector_mask_update()

    @qt.Slot()
    def show_previous_image(self):
        if self.image_index > 0:
            self.image_index -= 1
            self._load_current_image()

    @qt.Slot()
    def show_next_image(self):
        if self.image_index + 1 < len(self.image_paths):
            self.image_index += 1
            self._load_current_image()

    @qt.Slot()
    def choose_poni(self):
        filename = self._pick("Select a PONI Calibration File", "pyFAI geometry (*.poni);;All files (*)")
        if filename:
            try:
                self.integrator = pyFAI.load(filename)
                self._poni_wavelength = self.integrator.wavelength
            except Exception as exc:
                self.show_error("Invalid PONI File", str(exc))
                return
            self._detector_mask_cache.clear()
            self._cake_cache.clear()
            self._reference_curve_cache.clear()
            if self.image_data is not None and not detector_accepts_shape(
                self.integrator.detector, self.image_data.shape
            ):
                self._clear_incompatible_calibration()
                self.status_label.setText(
                    "PONI detector does not match the current image; PONI and mask cleared"
                )
                return
            self.poni_edit.setText(filename)
            self._apply_current_energy()
            self._show_detector_information()
            self._invalidate_detector_sum()
            self._start_detector_mask_update()
            if self.image_data is None:
                self.status_label.setText("PONI loaded; select an image to calculate its detector mask")

    @qt.Slot()
    def choose_mask(self):
        filename = self._pick("Select a Mask", IMAGE_FILTER)
        if not filename:
            return
        try:
            mask = read_image(filename)
            if self.image_data is not None and mask.shape != self.image_data.shape:
                raise ValueError(f"Mask shape {mask.shape} does not match image shape {self.image_data.shape}")
        except Exception as exc:
            self.show_error("Unable to Read Mask", str(exc))
            return
        self.mask_data = mask.astype(bool)
        self._effective_mask_cache_valid = False
        self._cake_cache.clear()
        self._reference_curve_cache.clear()
        self.mask_edit.setText(filename)
        if self.image_data is not None:
            self._update_image_plot()
            self._update_source_sum_label()
        self._invalidate_detector_sum()
        total = np.count_nonzero(self._effective_mask())
        self.status_label.setText(f"Mask loaded; {total:,} pixels excluded in total")

    def _choose_reference_image(self, kind):
        if self.image_data is None:
            self.show_error("Image Required", "Load the sample image before selecting a reference.")
            return
        p62_mode = self._measurement_mode if self._beamline == "p62" else None
        dataset_path = {
            "saxs": "/scan/data/saxs_raw",
            "asaxs": "/scan/data/saxs_raw",
            "waxs": "/scan/data/waxs_raw",
            "awaxs": "/scan/data/waxs_raw",
        }.get(p62_mode)
        filenames, _ = qt.QFileDialog.getOpenFileNames(
            self,
            f"Select One or More {kind.title()} Images",
            self.input_path,
            P62_IMAGE_FILTER if p62_mode else IMAGE_FILTER,
        )
        if not filenames:
            return
        self._set_input_path(str(Path(filenames[0]).parent))
        try:
            reference_sources = []
            for filename in filenames:
                reference_sources.extend(
                    expand_image_file(filename, dataset_path)
                )
            if p62_mode and len(reference_sources) > 1:
                reference_sources = []
                for filename in filenames:
                    reference_sources.extend(
                        expand_image_file(
                            filename, dataset_path, include_energy=True
                        )
                    )
            # Validate every loaded data file now so a frame-count mismatch is
            # reported before integration or batch export starts.
            for sample_index, sample_source in enumerate(self.image_paths):
                matching_reference_source(
                    reference_sources, sample_source, self.image_paths,
                    sample_index,
                )
            reference_source = matching_reference_source(
                reference_sources, self.image_paths[self.image_index],
                self.image_paths, self.image_index,
            )
            data = read_image(reference_source)
            require_integer_detector_image(data, f"{kind.title()} image")
            if data.shape != self.image_data.shape:
                raise ValueError(
                    f"{kind.title()} shape {data.shape} does not match image shape "
                    f"{self.image_data.shape}"
                )
            if data.dtype != self.image_data.dtype:
                raise ValueError(
                    f"{kind.title()} data type {data.dtype} does not match image data "
                    f"type {self.image_data.dtype}"
                )
        except Exception as exc:
            self.show_error(f"Invalid {kind.title()} Image", str(exc))
            return
        self._reference_sources[kind] = reference_sources
        setattr(self, f"{kind}_data", data)
        self._reference_curve_cache = {
            key: value for key, value in self._reference_curve_cache.items()
            if key[0] != kind.title()
        }
        reference_description = Path(filenames[0]).name
        if len(filenames) > 1:
            reference_description = f"{len(filenames)} files"
        if len(reference_sources) > 1:
            reference_description += f" [{len(reference_sources)} images/frames]"
        getattr(self, f"{kind}_edit").setText(reference_description)
        self.status_label.setText(
            f"{kind.title()} image loaded; select Show or click Integrate"
        )
        self._update_subtraction_status_block()

    @qt.Slot()
    def choose_empty(self):
        self._choose_reference_image("empty")

    @qt.Slot()
    def choose_background(self):
        self._choose_reference_image("background")

    @qt.Slot()
    def clear_empty(self):
        self._clear_reference("empty")

    @qt.Slot()
    def clear_background(self):
        self._clear_reference("background")

    def _clear_reference(self, kind, update_plot=True):
        """Remove one reference and update the displayed 1-D result in place."""
        title = kind.title()
        setattr(self, f"{kind}_data", None)
        self._reference_sources[kind] = []
        getattr(self, f"{kind}_edit").clear()
        show_check = getattr(self, f"show_{kind}_check")
        subtract_check = getattr(self, f"subtract_{kind}_check")
        show_blocker = qt.QSignalBlocker(show_check)
        subtract_blocker = qt.QSignalBlocker(subtract_check)
        show_check.setChecked(False)
        subtract_check.setChecked(False)
        del show_blocker, subtract_blocker
        self._reference_curve_cache = {
            key: value for key, value in self._reference_curve_cache.items()
            if key[0] != title
        }
        if update_plot and self._last_integration_payload is not None:
            payload = self._last_integration_payload
            payload["references"].pop(title, None)
            payload.get("reference_cache_keys", {}).pop(title, None)
            corrected = payload["sample"].copy()
            for intensity, _show, subtract, factor in payload[
                "references"
            ].values():
                if subtract:
                    corrected -= factor * intensity
            payload["corrected"] = corrected
            self._render_integration_payload(payload)
        self.status_label.setText(f"{title} reference cleared")
        self._update_subtraction_status_block()

    @qt.Slot()
    def clear_mask(self):
        self.mask_data = None
        self._effective_mask_cache_valid = False
        self._cake_cache.clear()
        self._reference_curve_cache.clear()
        self.mask_edit.clear()
        if self.image_data is not None:
            self._update_image_plot()
            self._update_source_sum_label()
        self._invalidate_detector_sum()
        detector_count = 0 if self.detector_mask is None else np.count_nonzero(self.detector_mask)
        self.status_label.setText(
            f"User mask cleared; detector mask still excludes {detector_count:,} pixels"
        )

    def _optional_range(self, edit, name):
        text = edit.text().strip()
        if not text or text.casefold() == "auto":
            return None
        match = re.fullmatch(
            r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*,\s*"
            r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
            text,
        )
        if match is None:
            raise ValueError(f"Enter {name} as minimum, maximum, or leave it Auto")
        lo, hi = float(match.group(1)), float(match.group(2))
        if lo >= hi:
            raise ValueError(f"The {name} minimum must be less than the maximum")
        return (lo, hi)

    def _integration_ranges(self):
        return (
            self._optional_range(self.radial_range_edit, "radial range"),
            self._optional_range(self.azimuth_range_edit, "azimuthal range"),
        )

    def _error_model(self):
        """Return pyFAI's error-model token, or None when errors are disabled."""
        return self.error_model_combo.currentData()

    @qt.Slot()
    def start_integration(self):
        if self.image_data is None:
            self.show_error("Image Required", "Select a 2-D diffraction image first.")
            return
        self._apply_current_energy()
        poni = self.poni_edit.text().strip()
        if not poni or not os.path.isfile(poni):
            self.show_error("PONI File Required", "Select a valid .poni calibration file.")
            return
        if self.mask_data is not None and self.mask_data.shape != self.image_data.shape:
            self.show_error("Invalid Mask Shape", "The mask dimensions do not match the image.")
            return
        try:
            radial_range, azimuth_range = self._integration_ranges()
        except ValueError as exc:
            self.show_error("Invalid Parameters", str(exc))
            return

        references = []
        for name, data, filename, show, subtract, factor in (
            ("Empty", self.empty_data, self.empty_edit.text().strip(),
             self.show_empty_check.isChecked(),
             self.subtract_empty_check.isChecked(), self.empty_factor.value()),
            ("Background", self.background_data, self.background_edit.text().strip(),
             self.show_background_check.isChecked(),
             self.subtract_background_check.isChecked(), self.background_factor.value()),
        ):
            if (show or subtract) and data is None:
                self.show_error(f"{name} Required", f"Load a valid {name.lower()} image first.")
                return
            if data is not None and (show or subtract):
                cache_key = self._reference_cache_key(
                    name, radial_range, azimuth_range
                )
                references.append((
                    name, data, show, subtract, factor, cache_key,
                    self._reference_curve_cache.get(cache_key),
                ))

        self.integrate_button.setEnabled(False)
        self._auto_integrate_images = True
        self.status_label.setText("Integrating…")
        self._operation_cancel_requested = False
        self._update_export_progress_block(
            "operation", "Integration", "running…"
        )
        cake_cache_key = self._cake_cache_key(radial_range, azimuth_range)
        # IntegrationWorker performs pyFAI 1-D integration and optional Cake /
        # reference work only. Detector Sum and ROI Sum are never calculated
        # here; their cached values are produced exclusively by Calculate.
        self._worker = IntegrationWorker(
            self.image_data, self.integrator, self._effective_mask(),
            self.points_spin.value(), self.unit_combo.currentText(), radial_range,
            azimuth_range, self._error_model(),
            references,
            self.cake_check.isChecked(),
            self._cake_cache.get(cake_cache_key)
            if self.cake_check.isChecked() else None,
            cake_cache_key,
        )
        if self._exporting_video:
            payloads = []
            errors = []
            self._worker.finished.connect(payloads.append)
            self._worker.failed.connect(errors.append)
            self._worker.run()
            self._worker = None
            if errors:
                qt.QTimer.singleShot(
                    0, lambda error=errors[-1]: self.integration_failed(error)
                )
            elif payloads:
                qt.QTimer.singleShot(
                    0, lambda payload=payloads[-1]: self.integration_finished(payload)
                )
            return
        self._thread = qt.QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._integration_worker_succeeded)
        self._worker.failed.connect(self._integration_worker_failed)
        self._worker.cancelled.connect(self._thread.quit)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.failed.connect(self._worker.deleteLater)
        self._worker.cancelled.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._integration_thread_finished)
        self._thread.finished.connect(self._thread.deleteLater)
        self._integration_worker_payload = None
        self._integration_worker_error = None
        self._update_navigation_buttons()
        self._thread.start()

    @qt.Slot()
    def save_current_view(self):
        if self.image_data is None:
            self.show_error("Nothing to Save", "Load an image first.")
            return
        source = self.image_paths[self.image_index]
        if source.frame is None:
            default_name = f"{Path(source.path).stem}_integrated.png"
        else:
            default_name = (
                f"{Path(source.path).stem}_frame_{source.frame + 1:04d}_integrated.png"
            )
        filename, _ = qt.QFileDialog.getSaveFileName(
            self, "Save Source and Integration Plots",
            str(Path(self.export_path) / default_name),
            "PNG image (*.png);;JPEG image (*.jpg *.jpeg)"
        )
        if not filename:
            return
        if not Path(filename).suffix:
            filename += ".png"
        self._set_export_path(Path(filename).parent)
        if not self._save_current_plot_image(filename):
            self.show_error("Save Failed", f"Could not save {filename}")
            return
        self.status_label.setText(f"Saved visible plots to {filename}")

    def _save_current_plot_image(self, filename):
        """Save the visible Source/right-side plots for both Plot and Save All."""
        return self._grab_combined_plots().save(filename)

    @qt.Slot()
    def copy_visible_plots(self):
        """Copy visible plot canvases without silx's unstable PNG round-trip."""
        qt.QApplication.clipboard().setImage(self._grab_combined_plots())
        self.status_label.setText("Copied visible plots to the clipboard")

    def _grab_combined_plots(self):
        """Capture exactly the visible plot canvases, without GUI controls."""
        source = self.image_plot.getWidgetHandle().grab().toImage()
        current = self.right_stack.currentWidget()
        if current in (self.detector_sum_container, self.roi_sum_container):
            plot = (
                self.detector_sum_plot
                if current is self.detector_sum_container else self.roi_sum_plot
            )
            right = plot.getWidgetHandle().grab().toImage()
            height = max(source.height(), right.height())
            combined = qt.QImage(
                source.width() + right.width(), height, qt.QImage.Format_RGB888
            )
            combined.fill(qt.QColor("white"))
            painter = qt.QPainter(combined)
            painter.drawImage(0, (height - source.height()) // 2, source)
            painter.drawImage(
                source.width(), (height - right.height()) // 2, right
            )
            painter.end()
            return combined
        result = self.result_plot.getWidgetHandle().grab().toImage()
        if self.cake_check.isChecked():
            cake = self.cake_plot.getWidgetHandle().grab().toImage()
            right_width = max(cake.width(), result.width())
            right_height = cake.height() + result.height()
            right = qt.QImage(right_width, right_height, qt.QImage.Format_RGB888)
            right.fill(qt.QColor("white"))
            right_painter = qt.QPainter(right)
            right_painter.drawImage((right_width - cake.width()) // 2, 0, cake)
            right_painter.drawImage(
                (right_width - result.width()) // 2, cake.height(), result
            )
            right_painter.end()
        else:
            right = result

        height = max(source.height(), right.height())
        combined = qt.QImage(
            source.width() + right.width(), height, qt.QImage.Format_RGB888
        )
        combined.fill(qt.QColor("white"))
        painter = qt.QPainter(combined)
        painter.drawImage(0, (height - source.height()) // 2, source)
        painter.drawImage(source.width(), (height - right.height()) // 2, right)
        painter.end()
        return combined

    def _select_video_axis_scales(self):
        """Ask which logarithmic axes to use for video integration plots."""
        dialog = qt.QDialog(self)
        dialog.setWindowTitle("Video Axis Scale")
        layout = qt.QVBoxLayout(dialog)
        layout.addWidget(qt.QLabel("Select the axis scale for the integration plot:"))
        log_x = qt.QCheckBox("Log X")
        log_y = qt.QCheckBox("Log Y")
        log_x.setChecked(self.result_plot.getXAxis().getScale() == "log")
        log_y.setChecked(self.result_plot.getYAxis().getScale() == "log")
        layout.addWidget(log_x)
        layout.addWidget(log_y)
        buttons = qt.QDialogButtonBox(
            qt.QDialogButtonBox.Ok | qt.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != qt.QDialog.Accepted:
            return False
        self.result_plot.getXAxis().setScale("log" if log_x.isChecked() else "linear")
        self.result_plot.getYAxis().setScale("log" if log_y.isChecked() else "linear")
        self.result_plot.getXAxisLogarithmicAction().setChecked(log_x.isChecked())
        self.result_plot.getYAxisLogarithmicAction().setChecked(log_y.isChecked())
        self.result_plot.resetZoom()
        return True

    @qt.Slot()
    def save_all_integrated_data(self):
        if not self.image_paths:
            self.show_error("No Images", "Load one or more images first.")
            return
        poni = self.poni_edit.text().strip()
        if not poni or not os.path.isfile(poni):
            self.show_error("PONI File Required", "Load a valid PONI file before exporting data.")
            return
        try:
            radial_range, azimuth_range = self._integration_ranges()
        except ValueError as exc:
            self.show_error("Invalid Parameters", str(exc))
            return
        output_dir = qt.QFileDialog.getExistingDirectory(
            self, "Select Folder for Integrated ASCII Data", self.export_path
        )
        if not output_dir:
            return
        self._set_export_path(output_dir)

        self._start_ascii_export(output_dir, reset_progress=True)

    def _start_ascii_export(self, output_dir, reset_progress=True):
        """Start the existing batch ASCII writer in a chosen directory."""

        poni = self.poni_edit.text().strip()
        if not poni or not os.path.isfile(poni):
            self.show_error(
                "PONI File Required",
                "Load a valid PONI file before exporting integrated data.",
            )
            return
        try:
            radial_range, azimuth_range = self._integration_ranges()
        except ValueError as exc:
            self.show_error("Invalid Parameters", str(exc))
            return
        if reset_progress:
            self._reset_export_progress()
        self._append_export_progress("Integrated data:")
        references = []
        if self._reference_sources["empty"]:
            references.append((
                "Empty", list(self._reference_sources["empty"]),
                self.empty_edit.text().strip(),
                self.empty_factor.value(), self.subtract_empty_check.isChecked(),
            ))
        if self._reference_sources["background"]:
            references.append((
                "Background", list(self._reference_sources["background"]),
                self.background_edit.text().strip(),
                self.background_factor.value(), self.subtract_background_check.isChecked(),
            ))
        self._batch_thread = qt.QThread(self)
        self._batch_worker = BatchIntegrationWorker(
            list(self.image_paths), output_dir, poni, self.mask_data,
            self.points_spin.value(), self.unit_combo.currentText(), radial_range,
            azimuth_range, self._error_model(),
            references, self._measurement_mode in ("asaxs", "awaxs"),
            self._measurement_mode == "asaxs",
        )
        self._batch_worker.moveToThread(self._batch_thread)
        self._batch_thread.started.connect(self._batch_worker.run)
        self._batch_worker.progress.connect(self._batch_data_progress)
        self._batch_worker.finished.connect(self._batch_worker_succeeded)
        self._batch_worker.failed.connect(self._batch_worker_failed)
        self._batch_worker.cancelled.connect(self._batch_thread.quit)
        self._batch_worker.finished.connect(self._batch_thread.quit)
        self._batch_worker.failed.connect(self._batch_thread.quit)
        self._batch_worker.finished.connect(self._batch_worker.deleteLater)
        self._batch_worker.failed.connect(self._batch_worker.deleteLater)
        self._batch_worker.cancelled.connect(self._batch_worker.deleteLater)
        self._batch_thread.finished.connect(self._batch_thread_finished)
        self._batch_thread.finished.connect(self._batch_thread.deleteLater)
        self._batch_worker_result = None
        self._batch_worker_error = None
        self.integrate_button.setEnabled(False)
        self.save_current_action.setEnabled(False)
        self.save_batch_plots_action.setEnabled(False)
        self.save_batch_video_action.setEnabled(False)
        self.save_data_action.setEnabled(False)
        self.save_ascii_video_action.setEnabled(False)
        self._update_navigation_buttons()
        self.status_label.setText("Exporting integrated ASCII data...")
        self._operation_cancel_requested = False
        self._update_export_progress_block(
            "operation", "Export", "starting…"
        )
        self._batch_thread.start()

    @qt.Slot(int, int, str)
    def _batch_data_progress(self, current, total, filename):
        source_progress = self._source_file_frame_progress(
            self.image_paths, current - 1
        )
        self.status_label.setText(
            f"Saving [{current}/{total}]; {source_progress}"
        )
        self._update_subtraction_status_block(current - 1)
        self._update_scrolling_filename(
            self.image_paths[current - 1], "data"
        )

    @qt.Slot(str, int)
    def _batch_worker_succeeded(self, output_dir, count):
        """Store the result; continue only after QThread has fully stopped."""
        self._batch_worker_result = (output_dir, count)

    @qt.Slot(str)
    def _batch_worker_failed(self, details):
        """Store the error; report it only after QThread has fully stopped."""
        self._batch_worker_error = details

    @qt.Slot()
    def _batch_thread_finished(self):
        result = self._batch_worker_result
        error = self._batch_worker_error
        self._batch_worker_result = None
        self._batch_worker_error = None
        # Keep the Python wrappers alive until Qt has processed deleteLater for
        # both objects. Dropping the last wrapper inside QThread.finished can
        # destroy a QObject while Qt is still dispatching its final signals.
        qt.QTimer.singleShot(
            50, lambda result=result, error=error: self._finalize_batch_thread(
                result, error
            )
        )

    def _finalize_batch_thread(self, result, error):
        self._batch_worker = None
        self._batch_thread = None
        if self._operation_cancel_requested:
            self._operation_stopped()
            return
        if error is not None:
            self._batch_data_failed(error)
        elif result is not None:
            # Run the next GUI/video stage on a fresh event-loop turn. This
            # also lets deferred HDF5/Fabio destruction complete on Windows.
            qt.QTimer.singleShot(
                0, lambda result=result: self._batch_data_finished(*result)
            )

    @qt.Slot(str, int)
    def _batch_data_finished(self, output_dir, count):
        if self._combined_export_plot_path is not None:
            plot_path = self._combined_export_plot_path
            self._combined_export_plot_path = None
            image = self._grab_combined_plots()
            if not image.save(plot_path):
                self._set_save_actions_enabled(True)
                self.integrate_button.setEnabled(True)
                self._update_navigation_buttons()
                self.show_error("Save Failed", f"Could not save {plot_path}")
                return
            self._set_save_actions_enabled(True)
            self.integrate_button.setEnabled(True)
            self._update_navigation_buttons()
            self._append_export_progress(f"Plot saved: {Path(plot_path).name}")
            self.status_label.setText(
                f"Saved ASCII data and current plot to {output_dir}"
            )
            return
        if self._combined_export_video_dir is not None:
            video_dir = self._combined_export_video_dir
            self._combined_export_video_dir = None
            self._append_export_progress(f"ASCII complete: [{count}/{count}]")
            self._combined_export_video_dir = video_dir
            self._start_next_combined_plot()
            return
        self.integrate_button.setEnabled(True)
        self.save_current_action.setEnabled(True)
        self.save_batch_plots_action.setEnabled(True)
        self.save_batch_video_action.setEnabled(True)
        self.save_data_action.setEnabled(True)
        self.save_ascii_video_action.setEnabled(True)
        self._update_navigation_buttons()
        self.status_label.setText(f"Saved {count} integrated .dat files to {output_dir}")
        self._append_export_progress(f"Complete: [{count}/{count}]")

    @qt.Slot(str)
    def _batch_data_failed(self, details):
        self._combined_export_video_dir = None
        self._combined_export_plot_path = None
        self.integrate_button.setEnabled(True)
        self.save_current_action.setEnabled(True)
        self.save_batch_plots_action.setEnabled(True)
        self.save_batch_video_action.setEnabled(True)
        self.save_data_action.setEnabled(True)
        self.save_ascii_video_action.setEnabled(True)
        self._update_navigation_buttons()
        self.show_error("ASCII Data Export Failed", details)
        self.status_label.setText("Integrated ASCII data export failed")
        self._append_export_progress("Integrated data: failed")

    def _start_video_export(self, sources, filename, progress_key=None):
        try:
            import imageio.v2 as imageio
        except ModuleNotFoundError:
            self.show_error(
                "Video Dependency Missing",
                "Batch Video requires imageio and imageio-ffmpeg. Install them with:\n"
                "python -m pip install imageio imageio-ffmpeg",
            )
            if self._batch_video_original_paths is not None:
                self.image_paths = self._batch_video_original_paths
                self.image_index = self._batch_video_original_index
                self._auto_integrate_images = self._batch_video_original_auto
                self._batch_video_original_paths = None
                self._batch_video_queue.clear()
            self._set_save_actions_enabled(True)
            return
        try:
            self._video_writer = imageio.get_writer(
                filename, fps=1.25, codec="libx264", quality=8,
                macro_block_size=2, ffmpeg_log_level="error"
            )
        except Exception as exc:
            self.show_error("Video Export Failed", str(exc))
            if self._batch_video_original_paths is not None:
                self.image_paths = self._batch_video_original_paths
                self.image_index = self._batch_video_original_index
                self._auto_integrate_images = self._batch_video_original_auto
                self._batch_video_original_paths = None
                self._batch_video_queue.clear()
            self._set_save_actions_enabled(True)
            self._update_navigation_buttons()
            return
        self._video_path = filename
        self._video_progress_key = progress_key or Path(filename).stem
        self._video_original_index = self.image_index
        self._video_original_auto = self._auto_integrate_images
        self.image_paths = list(sources)
        self._exporting_video = True
        self._auto_integrate_images = True
        self.save_current_action.setEnabled(False)
        self.save_batch_plots_action.setEnabled(False)
        self.save_batch_video_action.setEnabled(False)
        self.save_data_action.setEnabled(False)
        self.save_ascii_video_action.setEnabled(False)
        self.integrate_button.setEnabled(False)
        self.image_index = 0
        saving_current, saving_total = self._video_saving_position(
            self.image_paths[0]
        )
        self._update_video_status(
            f"Saving [{saving_current}/{saving_total}]; "
            f"{self._source_file_frame_progress(self.image_paths, 0)}",
        )
        self._load_current_image()

    @qt.Slot()
    def save_batch_videos(self):
        if len(self.image_paths) < 2:
            self.show_error("Multiple Images Required", "Load at least two files for batch video.")
            return
        poni = self.poni_edit.text().strip()
        if not poni or not os.path.isfile(poni):
            self.show_error("PONI File Required", "Load a valid PONI file before exporting videos.")
            return
        if not self._select_video_axis_scales():
            return
        output_dir = qt.QFileDialog.getExistingDirectory(
            self, "Select Folder for Batch Videos", self.export_path
        )
        if not output_dir:
            return
        self._set_export_path(output_dir)

        self._start_batch_video_export(output_dir, reset_progress=True)

    def _start_batch_video_export(
        self, output_dir, reset_progress=True, grouped_sources=None
    ):
        """Start grouped video export in an already selected directory."""

        self._batch_video_original_paths = list(self.image_paths)
        self._batch_video_original_index = self.image_index
        self._batch_video_original_auto = self._auto_integrate_images
        self._batch_video_queue = []
        grouped_sources = (
            group_batch_video_sources(self.image_paths)
            if grouped_sources is None else grouped_sources
        )
        if reset_progress:
            self._reset_export_progress()
        for prefix, ordered in grouped_sources:
            output_path = str(Path(output_dir) / f"{prefix}.mp4")
            self._batch_video_queue.append((prefix, ordered, output_path))
        prefix, sources, output_path = self._batch_video_queue.pop(0)
        self._start_video_export(sources, output_path, prefix)

    def _start_next_combined_plot(self):
        """Integrate one singleton group, then reuse Save Plot's save helper."""
        if not self._combined_export_plot_queue:
            self._exporting_plot = False
            if self._combined_export_video_groups:
                self.image_paths = self._plot_export_original_paths
                self.image_index = self._plot_export_original_index
                self._auto_integrate_images = self._plot_export_original_auto
                self._plot_export_original_paths = None
                self._start_batch_video_export(
                    self._combined_export_video_dir,
                    reset_progress=False,
                    grouped_sources=self._combined_export_video_groups,
                )
            else:
                self._combined_export_video_dir = None
                self._restore_plot_export_state()
                if self._batch_plot_only:
                    output_dir = self._batch_plot_output_dir
                    total = self._batch_plot_total
                    self._batch_plot_only = False
                    self._batch_plot_output_dir = ""
                    self._set_save_actions_enabled(True)
                    self.integrate_button.setEnabled(True)
                    self._update_navigation_buttons()
                    self.status_label.setText(
                        f"Saved {total} plot image(s) to {output_dir}"
                    )
                    self._append_export_progress(
                        f"Batch plots complete: [{total}/{total}]"
                    )
            return
        if self._plot_export_original_paths is None:
            self._plot_export_original_paths = list(self.image_paths)
            self._plot_export_original_index = self.image_index
            self._plot_export_original_auto = self._auto_integrate_images
        source, filename = self._combined_export_plot_queue.pop(0)
        if self._batch_plot_only:
            self._batch_plot_current += 1
            original_sources = self._plot_export_original_paths or self.image_paths
            source_progress = self._source_file_frame_progress(
                original_sources, self._batch_plot_current - 1
            )
            self.status_label.setText(
                f"Saving [{self._batch_plot_current}/{self._batch_plot_total}]; "
                f"{source_progress}"
            )
            self._update_scrolling_filename(source, "plot")
        self._exporting_plot = True
        self.image_paths = [source]
        self.image_index = 0
        self._auto_integrate_images = True
        self._plot_export_current_filename = filename
        self._load_current_image()

    def _restore_plot_export_state(self):
        if self._plot_export_original_paths is None:
            return
        self.image_paths = self._plot_export_original_paths
        self.image_index = self._plot_export_original_index
        self._auto_integrate_images = self._plot_export_original_auto
        self._plot_export_original_paths = None
        self._plot_export_current_filename = ""
        self._load_current_image()

    @qt.Slot()
    def save_batch_plots(self):
        """Save one PNG for every selected image/frame to one folder."""
        if not self.image_paths:
            self.show_error("No Images", "Load one or more images first.")
            return
        poni = self.poni_edit.text().strip()
        if not poni or not os.path.isfile(poni):
            self.show_error(
                "PONI File Required",
                "Load a valid PONI file before exporting batch plots.",
            )
            return
        try:
            self._integration_ranges()
        except ValueError as exc:
            self.show_error("Invalid Parameters", str(exc))
            return
        output_dir = qt.QFileDialog.getExistingDirectory(
            self, "Select Folder for Batch Plots", self.export_path
        )
        if not output_dir:
            return
        self._set_export_path(output_dir)
        self._combined_export_video_dir = None
        self._combined_export_video_groups = []
        self._combined_export_plot_queue = []
        for source in self.image_paths:
            if source.frame is None:
                plot_name = f"{Path(source.path).stem}_integrated.png"
            else:
                plot_name = (
                    f"{Path(source.path).stem}_frame_"
                    f"{source.frame + 1:04d}_integrated.png"
                )
            self._combined_export_plot_queue.append(
                (source, str(Path(output_dir) / plot_name))
            )
        self._batch_plot_only = True
        self._batch_plot_output_dir = output_dir
        self._batch_plot_total = len(self._combined_export_plot_queue)
        self._batch_plot_current = 0
        self._reset_export_progress()
        self._set_save_actions_enabled(False)
        self.integrate_button.setEnabled(False)
        self._start_next_combined_plot()

    @qt.Slot()
    def save_data_and_plots(self):
        """Export ASCII plus PNG for singleton groups or MP4 for frame groups."""
        if not self.image_paths:
            self.show_error("No Images", "Load one or more images first.")
            return
        poni = self.poni_edit.text().strip()
        if not poni or not os.path.isfile(poni):
            self.show_error(
                "PONI File Required",
                "Load a valid PONI file before exporting data and plots.",
            )
            return
        try:
            self._integration_ranges()
        except ValueError as exc:
            self.show_error("Invalid Parameters", str(exc))
            return
        grouped_sources = group_batch_video_sources(self.image_paths)
        video_groups = [group for group in grouped_sources if len(group[1]) > 1]
        multiple = len(self.image_paths) > 1
        if video_groups and not self._select_video_axis_scales():
            return
        output_dir = qt.QFileDialog.getExistingDirectory(
            self, "Select Folder for Data and Plots", self.export_path
        )
        if not output_dir:
            return
        self._set_export_path(output_dir)
        self._combined_export_video_dir = None
        self._combined_export_plot_path = None
        self._combined_export_plot_queue = []
        self._combined_export_video_groups = []
        if multiple:
            self._combined_export_video_dir = output_dir
            for prefix, ordered in grouped_sources:
                if len(ordered) == 1:
                    source = ordered[0]
                    if source.frame is None:
                        plot_name = f"{Path(source.path).stem}_integrated.png"
                    else:
                        plot_name = (
                            f"{Path(source.path).stem}_frame_"
                            f"{source.frame + 1:04d}_integrated.png"
                        )
                    self._combined_export_plot_queue.append(
                        (source, str(Path(output_dir) / plot_name))
                    )
                else:
                    self._combined_export_video_groups.append(
                        (prefix, ordered, str(Path(output_dir) / f"{prefix}.mp4"))
                    )
        else:
            source = self.image_paths[0]
            if source.frame is None:
                plot_name = f"{Path(source.path).stem}_integrated.png"
            else:
                plot_name = (
                    f"{Path(source.path).stem}_frame_"
                    f"{source.frame + 1:04d}_integrated.png"
                )
            self._combined_export_plot_path = str(Path(output_dir) / plot_name)
        self._start_ascii_export(output_dir, reset_progress=True)

    def _append_video_frame(self):
        if not self._exporting_video:
            return
        image = self._grab_combined_plots().convertToFormat(qt.QImage.Format_RGB888)
        width, height = image.width(), image.height()
        bytes_per_line = image.bytesPerLine()
        # Copy through Qt's bounded API. Resizing the sip.voidptr returned by
        # bits() can trigger Qt6Core's 0xc0000409 native buffer-overrun abort.
        image_bytes = image.constBits().asstring(height * bytes_per_line)
        frame = np.frombuffer(image_bytes, dtype=np.uint8).reshape(
            height, bytes_per_line
        )
        frame = frame[:, : width * 3].reshape(height, width, 3).copy()
        # H.264/yuv420 requires even dimensions. Pad instead of resizing the GUI.
        pad_height = height % 2
        pad_width = width % 2
        if pad_height or pad_width:
            frame = np.pad(
                frame,
                ((0, pad_height), (0, pad_width), (0, 0)),
                mode="edge",
            )
        try:
            self._video_writer.append_data(frame)
        except Exception as exc:
            self._finish_video_export(False, str(exc))
            return
        current = self.image_index + 1
        total = len(self.image_paths)
        source = self.image_paths[self.image_index]
        source_path = source.path if isinstance(source, ImageSource) else str(source)
        source_progress = self._source_file_frame_progress(
            self.image_paths, self.image_index
        )
        saving_current, saving_total = self._video_saving_position(source)
        self._update_video_status(
            f"Saving [{saving_current}/{saving_total}]; {source_progress}",
        )
        self._update_subtraction_status_block(self.image_index)
        self._update_scrolling_filename(source, "video")
        if current == total:
            self._update_export_progress_block(
                f"complete:{self._video_progress_key}",
                self._video_progress_key,
                f"complete [{total}/{total}]",
            )
        # Video export is sequential. Retaining per-source Cake payloads gives
        # no benefit and can exhaust memory for large detector stacks.
        self._last_integration_payload = None
        self._cake_cache.clear()
        del frame, image_bytes, image
        if self.image_index + 1 < len(self.image_paths):
            self.image_index += 1
            self._load_current_image()
        else:
            self._finish_video_export(True)

    def _finish_video_export(self, success, error="", cancelled=False):
        writer, self._video_writer = self._video_writer, None
        if writer is not None:
            writer.close()
        path = self._video_path
        original_index = self._video_original_index
        original_auto = self._video_original_auto
        self._exporting_video = False
        self._pending_auto_integration = False
        self._auto_integrate_images = False

        if self._batch_video_original_paths is not None and success and self._batch_video_queue:
            prefix, sources, output_path = self._batch_video_queue.pop(0)
            self._start_video_export(sources, output_path, prefix)
            return

        batch_mode = self._batch_video_original_paths is not None
        if batch_mode:
            self.image_paths = self._batch_video_original_paths
            original_index = self._batch_video_original_index
            original_auto = self._batch_video_original_auto
            self._batch_video_original_paths = None
            self._batch_video_queue.clear()
        self.save_current_action.setEnabled(True)
        self.save_batch_plots_action.setEnabled(True)
        self.save_batch_video_action.setEnabled(True)
        self.save_data_action.setEnabled(True)
        self.save_ascii_video_action.setEnabled(True)
        self.integrate_button.setEnabled(True)
        if 0 <= original_index < len(self.image_paths):
            self.image_index = original_index
            self._load_current_image()
        self._auto_integrate_images = original_auto
        if success:
            self.status_label.setText(
                "Batch videos saved" if batch_mode else f"Saved video to {path}"
            )
        elif not cancelled:
            self.show_error("Video Export Failed", error)
        self._update_navigation_buttons()
        if cancelled:
            self._operation_stopped()

    @qt.Slot(object)
    def _integration_worker_succeeded(self, payload):
        self._integration_worker_payload = payload

    @qt.Slot(str)
    def _integration_worker_failed(self, details):
        self._integration_worker_error = details

    @qt.Slot()
    def _integration_thread_finished(self):
        payload = self._integration_worker_payload
        error = self._integration_worker_error
        self._integration_worker_payload = None
        self._integration_worker_error = None
        qt.QTimer.singleShot(
            50, lambda payload=payload, error=error:
            self._finalize_integration_thread(payload, error)
        )

    def _finalize_integration_thread(self, payload, error):
        self._worker = None
        self._thread = None
        if self._operation_cancel_requested:
            self._operation_stopped()
            return
        if error is not None:
            qt.QTimer.singleShot(
                0, lambda error=error: self.integration_failed(error)
            )
        elif payload is not None:
            qt.QTimer.singleShot(
                0, lambda payload=payload: self.integration_finished(payload)
            )

    @qt.Slot(object)
    def integration_finished(self, payload):
        self._last_integration_payload = payload
        if payload.get("cake") is not None:
            self._cake_cache[payload["cake_cache_key"]] = payload["cake"]
        for name, cache_key in payload.get("reference_cache_keys", {}).items():
            self._reference_curve_cache[cache_key] = payload.get(
                "reference_cache_values", {}
            ).get(name, payload["references"][name][0])
        self._render_integration_payload(payload)
        # Scale only after every selected curve has been added. Scaling when the
        # Data curve is added would omit a later Subtracted data curve.
        self.result_plot.getXAxis().setAutoScale(True)
        self.result_plot.getYAxis().setAutoScale(True)
        self.result_plot.resetZoom()
        radial = payload["radial"]
        self.integrate_button.setEnabled(True)
        self.status_label.setText(
            f"Integration complete: {len(radial):,} data points"
        )
        self._update_subtraction_status_block()
        self._update_export_progress_block(
            "operation", "Integration", f"complete ({len(radial):,} points)"
        )
        self._update_navigation_buttons()
        if self._exporting_video:
            qt.QTimer.singleShot(0, self._append_video_frame)
        elif self._exporting_plot:
            qt.QTimer.singleShot(0, self._save_combined_plot_after_integration)

    def _save_combined_plot_after_integration(self):
        """Save singleton Save All output through the normal Plot save helper."""
        filename = self._plot_export_current_filename
        if not self._save_current_plot_image(filename):
            self._exporting_plot = False
            self._batch_plot_only = False
            self._restore_plot_export_state()
            self._set_save_actions_enabled(True)
            self.integrate_button.setEnabled(True)
            self.show_error("Save Failed", f"Could not save {filename}")
            return
        self._append_export_progress(
            f"Plot saved: {Path(filename).name}"
        )
        self._start_next_combined_plot()

    def _render_integration_payload(self, payload):
        radial = payload["radial"]
        cake = payload.get("cake")
        if cake is not None and cake["radial"].size and cake["azimuthal"].size:
            self._render_cake_image(payload, resetzoom=True)
        corrected_used = any(
            values[2] for values in payload["references"].values()
        )
        self.result_plot.clear()
        self.result_plot.addCurve(
            radial, payload["sample"], legend="Data", resetzoom=True
        )
        if corrected_used:
            self.result_plot.addCurve(
                radial, payload["corrected"], legend="Subtracted data",
                resetzoom=False,
            )
        for name, (intensity, _show, _subtract, factor) in payload["references"].items():
            show = (
                self.show_empty_check.isChecked() if name == "Empty"
                else self.show_background_check.isChecked()
            )
            if show:
                self.result_plot.addCurve(
                    radial, factor * intensity, legend=name,
                    resetzoom=False,
                )
        self.result_plot.setGraphXLabel(payload["unit"])
        self.result_plot.setGraphYLabel("Intensity")
        self.result_plot.replot()
        for curve in self.result_plot.getAllCurves():
            curve.sigItemChanged.connect(self._result_curve_changed)
        self._refresh_result_legend()

    def _render_cake_image(self, payload, resetzoom):
        """Render Cake with a strictly positive X extent on a logarithmic axis."""
        cake = payload["cake"]
        radial_axis = np.asarray(cake["radial"])
        azimuth_axis = np.asarray(cake["azimuthal"])
        intensity = np.asarray(cake["intensity"])
        positive_intensity = intensity[np.isfinite(intensity) & (intensity > 0)]
        if positive_intensity.size:
            cake_vmin, cake_vmax = np.percentile(positive_intensity, (1.0, 99.5))
            if cake_vmin >= cake_vmax:
                cake_vmin = float(positive_intensity.min())
                cake_vmax = float(positive_intensity.max())
            if cake_vmin >= cake_vmax:
                cake_vmax = cake_vmin * 10.0
        else:
            cake_vmin, cake_vmax = 1.0, 10.0
        radial_step = (
            float(np.mean(np.diff(radial_axis))) if radial_axis.size > 1 else 1.0
        )
        azimuth_step = (
            float(np.mean(np.diff(azimuth_axis))) if azimuth_axis.size > 1 else 1.0
        )
        if self._cake_log_mesh is not None:
            try:
                self._cake_log_mesh.remove()
            except ValueError:
                pass
            self._cake_log_mesh = None
        start = 0
        if self.cake_plot.getXAxis().getScale() == "log":
            left_edges = radial_axis - radial_step / 2.0
            positive = np.flatnonzero(left_edges > 0)
            if not positive.size:
                self.cake_plot.clear()
                return
            start = int(positive[0])
        self.cake_plot.clear()
        cake_colormap = Colormap()
        cake_colormap.setFromColormap(self.cake_plot.getDefaultColormap())
        if cake_colormap.getVRange() == (None, None):
            cake_colormap.setVRange(float(cake_vmin), float(cake_vmax))
        self.cake_plot.addImage(
            intensity[:, start:],
            legend="cake",
            origin=(
                float(radial_axis[start] - radial_step / 2.0),
                float(azimuth_axis[0] - azimuth_step / 2.0),
            ),
            scale=(radial_step, azimuth_step),
            colormap=cake_colormap,
            resetzoom=resetzoom,
        )
        active_image = self.cake_plot.getActiveImage()
        active_colormap = active_image.getColormap()
        if self._cake_colormap_signal_source is not None:
            try:
                self._cake_colormap_signal_source.sigChanged.disconnect(
                    self._cake_colormap_changed
                )
            except (RuntimeError, TypeError):
                pass
        self._cake_colormap_signal_source = active_colormap
        active_colormap.sigChanged.connect(self._cake_colormap_changed)
        if self.cake_plot.getXAxis().getScale() == "log":
            # silx's Matplotlib backend deliberately drops AxesImage items on
            # logarithmic axes. Draw the same values as a QuadMesh so Cake is
            # visible while retaining the silx image item for data/profile use.
            x_edges = np.concatenate((
                radial_axis[start:] - radial_step / 2.0,
                [radial_axis[-1] + radial_step / 2.0],
            ))
            y_edges = np.concatenate((
                azimuth_axis - azimuth_step / 2.0,
                [azimuth_axis[-1] + azimuth_step / 2.0],
            ))
            backend = self.cake_plot.getBackend()
            self._cake_log_mesh = backend.ax.pcolormesh(
                x_edges,
                y_edges,
                intensity[:, start:],
                shading="flat",
                rasterized=True,
            )
            self._update_cake_log_mesh_colors()
        self.cake_plot.setGraphXLabel(payload["unit"])
        self.cake_plot.setGraphYLabel("Azimuthal angle (°)")

    @qt.Slot()
    def _cake_colormap_changed(self):
        """Apply Cake's native silx colormap settings to its Log-X mesh."""
        self._update_cake_log_mesh_colors()
        self.cake_plot.replot()

    def _update_cake_log_mesh_colors(self):
        if self._cake_log_mesh is None or self._last_integration_payload is None:
            return
        image = self.cake_plot.getActiveImage()
        if image is None:
            return
        data = image.getData(copy=False)
        rgba = image.getColormap().applyToData(data)
        self._cake_log_mesh.set_array(None)
        self._cake_log_mesh.set_facecolors(
            rgba.reshape(-1, 4).astype(np.float32, copy=False) / 255.0
        )

    @qt.Slot(object)
    def _result_curve_changed(self, _event):
        """Refresh the manual legend after silx applies a curve style change."""
        qt.QTimer.singleShot(0, self._refresh_result_legend)

    @qt.Slot()
    def _refresh_result_legend(self):
        """Rebuild the frameless legend from the current visible curve styles."""
        backend = self.result_plot.getBackend()
        if hasattr(backend, "ax"):
            lines = [line for line in backend.ax.get_lines() if line.get_visible()]
            curves = [curve for curve in self.result_plot.getAllCurves()
                      if curve.isVisible()]
            for line, curve in zip(lines, curves):
                line.set_label(curve.getName())
            old_legend = backend.ax.get_legend()
            if old_legend is not None:
                old_legend.remove()
            if lines:
                backend.ax.legend(
                    handles=lines[:len(curves)], loc="best", frameon=False
                )
            backend.fig.canvas.draw_idle()

    @qt.Slot(str)
    def integration_failed(self, details):
        self.integrate_button.setEnabled(True)
        self._update_navigation_buttons()
        if self._exporting_video:
            self._finish_video_export(False, details)
            return
        self.show_error("Integration Failed", details)
        self.status_label.setText("Integration failed; check the image, PONI file, and mask")

    def show_error(self, title, message):
        box = qt.QMessageBox(self)
        box.setIcon(qt.QMessageBox.Critical)
        box.setWindowTitle(title)
        box.setText(message.splitlines()[-1] if message else title)
        if "\n" in message:
            box.setDetailedText(message)
        box.exec()


def main():
    app = qt.QApplication.instance() or qt.QApplication(sys.argv)
    with filter_libpng_iccp_warnings():
        window = MainWindow()
        window.show()
        return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
