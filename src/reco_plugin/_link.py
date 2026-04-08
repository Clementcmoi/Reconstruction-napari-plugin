# Imports
import numpy as np
import cupy as cp
from tqdm import tqdm
import gc

# Local imports
from .processing.cor import *
from .processing.phase import paganin_filter, unsharp_mask, paganin_filter_slice, get_padding_size_slice
from .processing.process import apply_flat_darkfield, double_flatfield_correction
from .processing.reconstruction import (
    reconstruct_from_sinogram_slice, create_angles, create_disk_mask
)
from .processing.sinogram import create_sinogram, create_sinogram_slice
from .processing.angles import (
    find_angles_in_dataset, find_opposite_pairs_best_match,
    create_sinogram_slice_from_pairs, create_sinograms_from_pairs
)
from .utils.qt_helpers import create_processing_dialog, PlotWindow, WorkerThread
from .utils.link_utils import (
    add_image_to_layer, clear_memory, get_projections, get_angles,
    apply_mask_and_reconstruct, convert_cor_to_shift, pad_and_shift_projection,
    resize_to_target, load_angles_and_create_sinograms, process_volume_bigdata
)


# ─── Thread helper ────────────────────────────────────────────────────────────

def run_in_thread(func, on_result=None, on_error=None):
    worker = WorkerThread(func)
    if on_result:
        worker.finished.connect(on_result)
    if on_error:
        worker.error.connect(on_error)
    worker.start()
    return worker, worker


# ─── Metadata / history helpers ───────────────────────────────────────────────

def _get_layer_history(viewer, layer_name):
    """Return the processing_history list from a layer's metadata, or []."""
    try:
        return list(viewer.layers[layer_name].metadata.get('processing_history', []))
    except (KeyError, AttributeError):
        return []


def _layer_has_step(viewer, layer_name, step):
    """True if any entry in processing_history starts with `step`."""
    return any(s.startswith(step) for s in _get_layer_history(viewer, layer_name))


def _history_has(history, step):
    """True if any entry in a history list starts with `step`."""
    return any(s.startswith(step) for s in history)


def _save_layer(viewer, widget, data, layer_name, metadata):
    """
    Add `data` as a napari Image layer named `layer_name`.
    Then set the sample_selection ComboBox to that layer.
    Must be called from the main thread.
    """
    # Remove existing layer with same name to avoid duplicates
    if layer_name in [l.name for l in viewer.layers]:
        viewer.layers.remove(layer_name)
    viewer.add_image(data, name=layer_name, metadata=metadata)
    idx = widget.sample_selection.findText(layer_name)
    if idx >= 0:
        widget.sample_selection.setCurrentIndex(idx)


# ─── Main-thread helpers ──────────────────────────────────────────────────────

def _get_source_hdf5_path(viewer, layer_name):
    """
    Return the original HDF5/NXS file path associated with a layer.
    Works on both direct nxs layers (which have 'paths') and on
    intermediate layers that carry 'hdf5_source_path' in their metadata.
    """
    try:
        meta = viewer.layers[layer_name].metadata
        paths = meta.get('paths', [None])
        if paths and paths[0]:
            return paths[0]
        return meta.get('hdf5_source_path')
    except Exception:
        return None


def _get_hdf5_path(viewer, experiment):
    return _get_source_hdf5_path(viewer, experiment.sample_images)


def _read_raw(viewer, experiment, widget):
    sample = np.array(viewer.layers[experiment.sample_images].data)
    dark = (np.mean(np.array(viewer.layers[experiment.darkfield].data), axis=0)
            if widget.darkfield_checkbox.isChecked() else None)
    flat = (np.mean(np.array(viewer.layers[experiment.flatfield].data), axis=0)
            if widget.flatfield_checkbox.isChecked() else None)
    return sample, flat, dark


def _read_paganin_params(experiment):
    try:
        return (float(experiment.energy),
                float(experiment.pixel),
                float(experiment.dist_object_detector),
                float(experiment.db))
    except (TypeError, ValueError):
        return None


# ─── Algo helpers ─────────────────────────────────────────────────────────────

def _resolve_algo(experiment):
    base    = experiment.recon_algo or 'FBP'
    use_gpu = experiment.recon_gpu if experiment.recon_gpu is not None else True
    if isinstance(use_gpu, str):
        use_gpu = use_gpu.lower() == 'true'
    return f"{base}_CUDA" if use_gpu else base


def _build_algo_params(experiment):
    base   = experiment.recon_algo or 'FBP'
    params = {}
    if base == 'FBP':
        params['filter_type'] = experiment.recon_filter_type or 'ram-lak'
    elif base in ('SIRT', 'SART', 'CGLS'):
        params['iterations'] = int(experiment.recon_iterations or 100)
        if base == 'SIRT':
            if experiment.recon_min_constraint is not None:
                params['min_constraint'] = float(experiment.recon_min_constraint)
            if experiment.recon_max_constraint is not None:
                params['max_constraint'] = float(experiment.recon_max_constraint)
    # pixel_size in meters → convert to mm for ASTRA geometry scaling
    # (ASTRA output will then be in mm⁻¹ instead of 1/pixel)
    try:
        pixel_m = float(experiment.pixel)
        if pixel_m > 0:
            params['pixel_size_mm'] = pixel_m * 1e3
    except (TypeError, ValueError):
        pass
    return params


# ─── COR test functions ───────────────────────────────────────────────────────

def call_standard_cor_test(experiment, viewer, widget):
    print("Starting Standard COR test.")
    dialog = create_processing_dialog(viewer.window._qt_window)
    try:
        experiment.update_parameters(widget, parameters_to_update=[
            "sample_images", "slice_idx",
            "cor_min", "cor_max", "cor_step", "double_flatfield",
            "recon_algo", "recon_gpu", "recon_filter_type", "recon_iterations",
            "recon_min_constraint", "recon_max_constraint"])
        slice_idx   = int(experiment.slice_idx)
        sigma       = float(experiment.sigma) if (widget.paganin_checkbox.isChecked()
                                                   and experiment.sigma) else 0
        coeff       = float(experiment.coeff) if (widget.paganin_checkbox.isChecked()
                                                   and experiment.coeff) else 0
        cor_range   = np.arange(int(experiment.cor_min),
                                int(experiment.cor_max) + int(experiment.cor_step),
                                int(experiment.cor_step))
        hdf5_path   = _get_hdf5_path(viewer, experiment)
        use_angles  = widget.angles_checkbox.isChecked()
        algo        = _resolve_algo(experiment)
        algo_params = _build_algo_params(experiment)

        # For COR test we only need one slice — use slice-only pipeline
        projs_slice, apply_unsharp = _get_slice_projs(
            viewer, experiment, widget, slice_idx)
    except Exception as e:
        print(f"Standard COR setup error: {e}")
        dialog.close()
        return

    def compute():
        projs_gpu     = cp.asarray(projs_slice)
        target_shape, slices = None, []
        for cor in tqdm(cor_range, desc="Standard COR"):
            sino   = pad_and_shift_projection(projs_gpu, cor)
            angles = (get_angles(hdf5_path, sino.shape[0]) if use_angles
                      else create_angles(sino, end=2 * np.pi))
            s = apply_mask_and_reconstruct(sino, angles, sigma, coeff,
                                           apply_unsharp=apply_unsharp,
                                           algo=algo, algo_params=algo_params)
            if target_shape is None:
                target_shape = s.shape
            slices.append(resize_to_target(s, target_shape))
        clear_memory([projs_gpu])
        return np.array(slices)

    def on_result(result):
        viewer.add_image(result, name="cor_test")
        dialog.close()

    def on_error(msg):
        print(f"Standard COR error: {msg}")
        dialog.close()

    widget._active_thread, widget._active_worker = run_in_thread(
        compute, on_result, on_error)


def call_find_global_cor(experiment, viewer, widget):
    print("Starting global COR calculation.")
    dialog = create_processing_dialog(viewer.window._qt_window)
    try:
        experiment.update_parameters(widget, parameters_to_update=[
            "sample_images", "slice_idx",
            "center_of_rotation", "acquisition_type", "double_flatfield"])
        sample_name = experiment.sample_images

        # Global COR needs the full volume
        projs_full, _ = _get_full_projs(viewer, experiment, widget)
    except Exception as e:
        print(f"Global COR setup error: {e}")
        dialog.close()
        return

    def compute():
        cor, plot_data = calc_cor(projs_full)
        return cor, plot_data

    def on_result(result):
        cor, plot_data = result
        cor = cor[np.isfinite(cor)]
        cor_mean = np.mean(
            cor[(cor > np.mean(cor) - np.std(cor)) & (cor < np.mean(cor) + np.std(cor))])
        widget.center_of_rotation_input.setText(str(cor_mean))
        widget.plot_window = PlotWindow(plot_data, cor_values=cor)
        widget.plot_window.show()
        dialog.close()

    def on_error(msg):
        print(f"Global COR error: {msg}")
        dialog.close()

    widget._active_thread, widget._active_worker = run_in_thread(
        compute, on_result, on_error)


def call_half_cor_test(experiment, viewer, widget):
    print("Starting half COR test.")
    dialog = create_processing_dialog(viewer.window._qt_window)
    try:
        experiment.update_parameters(widget, parameters_to_update=[
            "sample_images", "slice_idx",
            "double_flatfield", "center_of_rotation", "cor_fenetre",
            "recon_algo", "recon_gpu", "recon_filter_type", "recon_iterations",
            "recon_min_constraint", "recon_max_constraint"])
        slice_idx   = int(experiment.slice_idx)
        sigma       = float(experiment.sigma) if (widget.paganin_checkbox.isChecked()
                                                   and experiment.sigma) else 0
        coeff       = float(experiment.coeff) if (widget.paganin_checkbox.isChecked()
                                                   and experiment.coeff) else 0
        cor_test    = float(experiment.center_of_rotation)
        cor_fenetre = int(experiment.cor_fenetre)
        cor_range   = range(int(cor_test) - cor_fenetre, int(cor_test) + cor_fenetre)
        hdf5_path   = _get_hdf5_path(viewer, experiment)
        use_angles  = widget.angles_checkbox.isChecked()
        algo        = _resolve_algo(experiment)
        algo_params = _build_algo_params(experiment)

        projs_slice, apply_unsharp = _get_slice_projs(
            viewer, experiment, widget, slice_idx)
    except Exception as e:
        print(f"Half COR setup error: {e}")
        dialog.close()
        return

    def compute():
        projs_gpu = cp.asarray(projs_slice)
        target_shape, slices = None, []
        for cor in tqdm(cor_range, desc="Half COR"):
            if use_angles:
                sino, angles = load_angles_and_create_sinograms(hdf5_path, projs_gpu, cor)
            else:
                sino   = create_sinogram_slice(projs_gpu, 2 * cor)
                angles = create_angles(sino, end=np.pi)
            s = apply_mask_and_reconstruct(sino, angles, sigma, coeff,
                                           apply_unsharp=apply_unsharp,
                                           algo=algo, algo_params=algo_params)
            if target_shape is None:
                target_shape = s.shape
            slices.append(resize_to_target(s, target_shape))
        clear_memory([projs_gpu])
        return np.array(slices)

    def on_result(result):
        viewer.add_image(result, name="cor_test")
        dialog.close()

    def on_error(msg):
        print(f"Half COR error: {msg}")
        dialog.close()

    widget._active_thread, widget._active_worker = run_in_thread(
        compute, on_result, on_error)


# ─── Slice-only projection helper (no full-volume paganin) ────────────────────

def _get_slice_projs(viewer, experiment, widget, slice_idx):
    """
    Return (projs_2d, apply_unsharp) for a SINGLE slice.
    Paganin is applied only to that slice — the full volume is never computed.
    Called from the main thread.
    """
    sample_name   = widget.sample_selection.currentText()
    use_paganin   = widget.paganin_checkbox.isChecked()
    use_double_ff = widget.double_flatfield_checkbox.isChecked()
    history       = _get_layer_history(viewer, sample_name)

    # Already has paganin → just extract slice
    if _history_has(history, 'paganin'):
        data = np.array(viewer.layers[sample_name].data)
        projs_2d = data[:, slice_idx, :] if data.ndim == 3 else data
        return projs_2d, False   # unsharp already applied upstream if needed

    # Has flat/dark → apply paganin on slice ± margin only
    if _history_has(history, 'flat_dark'):
        data = np.array(viewer.layers[sample_name].data)
        if use_paganin:
            params = _read_paganin_params(experiment)
            if params:
                energy, pixel, dist, db = params
                # Pass full 3D volume so paganin_filter_slice can use context rows
                projs_2d = paganin_filter_slice(
                    data if data.ndim == 3 else data[:, np.newaxis, :],
                    slice_idx, energy, pixel, dist, db
                )['paganin']
                if use_double_ff:
                    projs_2d = double_flatfield_correction(
                        projs_2d[np.newaxis]
                    )['double_flatfield_corrected'][0]
                return projs_2d, True
        projs_2d = data[:, slice_idx, :] if data.ndim == 3 else data
        if use_double_ff:
            projs_2d = double_flatfield_correction(
                projs_2d[np.newaxis]
            )['double_flatfield_corrected'][0]
        return projs_2d, False

    # Raw data → apply flat/dark then paganin on slice ± margin only
    sample, flat, dark = _read_raw(viewer, experiment, widget)
    projs_full = apply_flat_darkfield(sample, flat, dark)['preprocess']

    apply_unsharp = False
    if use_paganin:
        params = _read_paganin_params(experiment)
        if params:
            energy, pixel, dist, db = params
            # Pass full 3D volume so paganin_filter_slice can use context rows
            projs_2d = paganin_filter_slice(
                projs_full, slice_idx, energy, pixel, dist, db
            )['paganin']
            apply_unsharp = True
    else:
        projs_2d = projs_full[:, slice_idx, :]

    if use_double_ff:
        projs_2d = double_flatfield_correction(
            projs_2d[np.newaxis]
        )['double_flatfield_corrected'][0]

    return projs_2d, apply_unsharp


# ─── Full-volume projection helper ────────────────────────────────────────────

def _get_full_projs(viewer, experiment, widget):
    """
    Return (projs_3d, apply_unsharp).
    Uses the selected layer's metadata to avoid recomputing existing steps.
    Called from the main thread.
    """
    sample_name   = widget.sample_selection.currentText()
    use_paganin   = widget.paganin_checkbox.isChecked()
    use_double_ff = widget.double_flatfield_checkbox.isChecked()
    history       = _get_layer_history(viewer, sample_name)

    # Already has paganin → use directly
    if _history_has(history, 'paganin'):
        projs = np.array(viewer.layers[sample_name].data)
        if use_double_ff and not _history_has(history, 'double_flatfield'):
            projs = double_flatfield_correction(projs)['double_flatfield_corrected']
        return projs, False

    # Has flat/dark → apply paganin on top if needed
    if _history_has(history, 'flat_dark'):
        projs = np.array(viewer.layers[sample_name].data)
        apply_unsharp = False
        if use_paganin:
            params = _read_paganin_params(experiment)
            if params:
                energy, pixel, dist, db = params
                projs = paganin_filter(projs, energy, pixel, dist, db)['paganin']
                apply_unsharp = True
        if use_double_ff:
            projs = double_flatfield_correction(projs)['double_flatfield_corrected']
        return projs, apply_unsharp

    # Raw → full pipeline
    sample, flat, dark = _read_raw(viewer, experiment, widget)
    projs = apply_flat_darkfield(sample, flat, dark)['preprocess']
    apply_unsharp = False
    if use_paganin:
        params = _read_paganin_params(experiment)
        if params:
            energy, pixel, dist, db = params
            projs = paganin_filter(projs, energy, pixel, dist, db)['paganin']
            apply_unsharp = True
    if use_double_ff:
        projs = double_flatfield_correction(projs)['double_flatfield_corrected']
    return projs, apply_unsharp


# ─── Step-by-step pipeline with intermediate saves ────────────────────────────

def _run_full_pipeline_with_saves(experiment, viewer, widget, dialog,
                                   then_reconstruct=True):
    """
    Compute preprocess → paganin (if checked) → reconstruction (if requested),
    saving each step as a separate layer in the viewer.
    Auto-selects the latest layer in the sample selector after each step.
    """
    try:
        sample_name   = widget.sample_selection.currentText()
        use_paganin   = widget.paganin_checkbox.isChecked()
        use_double_ff = widget.double_flatfield_checkbox.isChecked()
        history       = _get_layer_history(viewer, sample_name)
        paganin_params = _read_paganin_params(experiment) if use_paganin else None

        # Determine what needs to be computed
        need_flat_dark = 'flat_dark' not in history
        need_paganin   = use_paganin and ('paganin' not in history) and (paganin_params is not None)

        # Read data for needed steps
        if need_flat_dark:
            sample, flat, dark = _read_raw(viewer, experiment, widget)
            raw_data = (sample, flat, dark)
            src_projs = None
        else:
            raw_data  = None
            src_projs = np.array(viewer.layers[sample_name].data)

    except Exception as e:
        print(f"Pipeline setup error: {e}")
        dialog.close()
        return None, None, None

    return (need_flat_dark, need_paganin, paganin_params,
            raw_data, src_projs, sample_name,
            use_paganin, use_double_ff, history)


# ─── Preprocessing ────────────────────────────────────────────────────────────

def call_preprocess(experiment, viewer, widget):
    dialog = create_processing_dialog(viewer.window._qt_window)
    try:
        experiment.update_parameters(
            widget,
            parameters_to_update=["sample_images", "darkfield", "flatfield", "bigdata"])
        sample, flat, dark = _read_raw(viewer, experiment, widget)
        bigdata       = experiment.bigdata
        sample_name   = experiment.sample_images
        src_history   = _get_layer_history(viewer, sample_name)
        hdf5_src_path = _get_source_hdf5_path(viewer, sample_name)
    except Exception as e:
        print(f"Preprocess setup error: {e}")
        dialog.close()
        return

    def compute():
        return apply_flat_darkfield(sample, flat, dark)['preprocess']

    def on_result(result):
        if not bigdata:
            layer_name = f"preprocess_{sample_name}"
            meta = {
                'processing_history': src_history + ['flat_dark'],
                'source': sample_name,
                'hdf5_source_path': hdf5_src_path,
            }
            _save_layer(viewer, widget, result, layer_name, meta)
        clear_memory([sample, dark, flat])
        dialog.close()

    def on_error(msg):
        print(f"Preprocess error: {msg}")
        dialog.close()

    widget._active_thread, widget._active_worker = run_in_thread(
        compute, on_result, on_error)


def call_paganin(experiment, viewer, widget, one_slice=False):
    dialog = create_processing_dialog(viewer.window._qt_window)
    try:
        experiment.update_parameters(widget, parameters_to_update=[
            "sample_images", "slice_idx",
            "pixel", "dist_object_detector", "energy", "db", "sigma", "coeff"])
        params = _read_paganin_params(experiment)
        if params is None:
            raise ValueError("Paganin parameters (energy/pixel/dist/db) are not set.")
        energy, pixel, dist, db = params
        slice_idx     = int(experiment.slice_idx)
        sample_name   = experiment.sample_images
        src_history   = _get_layer_history(viewer, sample_name)
        hdf5_src_path = _get_source_hdf5_path(viewer, sample_name)

        # Use selected layer if already has flat/dark, else compute from raw
        if _history_has(src_history, 'flat_dark'):
            projs_ready  = np.array(viewer.layers[sample_name].data)
            raw          = None
            base_history = list(src_history)
        else:
            projs_ready  = None
            raw          = _read_raw(viewer, experiment, widget)
            base_history = src_history + ['flat_dark']
    except Exception as e:
        print(f"Paganin setup error: {e}")
        dialog.close()
        return

    def compute():
        p = projs_ready
        if p is None:
            sample, flat, dark = raw
            p = apply_flat_darkfield(sample, flat, dark)['preprocess']
        if one_slice:
            return paganin_filter_slice(p, slice_idx, energy, pixel, dist, db), base_history
        return paganin_filter(p, energy, pixel, dist, db)['paganin'], base_history

    def on_result(result):
        data, bh = result
        history = bh + [f'paganin(E={energy},px={pixel},dist={dist},db={db})']
        meta = {
            'processing_history': history,
            'source': sample_name,
            'hdf5_source_path': hdf5_src_path,
            'paganin': {'energy': energy, 'pixel': pixel, 'dist': dist, 'db': db},
        }
        layer_name = f"paganin_{sample_name}"
        _save_layer(viewer, widget, data, layer_name, meta)
        dialog.close()

    def on_error(msg):
        print(f"Paganin error: {msg}")
        dialog.close()

    widget._active_thread, widget._active_worker = run_in_thread(
        compute, on_result, on_error)


# ─── Process one slice ────────────────────────────────────────────────────────

def call_process_one_slice(experiment, viewer, widget):
    dialog = create_processing_dialog(viewer.window._qt_window)
    try:
        experiment.update_parameters(widget, parameters_to_update=[
            "sample_images", "center_of_rotation", "acquisition_type",
            "double_flatfield", "batch_size", "bigdata",
            "energy", "pixel", "dist_object_detector", "db",
            "sigma", "coeff",
            "recon_algo", "recon_gpu", "recon_filter_type", "recon_iterations",
            "recon_min_constraint", "recon_max_constraint"])
        slice_idx   = int(experiment.slice_idx)
        sigma       = float(experiment.sigma) if (widget.paganin_checkbox.isChecked()
                                                   and experiment.sigma) else 0
        coeff       = float(experiment.coeff) if (widget.paganin_checkbox.isChecked()
                                                   and experiment.coeff) else 0
        cor         = float(experiment.center_of_rotation)
        acq_type    = widget.acquisition_type_selection.currentIndex()
        use_angles  = widget.angles_checkbox.isChecked()
        hdf5_path     = _get_hdf5_path(viewer, experiment)
        algo          = _resolve_algo(experiment)
        algo_params   = _build_algo_params(experiment)
        source_name   = widget.sample_selection.currentText()
        hdf5_src_path = _get_source_hdf5_path(viewer, source_name)

        # Slice-only pipeline — no full-volume paganin
        projs_2d, apply_unsharp = _get_slice_projs(
            viewer, experiment, widget, slice_idx)
    except Exception as e:
        print(f"Process one slice setup error: {e}")
        dialog.close()
        return

    def compute():
        projs_gpu = cp.asarray(projs_2d)

        if acq_type == 0:  # Standard acquisition
            sinogram = pad_and_shift_projection(projs_gpu, cor)
            angles   = (get_angles(hdf5_path, sinogram.shape[0]) if use_angles
                        else create_angles(sinogram, end=2 * np.pi))
        else:              # Half acquisition
            if use_angles:
                sinogram, angles = load_angles_and_create_sinograms(
                    hdf5_path, projs_gpu, cor)
            else:
                sinogram = create_sinogram_slice(projs_gpu, 2 * cor)
                angles   = create_angles(sinogram, end=np.pi)

        slice_ = apply_mask_and_reconstruct(
            sinogram, angles, sigma, coeff, apply_unsharp=apply_unsharp,
            algo=algo, algo_params=algo_params)
        clear_memory([projs_gpu])
        return np.array(slice_)

    def on_result(result):
        meta = {
            'processing_history': _get_layer_history(viewer, source_name) + [
                f'recon({algo},cor={cor:.2f},slice={slice_idx})'
            ],
            'source': source_name,
            'hdf5_source_path': hdf5_src_path,
            'reconstruction': {
                'algo': algo, 'algo_params': algo_params,
                'cor': cor, 'slice': slice_idx,
                'sigma': sigma, 'coeff': coeff,
            },
        }
        viewer.add_image(result, name=f"slice_cor{cor:.1f}_{algo}", metadata=meta)
        dialog.close()

    def on_error(msg):
        print(f"Process one slice error: {msg}")
        dialog.close()

    widget._active_thread, widget._active_worker = run_in_thread(
        compute, on_result, on_error)


# ─── Process all slices ───────────────────────────────────────────────────────

def call_process_all_slices(experiment, viewer, widget):
    dialog = create_processing_dialog(viewer.window._qt_window)
    try:
        experiment.update_parameters(widget, parameters_to_update=[
            "sample_images", "center_of_rotation", "acquisition_type",
            "double_flatfield", "batch_size", "bigdata",
            "energy", "pixel", "dist_object_detector", "db",
            "sigma", "coeff",
            "recon_algo", "recon_gpu", "recon_filter_type", "recon_iterations",
            "recon_min_constraint", "recon_max_constraint"])
        cor         = float(experiment.center_of_rotation)
        sigma       = float(experiment.sigma) if (widget.paganin_checkbox.isChecked()
                                                   and experiment.sigma) else 0
        coeff       = float(experiment.coeff) if (widget.paganin_checkbox.isChecked()
                                                   and experiment.coeff) else 0
        acq_type    = widget.acquisition_type_selection.currentIndex()
        use_angles  = widget.angles_checkbox.isChecked()
        bigdata     = experiment.bigdata
        batch_size  = int(experiment.batch_size)
        hdf5_path   = _get_hdf5_path(viewer, experiment)
        use_paganin = widget.paganin_checkbox.isChecked()
        use_double_ff = widget.double_flatfield_checkbox.isChecked()
        algo        = _resolve_algo(experiment)
        algo_params = _build_algo_params(experiment)
        source_name   = widget.sample_selection.currentText()
        history       = _get_layer_history(viewer, source_name)
        hdf5_src_path = _get_source_hdf5_path(viewer, source_name)
        paganin_params = _read_paganin_params(experiment) if use_paganin else None

        # Determine what needs to be computed
        has_flat_dark = _history_has(history, 'flat_dark')
        has_paganin   = _history_has(history, 'paganin')

        need_flat_dark = not has_flat_dark
        need_paganin   = use_paganin and (not has_paganin) and (paganin_params is not None)

        # Read source data
        if need_flat_dark:
            sample, flat, dark = _read_raw(viewer, experiment, widget)
        else:
            sample = np.array(viewer.layers[source_name].data)
            flat = dark = None

        # Big-data: prepare source layer ref
        if bigdata:
            try:
                source_layer = viewer.layers[source_name]
            except KeyError:
                source_layer = None

    except Exception as e:
        print(f"Process all slices setup error: {e}")
        dialog.close()
        return

    def compute():
        # ── Step 1: flat/dark correction ──────────────────────────────────
        projs = sample
        current_history = list(history)

        if need_flat_dark:
            projs = apply_flat_darkfield(projs, flat, dark)['preprocess']
            current_history = current_history + ['flat_dark']
            preprocess_result = (projs.copy(), list(current_history))
        else:
            preprocess_result = None

        # ── Step 2: Paganin setup ─────────────────────────────────────────
        apply_unsharp  = False
        paganin_result = None
        pag_margin     = 0

        if need_paganin:
            energy, pixel, dist, db = paganin_params
            apply_unsharp   = True
            pag_history     = current_history + [
                f'paganin(E={energy},px={pixel},dist={dist},db={db})'
            ]
            if acq_type == 0:
                # Standard acquisition: apply Paganin per-slice with margin
                # to avoid computing the full volume at once.
                pag_margin      = get_padding_size_slice(energy, pixel, dist)
                current_history = pag_history
                # paganin_result stays None — full volume never materialised
            else:
                # Half acquisition: sinograms span all slices, need full volume
                projs           = paganin_filter(projs, energy, pixel, dist, db)['paganin']
                current_history = pag_history
                paganin_result  = (projs.copy(), list(current_history))
        elif has_paganin:
            apply_unsharp = False   # already applied upstream

        # Apply double flat-field now unless we'll do it per-slice (standard acq + paganin)
        if use_double_ff and not (need_paganin and acq_type == 0):
            projs = double_flatfield_correction(projs)['double_flatfield_corrected']

        # ── Step 3: Reconstruction ────────────────────────────────────────
        if bigdata:
            return ('bigdata', preprocess_result, paganin_result, projs,
                    current_history, apply_unsharp)

        n_slices = projs.shape[1]
        recon    = None

        if acq_type == 0:
            for i in tqdm(range(n_slices), desc="Reconstructing"):
                if need_paganin:
                    # Paganin on slice ± margin only
                    s = max(0, i - pag_margin)
                    e = min(n_slices, i + pag_margin + 1)
                    pag_chunk = paganin_filter(
                        projs[:, s:e, :], energy, pixel, dist, db
                    )['paganin']
                    proj_2d = pag_chunk[:, i - s, :]
                    if use_double_ff:
                        proj_2d = double_flatfield_correction(
                            proj_2d[np.newaxis]
                        )['double_flatfield_corrected'][0]
                    proj_gpu = cp.asarray(proj_2d)
                else:
                    proj_gpu = cp.asarray(projs[:, i, :])

                sino   = pad_and_shift_projection(proj_gpu, cor)
                angles = (get_angles(hdf5_path, sino.shape[0]) if use_angles
                          else create_angles(sino, end=2 * np.pi))
                slice_ = apply_mask_and_reconstruct(
                    sino, angles, sigma, coeff, apply_unsharp=apply_unsharp,
                    algo=algo, algo_params=algo_params)
                if recon is None:
                    h, w = slice_.shape
                    recon = np.zeros((n_slices, h, w), dtype=np.float32)
                recon[i] = slice_
        else:
            if use_angles:
                sino, angles = load_angles_and_create_sinograms(hdf5_path, projs, cor)
            else:
                sino   = create_sinogram(projs, 2 * cor)
                angles = create_angles(sino, end=np.pi)
            for i in tqdm(range(sino.shape[0]), desc="Reconstructing"):
                slice_ = apply_mask_and_reconstruct(
                    sino[i], angles, sigma, coeff, apply_unsharp=apply_unsharp,
                    algo=algo, algo_params=algo_params)
                if recon is None:
                    h, w = slice_.shape
                    recon = np.zeros((sino.shape[0], h, w), dtype=np.float32)
                recon[i] = slice_

        clear_memory([projs])
        return ('memory', preprocess_result, paganin_result, recon,
                current_history, apply_unsharp)

    def on_result(result):
        kind = result[0]
        preprocess_result = result[1]
        paganin_result    = result[2]
        final_data        = result[3]
        current_history   = result[4]

        # ── Save intermediate layers (not in bigdata mode) ────────────────
        if not bigdata:
            if preprocess_result is not None:
                pre_data, pre_history = preprocess_result
                pre_meta = {
                    'processing_history': pre_history,
                    'source': source_name,
                    'hdf5_source_path': hdf5_src_path,
                }
                _save_layer(viewer, widget, pre_data,
                            f"preprocess_{source_name}", pre_meta)

            if paganin_result is not None:
                pag_data, pag_history = paganin_result
                pag_meta = {
                    'processing_history': pag_history,
                    'source': source_name,
                    'hdf5_source_path': hdf5_src_path,
                    'paganin': dict(zip(
                        ('energy', 'pixel', 'dist', 'db'), paganin_params
                    )) if paganin_params else {},
                }
                _save_layer(viewer, widget, pag_data,
                            f"paganin_{source_name}", pag_meta)

        # ── Save reconstruction ────────────────────────────────────────────
        recon_history = current_history + [
            f'recon({algo},cor={cor:.2f},acq={"half" if acq_type else "std"})'
        ]
        recon_meta = {
            'processing_history': recon_history,
            'source': source_name,
            'hdf5_source_path': hdf5_src_path,
            'reconstruction': {
                'algo': algo, 'algo_params': algo_params,
                'cor': cor, 'sigma': sigma, 'coeff': coeff,
                'acquisition': 'half' if acq_type else 'standard',
            },
        }
        base_name = f"Vol_cor{cor:.1f}_{algo}"
        existing = {l.name for l in viewer.layers}
        recon_name = base_name
        idx = 1
        while recon_name in existing:
            recon_name = f"{base_name}_{idx}"
            idx += 1

        if kind == 'bigdata':
            e, px, d, db = paganin_params if paganin_params else (0, 0, 0, 0)
            dask_arr, h5_out, _ = process_volume_bigdata(
                source_layer     = source_layer,
                projs_in_memory  = final_data if not need_flat_dark else None,
                flat=flat, dark=dark,
                acq_type         = acq_type,
                cor              = cor,
                paganin_on       = need_paganin,
                energy=e, pixel=px, dist=d, db=db,
                double_ff        = use_double_ff,
                use_angles       = use_angles,
                hdf5_angles_path = hdf5_path,
                sigma=sigma, coeff=coeff,
                batch_size       = batch_size,
                algo             = algo,
                algo_params      = algo_params,
            )
            widget._bigdata_h5_out = h5_out
            viewer.add_image(dask_arr, name=f"bigdata_{recon_name}", metadata=recon_meta)
        else:
            viewer.add_image(final_data, name=recon_name, metadata=recon_meta)

        dialog.close()

    def on_result_safe(result):
        try:
            on_result(result)
        except Exception:
            import traceback
            print(f"Process all slices on_result error:\n{traceback.format_exc()}")
            dialog.close()

    def on_error(msg):
        print(f"Process all slices error: {msg}")
        dialog.close()

    widget._active_thread, widget._active_worker = run_in_thread(
        compute, on_result_safe, on_error)
