# Imports
from cupyx.scipy.ndimage import shift
import numpy as np
import cupy as cp
from skimage.transform import resize
import h5py
import gc
import os
import tempfile

from tqdm import tqdm

# Local imports
from ..processing.cor import *
from ..processing.phase import unsharp_mask, paganin_filter, get_padding_size_slice
from ..processing.process import apply_flat_darkfield, double_flatfield_correction
from ..processing.sinogram import create_sinogram_slice
from ..processing.reconstruction import (
    reconstruct_from_sinogram_slice, create_angles,
    create_disk_mask
)
from ..processing.angles import (
    find_angles_in_dataset, find_opposite_pairs_best_match,
    create_sinogram_slice_from_pairs, create_sinograms_from_pairs
)

def add_image_to_layer(results, img_name, viewer, metadata=None):
    for name, image in results.items():
        if isinstance(image, cp.ndarray):
            image = image.get()
        viewer.add_image(
            image.real,
            name=f"{name}_{img_name}",
            metadata=metadata or {},
        )

def clear_memory(variables):
    for var in variables:
        del var
    gc.collect()
    cp._default_memory_pool.free_all_blocks()

def get_projections(viewer, prefix, slice_idx=None, fallback_func=None):
    for layer in viewer.layers:
        if layer.name.startswith(prefix):
            try:
                key = prefix 
                return {key: layer.data}
            except Exception as e:
                print(f"Error retrieving data from layer {layer.name}: {e}")
    return fallback_func() if fallback_func else None

def get_angles(hdf5_path, shape, full=True):
    if hdf5_path and hasattr(h5py, 'File'):
        with h5py.File(hdf5_path, "r") as f:
            angles = np.radians(find_angles_in_dataset(f, shape))[0]
            if not full:
                angles = angles[:shape]
    else:
        angles = create_angles(np.empty((shape, shape)), end=2 * np.pi if full else np.pi)
    return angles

def apply_mask_and_reconstruct(
    sinogram, angles, sigma, coeff,
    apply_unsharp=False,
    algo='FBP_CUDA', algo_params=None,
):
    mask   = create_disk_mask(sinogram)
    slice_ = reconstruct_from_sinogram_slice(
        sinogram, angles, algo=algo, algo_params=algo_params) * mask
    if apply_unsharp:
        # gaussian_filter(mode='reflect') bleeds values into the corners,
        # so re-apply the mask after unsharp to keep the exterior at zero.
        slice_ = unsharp_mask(cp.asarray(slice_), sigma, coeff).get() * mask
    return slice_

def convert_cor_to_shift(cor, width):
    shift = width // 2 - cor
    return shift

def pad_and_shift_projection(projs, cor):
    shift_value = convert_cor_to_shift(cor, projs.shape[1])
    # ceil to ensure enough padding for sub-pixel (float) shifts
    pad_width = int(np.ceil(abs(shift_value)))

    padded_projs = cp.pad(projs, ((0, 0), (pad_width, pad_width)), mode='constant', constant_values=0)
    shifted = shift(padded_projs, (0, shift_value), order=1, mode='constant').get()

    return shifted

def resize_to_target(slice_, target_shape):
    if slice_.shape != target_shape:
        resized = resize(slice_, target_shape, mode='constant', anti_aliasing=True)
        return resized
    return slice_

def load_angles_and_create_sinograms(hdf5_path, projs, cor):
    if not hdf5_path:
        raise ValueError("No HDF5 path provided.")

    with h5py.File(hdf5_path, 'r') as hdf5_file:
        angles = np.radians(find_angles_in_dataset(hdf5_file, projs.shape[0])[0])
        pairs = find_opposite_pairs_best_match(angles)
        angles = angles[:pairs[-1][0] + 1]
        if projs.ndim != 2:
            sino = create_sinograms_from_pairs(projs, 2 * cor, pairs)
        else:
            sino = create_sinogram_slice_from_pairs(projs, 2 * cor, pairs)

    return sino, angles


def process_volume_bigdata(
    source_layer,
    projs_in_memory,
    flat, dark,
    acq_type, cor,
    paganin_on, energy, pixel, dist, db,
    double_ff,
    use_angles, hdf5_angles_path,
    sigma, coeff,
    batch_size=50,
    algo='FBP_CUDA',
    algo_params=None,
):
    """
    Process a full volume slice by slice in batches, writing the reconstruction
    directly to a temporary HDF5 file to avoid RAM overflow.

    Parameters
    ----------
    source_layer : napari Image layer
        Used to retrieve the HDF5 file path and dataset key when
        projs_in_memory is None (lazy loading from disk).
    projs_in_memory : np.ndarray or None
        Pre-processed projections already in RAM, shape (n_angles, n_slices, width).
        If None, data is read lazily from the HDF5 file referenced by source_layer.
    flat, dark : np.ndarray or None
        Flat/dark field arrays.  Only used when projs_in_memory is None.
    acq_type : int
        0 = standard acquisition, 1 = half acquisition.
    cor : int
        Center of rotation (pixels).
    paganin_on : bool
    energy, pixel, dist, db : float
        Paganin parameters.  Only used when projs_in_memory is None.
    double_ff : bool
    use_angles : bool
    hdf5_angles_path : str or None
        Path to the HDF5 file containing angle data (standard acquisition).
    sigma, coeff : float
        Unsharp mask parameters.
    batch_size : int
        Number of slices processed per batch.

    Returns
    -------
    dask_arr : dask.array.Array
        Lazy dask array pointing to the temp HDF5 file.
    h5_out : h5py.File
        Open HDF5 file handle (must stay alive as long as dask_arr is used).
    temp_h5_path : str
        Path to the temporary HDF5 file.
    """
    import dask.array as da

    # ------------------------------------------------------------------ #
    #  Determine the data source                                           #
    # ------------------------------------------------------------------ #
    if projs_in_memory is not None:
        n_angles, n_slices, width = projs_in_memory.shape
        def _load_range(start, end):
            return projs_in_memory[:, start:end, :]
        skip_preprocess = True        # already flat/dark/paganin corrected
    else:
        meta = source_layer.metadata if source_layer is not None else {}
        h5_source_path = (meta.get('paths') or [None])[0]
        dataset_key = meta.get('dataset_key')
        if not h5_source_path or not dataset_key:
            raise ValueError(
                "No HDF5 source found for this layer. "
                "Load data from a .nxs or .tdf file to use Big Data mode."
            )
        with h5py.File(h5_source_path, 'r') as f:
            n_angles, n_slices, width = f[dataset_key].shape
        def _load_range(start, end):
            with h5py.File(h5_source_path, 'r') as f:
                return f[dataset_key][:, start:end, :].astype(np.float32)
        skip_preprocess = False

    # ------------------------------------------------------------------ #
    #  Paganin margin (context rows needed for correct phase retrieval)   #
    # ------------------------------------------------------------------ #
    paganin_margin = 0
    if paganin_on and not skip_preprocess:
        paganin_margin = get_padding_size_slice(energy, pixel, dist)

    # ------------------------------------------------------------------ #
    #  Create temporary output HDF5 file                                  #
    # ------------------------------------------------------------------ #
    temp_dir = tempfile.mkdtemp(prefix="reco_bigdata_")
    temp_h5_path = os.path.join(temp_dir, "reconstruction.h5")
    print(f"[BigData] Temp output: {temp_h5_path}")
    print(f"[BigData] {n_slices} slices  |  batch_size={batch_size}")

    h5_out = h5py.File(temp_h5_path, 'w')
    recon_dataset = None
    n_batches = (n_slices + batch_size - 1) // batch_size

    # ------------------------------------------------------------------ #
    #  Main batch loop                                                     #
    # ------------------------------------------------------------------ #
    for batch_idx in tqdm(range(n_batches), desc="[BigData] Batches"):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, n_slices)

        # Add Paganin context margin when reading from HDF5
        load_start = max(0, batch_start - paganin_margin)
        load_end = min(n_slices, batch_end + paganin_margin)

        batch = _load_range(load_start, load_end)   # (n_angles, loaded, width)

        if not skip_preprocess:
            dark_batch = dark[load_start:load_end, :] if dark is not None else None
            flat_batch = flat[load_start:load_end, :] if flat is not None else None
            batch = apply_flat_darkfield(batch, flat_batch, dark_batch)['preprocess']

            if paganin_on:
                batch = paganin_filter(batch, energy, pixel, dist, db)['paganin']

            # Remove the context margin — keep only the target slices
            trim_start = batch_start - load_start
            batch = batch[:, trim_start: trim_start + (batch_end - batch_start), :]

        if double_ff:
            batch = double_flatfield_correction(batch)['double_flatfield_corrected']

        # -------------------------------------------------------------- #
        #  Reconstruct each slice in the batch                            #
        # -------------------------------------------------------------- #
        for i in range(batch_end - batch_start):
            proj_slice = cp.asarray(batch[:, i, :])   # (n_angles, width) on GPU

            if acq_type == 0:
                sino = pad_and_shift_projection(proj_slice, cor)
                angles = (get_angles(hdf5_angles_path, sino.shape[0])
                          if use_angles
                          else create_angles(sino, end=2 * np.pi))
            else:
                sino = create_sinogram_slice(proj_slice, 2 * cor)
                angles = create_angles(sino, end=np.pi)

            recon_slice = apply_mask_and_reconstruct(
                sino, angles, sigma, coeff, apply_unsharp=paganin_on,
                algo=algo, algo_params=algo_params,
            )   # numpy array (h, w)

            # Create the dataset on first write (shape depends on reconstruction)
            if recon_dataset is None:
                h, w = recon_slice.shape
                recon_dataset = h5_out.create_dataset(
                    'reconstruction',
                    shape=(n_slices, h, w),
                    dtype=np.float32,
                    chunks=(1, h, w),
                )

            recon_dataset[batch_start + i] = recon_slice

        # Free batch from RAM + GPU memory pool
        del batch
        gc.collect()
        cp._default_memory_pool.free_all_blocks()

    # ------------------------------------------------------------------ #
    #  Return a lazy dask array — h5_out must stay open                   #
    # ------------------------------------------------------------------ #
    dask_arr = da.from_array(h5_out['reconstruction'], chunks=(10, -1, -1))
    return dask_arr, h5_out, temp_h5_path