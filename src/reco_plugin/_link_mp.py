# Imports
from cupyx.scipy.ndimage import shift
import numpy as np
import cupy as cp
from tqdm import tqdm
from skimage.transform import resize
import h5py
import gc

# Local imports
from .processing.cor import *
from .processing.phase import paganin_filter, unsharp_mask, paganin_filter_slice
from .processing.process import apply_flat_darkfield, double_flatfield_correction
from .processing.reconstruction import (
    reconstruct_from_sinogram_slice, create_angles,
    create_disk_mask
)
from .processing.sinogram import create_sinogram, create_sinogram_slice
from .processing.angles import (
    find_angles_in_dataset, find_opposite_pairs_best_match,
    create_sinogram_slice_from_pairs, create_sinograms_from_pairs
)
from .utils.qt_helpers import create_processing_dialog, PlotWindow

from .utils.link_utils import (
    add_image_to_layer, clear_memory, get_projections, get_angles,
    apply_mask_and_reconstruct, convert_cor_to_shift, pad_and_shift_projection,
    resize_to_target, load_angles_and_create_sinograms
)

def try_paganin_filter(experiment, viewer, widget):
    """
    Apply the Paganin filter to the projections.
    """
    dialog = create_processing_dialog(viewer.window._qt_window)

    