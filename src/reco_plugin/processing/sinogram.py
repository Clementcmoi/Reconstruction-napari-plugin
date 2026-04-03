import numpy as np
import cupy as cp
from cupyx.scipy.ndimage import shift as cp_shift
from tqdm import tqdm

def apply_left_weighting(CoR: int) -> cp.ndarray:
    """
    Generate linear weights for the left part of the projections.
    CoR must be a positive integer.
    """
    weights = cp.linspace(0, 1, CoR)
    weights = weights[None, :]  # Expand for 2D
    return weights

def create_sinogram_slice(proj: cp.ndarray, CoR) -> np.ndarray:
    """
    Create a sinogram from a set of projections, applying left weighting.
    CoR can be a float for sub-pixel precision: the integer part determines
    the array geometry, the fractional part is applied as a sub-pixel shift.
    """
    theta, ny = proj.shape

    # Split into integer geometry and fractional sub-pixel correction
    CoR_int = int(round(CoR))
    frac = CoR - CoR_int          # signed fractional pixels

    effective_CoR = min(CoR_int, ny)
    weights = apply_left_weighting(effective_CoR)

    proj_copy = cp.copy(proj)
    proj_copy[:, :effective_CoR] *= weights

    sino = cp.zeros((theta // 2, 2 * ny - CoR_int))

    flip = proj_copy[:theta // 2, ::-1]
    sino[:, :ny] += flip
    sino[:, -ny:] += proj_copy[theta // 2:, :]

    # Apply fractional shift along the detector axis (axis=1)
    if abs(frac) > 1e-6:
        sino = cp_shift(sino, (0, frac), order=1, mode='nearest')

    return sino.get()

def create_sinogram(projs: np.ndarray, CoR) -> np.ndarray:
    """
    Create sinograms from a set of projections, processing one slice at a time on the GPU.
    CoR can be a float for sub-pixel precision.
    """
    sinos = []
    for i in tqdm(range(projs.shape[1]), desc="Creating sinograms"):
        sino = create_sinogram_slice(cp.asarray(projs[:, i, :]), CoR)
        sinos.append(sino)

    return np.array(sinos)