import numpy as np

def apply_flat_darkfield(sample: np.ndarray, flatfield=None, darkfield=None) -> dict:
    """
    Apply flatfield and darkfield corrections to the sample data and return a dictionary.
    """
    if darkfield is not None:
        sample = sample - darkfield

    if flatfield is not None:
        sample = sample / (flatfield - darkfield if darkfield is not None else flatfield)

    return {"preprocess": sample}

def double_flatfield_correction(projs: np.ndarray) -> dict:
    """
    Apply a double flatfield correction to the projections and return a dictionary.
    """
    print("Applying double flatfield correction...")
    mean_proj = np.mean(projs, axis=0)
    mean_proj[mean_proj == 0] = 1e-6
    I_corr = projs / mean_proj
    return {"double_flatfield_corrected": I_corr}