import astra
import numpy as np
from skimage.draw import disk
import h5py

from math import pi

def create_angles(sinogram: np.ndarray, end: float = pi) -> np.ndarray:
    """
    Create angles for a sinogram.
    """
    if sinogram.ndim == 2:
        num_angles = sinogram.shape[0]  # Nombre d'angles pour un sinogramme 2D
    elif sinogram.ndim == 3:
        num_angles = sinogram.shape[1]  # Nombre d'angles pour un sinogramme 3D
    else:
        raise ValueError("Sinogram must be 2D or 3D.")

    angles = np.linspace(0, end, num_angles, endpoint=False)
    return angles

def find_angles_in_dataset(file, nz, group=None, path="", results=None, start_tol=10, end_tol=10):
    """
    Trouve tous les datasets 1D de longueur `nz`, commençant à ~0 et finissant à ~360.
    Retourne également les tableaux de valeurs correspondants.
    """
    if results is None:
        results = []

    if group is None:
        group = file

    for key in group:
        item = group[key]
        current_path = f"{path}/{key}"
        
        if isinstance(item, h5py.Group):
            find_angles_in_dataset(
                file, nz, group=item, path=current_path, results=results,
                start_tol=start_tol, end_tol=end_tol
            )
        elif isinstance(item, h5py.Dataset):
            if item.ndim == 1 and item.shape[0] == nz:
                try:
                    data = item[()]
                    if (
                        abs(data[0] - 0) <= start_tol and
                        abs(data[-1] - 360) <= end_tol
                    ):
                        results.append(data)
                except Exception as e:
                    print(f"Erreur lors de la lecture de {current_path} : {e}")
    return results

def degrees_to_radians(degrees: np.ndarray) -> np.ndarray:
    """
    Convert degrees to radians.
    """
    return np.radians(degrees)

# Iterative algorithms that accept a number of iterations
_ITERATIVE_ALGOS = ('SIRT_CUDA', 'SART_CUDA', 'CGLS_CUDA')


def reconstruct_from_sinogram_slice(
    sinogram: np.ndarray,
    angles: np.ndarray,
    algo: str = 'FBP_CUDA',
    algo_params: dict = None,
) -> np.ndarray:
    """
    Reconstruct a 2D slice from a sinogram using ASTRA Toolbox.

    Parameters
    ----------
    sinogram : (n_angles, width) array
    angles   : array of angles in radians
    algo     : ASTRA algorithm name – FBP_CUDA, BP_CUDA, FP_CUDA,
                                       SIRT_CUDA, SART_CUDA, CGLS_CUDA
    algo_params : dict with optional keys
        filter_type      – FBP_CUDA filter (default 'ram-lak')
        iterations       – iterative algorithms (default 100)
        relaxation       – SART_CUDA RelaxationFactor (default 1.0)
        min_constraint   – SIRT_CUDA lower bound (optional)
        max_constraint   – SIRT_CUDA upper bound (optional)
    """
    if algo_params is None:
        algo_params = {}

    # pixel_size_mm: physical size of one detector pixel in mm.
    # ASTRA output is in 1/pixel by default (detector_spacing=1).
    # Setting detector_spacing = pixel_size_mm makes the output directly in mm⁻¹.
    pixel_size_mm = float(algo_params.get('pixel_size_mm', 1.0))

    proj_geom = astra.create_proj_geom('parallel', pixel_size_mm, sinogram.shape[1], angles)
    vol_geom  = astra.create_vol_geom(sinogram.shape[1], sinogram.shape[1],
                                       -sinogram.shape[1] / 2 * pixel_size_mm,
                                        sinogram.shape[1] / 2 * pixel_size_mm,
                                       -sinogram.shape[1] / 2 * pixel_size_mm,
                                        sinogram.shape[1] / 2 * pixel_size_mm)

    sinogram_id = astra.data2d.create('-sino', proj_geom, sinogram)
    rec_id      = astra.data2d.create('-vol',  vol_geom)

    cfg = astra.astra_dict(algo)
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId']     = sinogram_id

    # ── Algorithm-specific options ─────────────────────────────────────────
    options = {}
    if algo in ('FBP_CUDA', 'FBP'):
        options['FilterType'] = algo_params.get('filter_type', 'ram-lak')
    elif algo in ('SIRT_CUDA', 'SIRT'):
        if 'min_constraint' in algo_params:
            options['MinConstraint'] = float(algo_params['min_constraint'])
        if 'max_constraint' in algo_params:
            options['MaxConstraint'] = float(algo_params['max_constraint'])

    if options:
        cfg['option'] = options

    alg_id = astra.algorithm.create(cfg)

    _iterative = ('SIRT_CUDA', 'SART_CUDA', 'CGLS_CUDA', 'SIRT', 'SART', 'CGLS')
    if algo in _iterative:
        astra.algorithm.run(alg_id, int(algo_params.get('iterations', 100)))
    else:
        astra.algorithm.run(alg_id)

    reconstruction = astra.data2d.get(rec_id)

    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)
    astra.data2d.delete(sinogram_id)

    return reconstruction

def create_disk_mask(sinogram: np.ndarray) -> np.ndarray:
    """
    Create a circular disk mask for the sinogram.
    """
    if sinogram.ndim == 2:
        _, width = sinogram.shape
    elif sinogram.ndim == 3:
        _, width = sinogram.shape[1:]
        
    disk_mask = np.zeros((width, width))
    rr, cc = disk((width // 2, width // 2), (width // 2))
    disk_mask[rr, cc] = 1

    return disk_mask

def reconstruct(one_slice, angles, center_of_rotation_px, pixel_size):
    """
    Reconstruction d'une tranche à partir des projections avec ASTRA et centre de rotation décalé.
    """
    import astra
    import numpy as np
    import matplotlib.pyplot as plt

    num_proj, det_count = one_slice.shape

    # Décalage réel entre centre de détecteur et centre de rotation
    shift = -(center_of_rotation_px - det_count / 2)
    print(shift)

    vectors = np.zeros((num_proj, 6), dtype=np.float32)

    for i in range(num_proj):
        theta = angles[i]  # ASTRA = sens horaire

        # 1. Direction du rayon
        ray_x = np.sin(theta)
        ray_y = -np.cos(theta)

        # 2. Centre du détecteur, tournant autour de (shift + center, 0)
        d_x = shift * np.cos(theta)
        d_y = shift * np.sin(theta)

        # 3. Vecteur détecteur (pixel 0 → 1)
        u_x = np.cos(theta)
        u_y = np.sin(theta)

        vectors[i, :] = [ray_x, ray_y, d_x, d_y, u_x, u_y]

    # Création des géométries ASTRA
    proj_geom = astra.create_proj_geom('parallel_vec', det_count, vectors)
    vol_size = 2 * int(det_count - center_of_rotation_px)  # Taille de la reconstruction
    vol_geom = astra.create_vol_geom(vol_size, vol_size)

    # Reconstruction
    proj_id = astra.data2d.create('-sino', proj_geom, one_slice)
    rec_id = astra.data2d.create('-vol', vol_geom)

    cfg = astra.astra_dict('FBP_CUDA')
    cfg['ProjectionDataId'] = proj_id
    cfg['ReconstructionDataId'] = rec_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)

    reconstruction = astra.data2d.get(rec_id)

    # Nettoyage
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(proj_id)
    astra.data2d.delete(rec_id)

    # Affichage
    plt.figure(figsize=(6, 6))
    plt.imshow(reconstruction, cmap='gray')
    plt.title(f"Reconstruction avec centre de rotation à {center_of_rotation_px:.2f} px")
    plt.show()

    return reconstruction * pixel_size * 1e6
