import cupy as cp
import numpy as np
from tqdm import tqdm

def process_try_center_of_rotation(self, widget):
    """
    Process the center of rotation based on the widget values.
    """
    print(f"Processing Center of Rotation")
    try:
        self.center_of_rotation = int(widget.center_of_rotation.value())
        print(f"Center of Rotation: {self.center_of_rotation}")
    except Exception as e:
        print(f"Error processing center of rotation: {e}")

def process_precise_local(self, widget):
    """
    Process the precise local based on the widget values.
    """
    print(f"Processing Precise Local")
    try:
        self.precise_local = int(widget.precise_local.value())
        print(f"Precise Local: {self.precise_local}")
    except Exception as e:
        print(f"Error processing precise local: {e}")

def calc_cor(projs: np.ndarray) -> tuple:
    """
    projs: projections [angles, hauteur, largeur]
    """
    if projs.ndim != 3:
        theta, nx = projs.shape
        ny = 1
    else :
        theta, ny, nx = projs.shape

    start = 0
    stop = ny
    step = 10
    cor = cp.zeros((stop - start + step - 1) // step, dtype=cp.float32)
    plot_data = []

    i = 0
    for y in tqdm(range(start, stop, step), desc="Recherche du COR par ligne"):
        sino1 = cp.asarray(projs[:theta // 2, y, ::-1])
        sino2 = cp.asarray(projs[theta // 2:, y, :])

        errors = cp.zeros(nx - 1, dtype=cp.float16)
        for shift in range(1, nx):
            t1 = sino1[:, -shift:]
            t2 = sino2[:, :shift]
            if t1.shape != t2.shape:
                continue
            mse = cp.mean((t1 - t2) ** 2)
            errors[shift - 1] = mse

        best_shift = cp.argmin(errors)
        plot_data.append(errors.get())
        cor[i] = (best_shift) / 2
        i += 1

    return cor.get(), plot_data