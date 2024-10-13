import numpy as np
from . import _ovf_reader


def save_vtk(filename: str, scalar_dict: dict) -> None:
    """Saves the scalar or vector field as a VTK file."""
    from pyevtk.hl import gridToVTK
    shapes = [v.shape for v in scalar_dict.values() if isinstance(v, np.ndarray)]
    if any(shapes[0] != shape for shape in shapes):
        raise ValueError("All scalar fields must have the same shape.")

    x_size, y_size, z_size = shapes[0]
    x = np.arange(x_size)
    y = np.arange(y_size)
    z = np.arange(z_size)
    gridToVTK(filename, x, y, z, pointData=scalar_dict)


def save_mat(filename: str, **data_dictionary) -> None:
    from scipy.io import savemat
    savemat(filename, data_dictionary)


def load_mat(filename: str) -> dict:
    from scipy.io import savemat
    savemat(filename, data_dictionary)

def load_ovf(filename: str) -> _ovf_reader.OVF:
    return _ovf_reader.OVF(filename)


def see_keys(data: dict, prefix: str = '') -> None:
    """Recursively prints keys of a dictionary. Useful for HDF5 files."""
    try:
        keys = list(data.keys())
    except AttributeError:
        return None

    for j in keys:
        previous = prefix + j
        print(previous)
        see_keys(data[j], previous + '/')


def pil_save(img: np.array, filename: str, cmap: str, vmin: float, vmax: float, alpha: bool = False,
             alpha_thresh: int = 750) -> None:
    """Saves an image as a PNG file."""
    import matplotlib as mpl
    from PIL import Image
    img = np.clip(img, vmin, vmax)
    img = (img - vmin) / (vmax - vmin)
    c = mpl.colormaps[cmap]
    save_im = c(img) * 255
    if alpha:
        mask = np.sum(save_im, -1) > alpha_thresh
        save_im[mask, -1] = 0
    save_im = np.uint8(save_im)
    save_im = Image.fromarray(save_im)
    save_im.save(filename)
