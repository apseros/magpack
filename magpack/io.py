import numpy as np
from magpack import _ovf_reader


def save_vtk(filename: str, scalar_dict: dict) -> None:
    """Saves a dictionary of scalar fields to a VTK file.

    :param filename:        Name of output file.
    :param scalar_dict:     Dictionary of scalar fields with the same shape.
    """
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
    """Saves function arguments to a .mat file. Wrapper for scipy.io.savemat.

    :param filename:            Name of output file.
    :param data_dictionary:     Unpacked dictionary of data."""
    from scipy.io import savemat
    savemat(filename, data_dictionary)


def load_mat(filename: str) -> dict:
    """Loads a .mat file. Wrapper for scipy.io.loadmat.

    :param filename:    Name of input file.
    :return:            Dictionary of loaded variables."""
    from scipy.io import loadmat
    return loadmat(filename)


def load_ovf(filename: str) -> _ovf_reader.OVF:
    """Loads a .ovf file and return an OVF object.

    The magnetization can be accessed using OVF.magnetization and metadata using the OVF.properties.

    :param filename:    Name of input file.
    :return:            OVF object."""
    return _ovf_reader.OVF(filename)


def see_keys(data: dict, prefix: str = '') -> None:
    """Recursively prints keys of a dictionary. Useful for HDF5 files.

    :param data:    Data (e.g. an HDF5 file).
    :param prefix:  Prefix to prepend to keys."""
    try:
        keys = list(data.keys())
    except AttributeError:
        return None

    for j in keys:
        previous = prefix + j
        print(previous)
        see_keys(data[j], previous + '/')


def pil_save(img: np.array, filename: str, cmap: str = 'viridis', vmin: float = None, vmax: float = None,
             alpha: bool = False, alpha_thresh: int = 750, indexing: str = 'ij') -> None:
    """Saves a numpy array as a full resolution png file.

    :param img:             Array to be saved.
    :param filename:        Name of output file.
    :param cmap:            Matplotlib colormap name.
    :param vmin:            Lower bound for colorbar axis (defaults to minimum value in the img array).
    :param vmax:            Upper bound for colorbar axis (defaults to maximum value in the img array).
    :param alpha:           Option to make bright pixels (white) transparent.
    :param alpha_thresh:    Threshold value for transparency (max 765 = 255*3).
    :param indexing:        Indexing scheme (xy is for matplotlib convention, default is ij).
    """
    import matplotlib as mpl
    from PIL import Image

    vmin = img.min() if vmin is None else vmin
    vmax = img.max() if vmax is None else vmax

    img = np.flip(img.T, axis=0) if indexing == 'ij' else img

    img = np.clip(img, vmin, vmax)
    img = (img - vmin) / (vmax - vmin)
    c = mpl.colormaps[cmap]
    save_im = c(img) * 255
    if alpha:
        mask = np.sum(save_im, -1) >= alpha_thresh
        save_im[mask, -1] = 0
    save_im = np.uint8(save_im)
    save_im = Image.fromarray(save_im)
    save_im.save(filename)
