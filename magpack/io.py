import numpy as np
from magpack import _ovf_reader
from typing import Optional


def save_vtk(filename: str, scalars: Optional[dict], vectors: Optional[dict], colors: Optional[dict]) -> None:
    """Saves data into a VTK file.

    :param filename:    Name of output file.
    :param scalars:     Dictionary of scalar fields with the same shape.
    :param vectors:     Dictionary of vector data.
    :param colors:      Dictionary of color data.
    :return:            None
    """

    from pyevtk.vtk import VtkFile, VtkRectilinearGrid
    # validate inputs
    scalar_shapes = [item.shape for item in scalars.values()]
    vector_shapes = [item.shape[-3:] for item in vectors.values()]
    color_shapes = [item.shape[:3] for item in colors.values()]

    # convert shapes to set and check if they are all the same
    shape = {*scalar_shapes, *vector_shapes, *color_shapes}
    if len(shape) != 1:
        raise ValueError('All scalars and vectors should have the same shape.')

    shape = shape.pop()
    w = VtkFile(filename, VtkRectilinearGrid)
    w.openGrid(start=(0, 0, 0), end=tuple(dim - 1 for dim in shape))
    w.openPiece(start=(0, 0, 0), end=tuple(dim - 1 for dim in shape))

    # Point data (value is assigned to the edge points
    xx, yy, zz = [np.arange(dim + 1) for dim in shape]
    w.openData("Point", scalars=scalars.keys(), vectors=[*vectors.keys(), *colors.keys()])
    for scalar_field_name, scalar_field in scalars.items():
        print(f"Writing {scalar_field_name=}")
        w.addData(scalar_field_name, scalar_field)

    for vector_field_name, vector_field in vectors.items():
        print(f"Writing {vector_field_name=}")
        w.addData(vector_field_name, (vector_field[0], vector_field[1], vector_field[2]))

    for color_field_name, color_field in colors.items():
        print(f"Writing {color_field_name=}")
        w.addData(color_field_name, (color_field[..., 0], color_field[..., 1], color_field[..., 2]))

    w.closeData("Point")

    # Coordinates of cell vertices
    w.openElement("Coordinates")
    w.addData("x_coordinates", xx)
    w.addData("y_coordinates", yy)
    w.addData("z_coordinates", zz)
    w.closeElement("Coordinates")

    w.closePiece()
    w.closeGrid()

    for scalar_field in scalars.values():
        w.appendData(scalar_field)
    for vector_field in vectors.values():
        w.appendData((vector_field[0], vector_field[1], vector_field[2]))
    for color_field in colors.values():
        w.appendData(tuple(np.ascontiguousarray(color_field[..., i]) for i in range(3)))
    w.appendData(xx).appendData(yy).appendData(zz)
    w.save()


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

    # in case of RGB data
    if img.ndim == 3 and img.shape[2] in [3, 4]:
        if img.max() <= 1:
            img = img * 255
        save_im = Image.fromarray(np.uint8(img))
        save_im.save(filename)
        return None

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
