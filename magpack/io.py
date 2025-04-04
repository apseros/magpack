import logging
import numpy as np
from magpack import _ovf_reader
from pyevtk.vtk import VtkFile, VtkRectilinearGrid
import matplotlib as mpl
from PIL import Image
from scipy.io import savemat, loadmat


def save_vtk(filename, scalars=None, vectors=None, colors=None):
    """Saves data into a VTK file.

    Parameters
    ----------
    filename : str
        Name of output file.
    scalars : dict (optional)
        Dictionary of scalar fields.
    vectors : dict (optional)
        Dictionary of vector data. Vectors should be provided in the shape (3, ...).
    colors : dict (optional)
        Dictionary of color data. Colors should be provided in the shape (..., 3).
    """

    if all(x is None for x in (scalars, vectors, colors)):
        raise ValueError("At least one of scalars, vectors, or colors must not be None.")

    if scalars is None:
        scalars = {}
    if vectors is None:
        vectors = {}
    if colors is None:
        colors = {}

    # validate inputs
    scalar_shapes = [item.shape for item in scalars.values()]
    vector_shapes = [item.shape[1:] for item in vectors.values()]
    color_shapes = [item.shape[:-1] for item in colors.values()]

    # convert shapes to set and check if they are all the same
    shape = {*scalar_shapes, *vector_shapes, *color_shapes}
    if len(shape) != 1:
        raise ValueError("All scalars and vectors should have the same shape.")
    shape = shape.pop()

    def expand_dimensions(s, v, c):
        """Expands dimensions of scalars, vectors, and colors appropriately."""
        s = {k: v[..., np.newaxis] for k, v in s.items()}
        v = {k: v[..., np.newaxis] for k, v in v.items()}
        c = {k: v[np.newaxis, ...] for k, v in c.items()}
        return s, v, c

    if len(shape) == 1:
        scalars, vectors, colors = expand_dimensions(scalars, vectors, colors)
        shape = (*shape, 1)
    if len(shape) == 2:
        scalars, vectors, colors = expand_dimensions(scalars, vectors, colors)
        shape = (*shape, 1)

    ndim = len(shape)
    w = VtkFile(filename, VtkRectilinearGrid)
    w.openGrid(start=(0,) * ndim, end=tuple(dim - 1 for dim in shape))
    w.openPiece(start=(0,) * ndim, end=tuple(dim - 1 for dim in shape))

    # Point data (value is assigned to the edge points
    xx, yy, zz = [np.arange(dim + 1) for dim in shape]
    w.openData("Point", scalars=scalars.keys(), vectors=[*vectors.keys(), *colors.keys()])
    for scalar_field_name, scalar_field in scalars.items():
        logging.info(f"Writing {scalar_field_name=}")
        w.addData(scalar_field_name, scalar_field)

    for vector_field_name, vector_field in vectors.items():
        logging.info(f"Writing {vector_field_name=}")
        w.addData(vector_field_name, (vector_field[0], vector_field[1], vector_field[2]))

    for color_field_name, color_field in colors.items():
        logging.info(f"Writing {color_field_name=}")
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


def save_mat(filename, **data_dictionary):
    """Saves function arguments to a .mat file. Wrapper for scipy.io.savemat.

    Parameters
    ----------
    filename : str
        Name of output file.
    data_dictionary :
        Dictionary of data to save
    """
    savemat(filename, data_dictionary)


def load_mat(filename):
    """Loads a .mat file. Wrapper for scipy.io.loadmat.

    Parameters
    ----------
    filename : str
        Name of output file.

    Returns
    -------
    dict
        Dictionary with variables."""
    return loadmat(filename)


def load_ovf(filename):
    """Loads a .ovf file and returns an OVF object.

    Parameters
    ----------
    filename : str
        Name of output file.

    Returns
    -------
    OVF
        The magnetization can be accessed using OVF.magnetization and metadata using the OVF.properties.
    """
    return _ovf_reader.OVF(filename)


def see_keys(data, prefix=''):
    """Recursively prints keys of a dictionary. Useful for HDF5 files.

    Parameters
    ----------
    data : dict
        Dictionary to print.
    prefix : str (optional)
        Prefix to prepend to keys.
    """
    try:
        keys = list(data.keys())
    except AttributeError:
        return None

    for j in keys:
        previous = prefix + j
        print(previous)
        see_keys(data[j], previous + '/')


def save_image(img, filename, cmap='viridis', vmin=None, vmax=None, alpha=False, alpha_thresh=750, indexing='ij'):
    r"""Saves a numpy array as a full resolution png file.

    Parameters
    ----------
    img : np.ndarray
        Image to save.
    filename : str
        Name of output file.
    cmap : str (optional)
        Matplotlib colormap name.
    vmin : float (optional)
        Lower bound for colorbar axis (defaults to minimum value in the img array).
    vmax : float (optional)
        Upper bound for colorbar axis (defaults to maximum value in the img array).
    alpha : bool (optional)
        Option to make bright pixels (white) transparent.
    alpha_thresh : int (optional)
        Threshold value for transparency, maximum value is :math:`765 (=255\times3)`.
    indexing : {'ij', 'xy'} (optional)
        Indexing scheme (xy for matplotlib convention, default is ij).
    """
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


def white_to_alpha(image_path, output_path, tolerance=1):
    """Converts white or bright pixels of a png image to transparent.

    Parameters
    ----------
    image_path : str
        Path to image to convert.
    output_path : str
        Path to output image.
    tolerance : float (optional)
        Tolerance for transparency. With 0 tolerance only strictly white pixels become transparent.

    """
    img = Image.open(image_path)
    # Convert to RGBA if not already in RGBA mode
    tolerance = 1e-10 if tolerance <= 0 else tolerance

    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    data = np.array(img, dtype=float)
    intensities = np.sqrt(np.sum(data[..., :3] ** 2, axis=-1))  # convert rgb values to intensities
    max_intensity = np.sqrt(3) * 255
    delta_intensity = max_intensity - intensities

    # the degree of transparency is the relative intensity,
    alpha = np.minimum(delta_intensity / tolerance * np.sqrt(3), 255)

    output_img = np.concatenate([data[..., :3], alpha[..., np.newaxis]], axis=-1).astype(np.uint8)
    Image.fromarray(output_img).save(output_path)
