import numpy as np
from functools import reduce
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.widgets import PolygonSelector
from scipy.signal.windows import tukey
from magpack.structures import create_mesh
from scipy.special import gamma
import scipy.ndimage as ndi
import magpack.io


def non_affine_transform(data, matrix, order=1):
    r"""Applies a non-affine transformation to the input image.

    Parameters
    ----------
    data : np.ndarray
        The image to be transformed.
    matrix : np.ndarray
        The non-affine transformation matrix.
    order : int
        The interpolation order of the non-affine transformation (between 0 and 5).

    Returns
    -------
    np.ndarray
        The transformed image.

    Notes
    -----
    The matrix describing the non-affine transformation is given by:

    .. math::
        \begin{pmatrix} x' \\ y' \\ z' \end{pmatrix} =
        \begin{pmatrix} \text{ScaleX} & \text{SkewX} & \text{TransX} \\
         \text{SkewY} & \text{ScaleY} & \text{TransY} \\
         \text{PerspX} & \text{PerspY} & \text{Norm} \\
        \end{pmatrix} \begin{pmatrix} x \\ y \\ 1 \end{pmatrix}

    and then perspective in the final image is achieved through

    .. math::
        X = x' / z',\quad
        Y = y' / z'

    See Also
    --------
    get_perspective_matrix, remove_distortion

    """
    if matrix.ndim != 2:
        raise RuntimeError("Transformation matrix must be two-dimensional.")
    if matrix.shape[0] != matrix.shape[1]:
        raise RuntimeError("Transformation matrix must be square")
    if data.ndim != 2:
        raise RuntimeError("Image must be two-dimensional.")
    shape = data.shape

    if matrix.shape[0] == data.ndim:
        new_mat = np.eye(data.ndim + 1)
        new_mat[0:2, 0:2] = matrix
        mat = new_mat
    else:
        mat = matrix

    xx, yy = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')

    coords = np.stack([xx, yy, np.ones_like(xx)], axis=0)  # Shape (3, H, W)
    transformed = mat @ coords.reshape(3, -1)  # Matrix multiplication (3, H*W)
    x_, y_, z_ = transformed.reshape(3, *shape)

    x__ = np.divide(x_, z_, where=z_ != 0)
    y__ = np.divide(y_, z_, where=z_ != 0)

    return ndi.map_coordinates(data, [x__, y__], order=order)


def get_perspective_matrix(source, destination):
    r"""Provides the non-affine matrix that maps four points on the source image to the destination image.

    Parameters
    ----------
    source : array_like, tuple
        Array of four pairs of coordinates from source. ``[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]``
    destination : array_like, tuple
        Array of four pairs of destination coordinates. ``[[x'1,y'1], [x'2,y'2], [x'3,y'3], [x'4,y'4]]``

    Returns
    -------
    np.ndarray
        Non-affine transformation matrix that maps the points from the source to the destination.

    Notes
    -----
    The expressions describing the map between the source (x, y) and destination image (X, Y) are:

    .. math::
        X = \frac{m_{11}x + m_{12}y + m_{13}}{m_{31}x + m_{32}y + 1}, \quad
        Y = \frac{m_{21}x + m_{22}y + m_{23}}{m_{31}x + m_{32}y + 1}

    with eight unknowns, the matrix elements :math:`m_{11}, m_{12}, ..., m_{33}` can be re-labeled as
    :math:`m_{1}, m_{2}, ..., m_{8}`. By forming four pairs of simultaneous equations, these elements can be determined:

    .. math::
        X = m_{1}x + m_{2}y + m_{3} - m_{7}xX - m_{8}yX + 1 \\
        Y = m_{4}x + m_{5}y + m_{6} - m_{7}xY - m_{8}yY + 1

    The equations are solved using linear algebra to calculate the perspective matrix.

    See Also
    --------
    non_affine_transform, remove_distortion

    """
    if type(source) == list:
        source = np.array(source)
    if type(destination) == list:
        destination = np.array(destination)

    a = np.zeros((8, 8))
    b = np.zeros(8)
    if destination.shape != (4, 2) or source.shape != (4, 2):
        raise ValueError("Expected 4 pairs of coordinates to map.")

    a[:4, 0] = a[4:, 3] = destination[:, 0]  # x scale, y skew
    a[:4, 1] = a[4:, 4] = destination[:, 1]  # y scale, x skew
    a[:4, 2] = a[4:, 5] = 1  # transpose x, transpose y
    a[:4, 6] = -source[:, 0] * destination[:, 0]
    a[:4, 7] = -source[:, 0] * destination[:, 1]
    a[4:, 6] = -source[:, 1] * destination[:, 0]
    a[4:, 7] = -source[:, 1] * destination[:, 1]

    b[:4] = source[:, 0]
    b[4:] = source[:, 1]

    # solve for the 8 parameters then add normalizing 1
    output = np.reshape(np.hstack([np.linalg.inv(a).dot(b), 1]), (3, 3))

    return output


def remove_distortion(img, filename=None, show_result=False, margin=0.05, order=1):
    """Remove perspective distortion from an image.

    Parameters
    ----------
    img : np.ndarray, str
        Image to be transformed as numpy array or filename.
    filename : str, optional
        Name of output file (Optional).
    show_result : bool
        Shows resulting image before returning array if True.
    margin : float, optional
        Margin for polygon selector.
    order:
        Order of non-affine transformation.

    Returns
    -------
    np.ndarray, optional
        Transformed image as numpy array or None if saved to file.

    See Also
    --------
    get_perspective_matrix, remove_distortion
    """
    if type(img) == str:
        img = np.asarray(Image.open(img))

    # initial position of selector
    x_low, x_high, y_low, y_high = np.outer(img.shape[:2], [margin, 1 - margin]).flatten()
    initial_vert = [(x_low, y_low), (x_low, y_high), (x_high, y_high), (x_high, y_low)]

    fig, ax = plt.subplots()
    ax.imshow(img.swapaxes(0, 1), origin='lower')
    selector = PolygonSelector(ax, lambda *args: None, props={'color': 'red'})
    selector.verts = initial_vert
    plt.show()

    src = np.array(selector.verts)
    dest = np.array([(0, 0), (0, img.shape[1]), (img.shape[0], img.shape[1]), (img.shape[0], 0)])

    m = get_perspective_matrix(src, dest)

    if img.ndim == 2:  # for greyscale image
        img_de = non_affine_transform(img, m, order=order)
    else:
        img_de = np.stack([non_affine_transform(color, m, order=order) for color in img.transpose(2, 0, 1)], axis=-1)
    if filename:
        magpack.io.save_image(img_de, filename)
        return None

    if show_result:
        plt.imshow(img_de.swapaxes(0, 1), origin='lower')
        plt.show()

    return img_de


def rgb2gray(rgb):
    """Converts RGB/RGBA data of shape (x, y, ..., 3) to grayscale.

    The output is in the same range as the input (e.g. [0,1] or [0,255]). Only the first 3 indices from the last
    dimension are used.

    Parameters
    ----------
    rgb : np.ndarray
        Numpy array of shape (x, y, ..., 3) to be converted

    Returns
    -------
    gray : np.ndarray
        Numpy array of shape (x, y, ...) grayscale data
    """
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def hls2rgb(hue, lightness, saturation):
    """Convert HLS values (Hue, Lightness, Saturation) to RGB values (Red, Green, Blue) for plotting.

    Parameters
    ----------
    hue, lightness, saturation : array_like
        Values for the hue [0, 2pi], lightness [0, 1] and saturation [0, 1] to be converted to RGB.

    Returns
    -------
    np.ndarray
        Numpy array of size (input.shape, 3) with RGB values in the 0 to 255 range.
    """
    hue = hue % (2 * np.pi)
    section = np.pi / 3
    c = (1 - np.abs(2 * lightness - 1)) * saturation
    x = c * (1 - np.abs((hue / section) % 2 - 1))
    m = lightness - c / 2

    c, x = c + m, x + m

    sextant = hue // section % 6
    result = np.where(sextant == 0, [c, x, m], 0) + np.where(sextant == 1, [x, c, m], 0) + \
             np.where(sextant == 2, [m, c, x], 0) + np.where(sextant == 3, [m, x, c], 0) + \
             np.where(sextant == 4, [x, m, c], 0) + np.where(sextant == 5, [c, m, x], 0)

    result *= 255
    return np.moveaxis(result, 0, -1).astype(np.uint8)


def complex_color(z, saturation=0.6, log=False):
    """Applies complex domain coloring to a 3D vector field.

    Parameters
    ----------
    z : np.ndarray
        Input complex number.
    saturation : float
        Color saturation value between 0...1.
    log : bool
        Boolean option for logarithmic coloring according to the magnitude.

    Returns
    -------
    np.ndarray
        RBG array with shape (input_shape, 3) for plotting."""
    radial = np.log(np.abs(z) + 1) if log else np.abs(z)
    hue = np.angle(z) + np.pi
    lightness = radial / np.max(radial)
    return hls2rgb(hue, lightness, saturation)


def fft(data):
    """Returns the shifted fast Fourier transform for plotting.

    Parameters
    ----------
    data : np.ndarray
        Data to perform N-dimensional fast Fourier transform.

    Returns
    -------
    np.ndarray
        Fourier transform of data with zero frequency component in the middle.

    See Also
    --------
    ifft, intensity_fft"""

    return np.fft.fftshift(np.fft.fftn(data))


def intensity_fft(data):
    """Returns the intensity of a shifted fast Fourier transform for plotting.

    Parameters
    ----------
    data : np.ndarray
        Data to perform N-dimensional fast Fourier transform.

    Returns
    -------
    np.ndarray
        Intensity of Fourier transform of data with zero frequency component in the middle.

    See Also
    --------
    fft, ifft"""
    return np.abs(fft(data))


def ifft(data):
    """Returns the shifted inverse fast Fourier transform for plotting.

    Parameters
    ----------
    data : np.ndarray
        Data to perform N-dimensional inverse fast Fourier transform.

    Returns
    -------
    np.ndarray
        Inverse Fourier transform of data with zero frequency component in the middle.

    See Also
    --------
    fft, intensity_fft"""

    return np.fft.ifftn(np.fft.ifftshift(data))


def fourier_shell_correlation(img1, img2, half_bit=True, window=0.5, rgb=True):
    r"""
    Compute the Fourier shell correlation (FSC) between two images.

    Parameters
    ----------
    img1, img2 : np.ndarray
        First and second image as ndarray types.
    half_bit : bool, optional
        If True, use the half-bit threshold; otherwise, use the one-bit threshold.
    window : float, optional
        Apply a Tukey window for non-periodic images.
    rgb : bool, optional
        If True, process images as RGB.

    Returns
    -------
    fsc : np.ndarray
        Fourier shell correlation curve.
    threshold : np.ndarray
        Threshold curve
    resolution : float
        Resolution in pixels

    Notes
    -----
    The Fourier shell correlation is computed using the following equation:

    .. math::
        C(r) = \frac{\Re\left\{\sum_{r_i \in r} F_1 (r_i) \cdot \sum_{r_i \in r} F_2 (r_i)^*\right\}}
        {\sqrt{\sum_{r_i \in r} |F_1 (r_i)|^2 \cdot \sum_{r_i \in r} |F_2 (r_i)|^2}}

    .. warning:: This function has not been tested for N-dimensional input.

    """

    if img2.shape != img1.shape:
        raise ValueError('Images must have same shape.')

    shape = img1.shape
    ndim = img1.ndim

    if not rgb:
        img1 = img1 * reduce(np.multiply.outer, [tukey(dim, window) for dim in shape])
        img2 = img2 * reduce(np.multiply.outer, [tukey(dim, window) for dim in shape])

    f1, f2 = fft(img1), fft(img2)
    max_dim = np.min(img1.shape) // 2 if not rgb else np.min(img1.shape[:-1]) // 2

    num = np.real(f1 * f2.conj())
    denom = np.sqrt(np.abs(f1) ** 2 * np.abs(f2) ** 2)

    rr = np.floor(np.hypot(*create_mesh(*shape)))

    def ring_sum(r):
        mask = rr == r
        return num[mask].sum() / denom[mask].sum()

    fsc = np.vectorize(ring_sum)(np.arange(max_dim))

    # threshold calculation
    snr = np.sqrt(2) if half_bit else 2
    snr_half = snr / 2 - 0.5
    snr_factor = 2 * np.sqrt(snr_half)

    # number of elements in the ring / shell is equal to 2πr, 4πr^2 etc
    # the first half accounts for the powers of r
    volume = np.sqrt(np.arange(1, max_dim + 1)) ** (ndim - 1) * (2 * np.pi ** (ndim / 2) / gamma(ndim / 2))

    thresh = (1 + snr_factor + snr_half * volume) / ((snr_half + 1) * volume + snr_factor)

    # find intersections
    idx = np.argwhere(np.diff(np.sign(fsc - thresh))).flatten()
    if not np.any(idx):
        resolution = 1
    else:
        resolution = 1 / (idx[0] / max_dim)
    return fsc, thresh, resolution
