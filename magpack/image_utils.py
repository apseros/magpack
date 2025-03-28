from PIL import Image
import numpy as np
import scipy.ndimage as ndi
from matplotlib import pyplot as plt
from matplotlib.widgets import PolygonSelector
from scipy.signal.windows import tukey
from typing import Union, Optional
from magpack.structures import create_mesh
import magpack.io


def non_affine_transform(data: np.ndarray, matrix: np.ndarray, order: int = 1) -> np.ndarray:
    r"""Applies a non-affine transformation to the input image.

    The matrix describing the non-affine transformation is described as

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


def get_perspective_matrix(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    r"""Provides the non-affine matrix that maps four points on the source image to the destination image.

    The expressions describing the map between the source (x, y) and destination image (X, Y) are:

    .. math::
        X = \frac{m_{11}x + m_{12}y + m_{13}}{m_{31}x + m_{32}y + 1}, \quad
        Y = \frac{m_{21}x + m_{22}y + m_{23}}{m_{31}x + m_{32}y + 1}

    with eight unknowns, the matrix elements :math:`m_{11}, m_{12}, ..., m_{33}` can be re-labeled as
    :math:`m_{1}, m_{2}, ..., m_{8}`. By forming four pairs of simultaneous equations, these elements can be determined:

    .. math::
        X = m_{1}x + m_{2}y + m_{3} - m_{7}xX - m_{8}yX + 1 \\
        Y = m_{4}x + m_{5}y + m_{6} - m_{7}xY - m_{8}yY + 1

    The equations are solved using linear algebra.

    :param source:      Array of four pairs of coordinates from source. [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    :param destination: Array of four pairs of destination coordinates. [[x'1,y'1], [x'2,y'2], [x'3,y'3], [x'4,y'4]]
    :return:            Non-affine transformation matrix that maps the points from the source to the destination.
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


def remove_distortion(img: Union[np.ndarray, str], filename: str = None, show_result: bool = False,
                      margin: float = 0.05, order: int = 1) -> Optional[np.ndarray]:
    """Remove perspective distortion from an image.

    :param img:         Image to be transformed as numpy array or filename.
    :param filename:    Name of output file (Optional).
    :param show_result: Shows resulting image before returning array if True.
    :param margin:      Margin for polygon selector.
    :param order:       Order of non-affine transformation.
    :return:            Transformed image as numpy array or None if saved to file.
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
    if img.ndim == 2:
        img_de = non_affine_transform(img, m, order=order)
    else:
        img_de = np.stack([non_affine_transform(channel, m, order=order) for channel in img.transpose(2, 0, 1)],
                          axis=-1)
    if filename:
        magpack.io.save_image(img_de, filename)
        return None

    if show_result:
        plt.imshow(img_de.swapaxes(0, 1), origin='lower')
        plt.show()

    return img_de


def rgb2gray(rgb: np.ndarray) -> np.ndarray:
    """Converts RGB/RGBA data of shape (x, y, ..., 3) to grayscale.

    The output is in the same range as the input, so either [0,1] or [0,255] ranges work.
    Only 3 indices from the last dimension are used, the rest are discarded.

    :param rgb:     Numpy array of shape (x, y, ..., 3) to be converted
    :return:        Numpy array of shape (x, y, ...) grayscale data
    """
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def hls2rgb(hue: np.ndarray, lightness: np.ndarray, saturation: Union[np.ndarray, float]) -> np.ndarray:
    """Convert HLS values (Hue, Lightness, Saturation) to RGB values (Red, Green, Blue) for image plotting.

    :param hue:         Hue [0, 2pi].
    :param lightness:   Lightness [0, 1].
    :param saturation:  Saturation [0, 1].

    :returns:           Numpy array of size input.shape + (3,) with (r, g, b) values in the [0,255] range
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


def rgb2lab(r, g, b) -> np.ndarray:
    """Converts RGB of shape (..., 3) to LAB.

    :param r:         red value [0, 1].
    :param g:         green value [0, 1].
    :param b:         blue value [0, 1].
    :return:          Inpout arrays in Lab coordinates.
    """
    m1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                   [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]])
    temp = np.einsum('ij,j...->i...', m1, np.stack([r, g, b]))
    m2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                   [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]])
    return np.einsum('ij,j...->i...', m2, np.cbrt(temp))


def lab2rgb(lum, a, b) -> np.ndarray:
    """Converts LAB of shape (..., 3) to RGB.

    :param lum:     Lab lum value.
    :param a:       Lab a value.
    :param b:       Lab b value.
    :return:        Corresponding RGB array.
    """
    m1 = np.array([[1, +0.3963377774, 0.2158037573],
                   [1, -0.1055613458, -0.0638541728],
                   [1, -0.0894841775, -1.2914855]])
    temp = np.einsum('ij,j...->i...', m1, np.stack([lum, a, b]))
    m2 = np.array([[+4.0767416621, -3.3077115913, +0.2309699292],
                   [-1.2684380046, +2.6097574011, -0.3413193965],
                   [-0.0041960863, -0.7034186147, 1.7076147010]])
    return np.einsum('ij,j...->i...', m2, np.power(temp, 3))


def complex_color(z: np.ndarray, saturation=0.6, log=False) -> np.ndarray:
    """Applies complex domain coloring to a 3D vector field.

    :param z:           Input complex number.
    :param saturation:  0...1 for color saturation.
    :param log:         Logarithmic coloring according to the magnitude.
    :return:            RBG array with shape (input_shape, 3) for plotting."""
    radial = np.log(np.abs(z) + 1) if log else np.abs(z)
    hue = np.angle(z) + np.pi
    lightness = radial / np.max(radial)
    return hls2rgb(hue, lightness, saturation)


def fft(data: np.ndarray) -> np.ndarray:
    """Return the shifted fast Fourier transform for plotting.

    :param data:    Data to perform N-dimensional fast Fourier transform.
    :return:        Fourier transform of data with zero frequency component in the middle."""
    return np.fft.fftshift(np.fft.fftn(data))


def intensity_fft(data: np.ndarray) -> np.ndarray:
    """Return the intensity of a shifted fast Fourier transform for plotting.

    :param data:    Data to perform N-dimensional fast Fourier transform.
    :return:        Intensity of Fourier transform of data with zero frequency component in the middle."""
    return np.abs(fft(data))


def ifft(data: np.ndarray) -> np.ndarray:
    """Return the shifted inverse fast Fourier transform for plotting.

    :param data:    Data to perform N-dimensional inverse fast Fourier transform.
    :return:        Inverse Fourier transform of data with zero frequency component in the middle."""
    return np.fft.ifftn(np.fft.ifftshift(data))


def fourier_shell_correlation(img1: np.ndarray, img2: np.ndarray, half_bit: bool = True, ring_size=1,
                              window: float = 0.5) -> np.ndarray:
    r"""Computes the Fourier shell correlation between two images.

    :param img1:        First image.
    :param img2:        Second image.
    :param half_bit:    If true, use the half-bit threshold, otherwise use the one bit threshold.
    :param ring_size:   The size of the ring window to compute Fourier shell (larger leads to smoother FSC).
    :param window:      Tukey window option for non-periodic images.
    :return:            Fourier shell correlation curve.


    The Fourier shell correlation is given by th equation:

    .. math::

        C(r) = \frac{\Re\left\{\sum_{r_i \in r} F_1 (r_i) \cdot \sum_{r_i \in r} F_2 (r_i)^*\right\}}
        {\sqrt{\sum_{r_i \in r} |F_1 (r_i)|^2 \cdot \sum_{r_i \in r} |F_2 (r_i)|^2}}

    """
    shape = img1.shape
    if img2.shape != shape:
        raise ValueError('Images must have same shape.')

    if window:
        img1 = img1 * np.outer(*[tukey(dim, window) for dim in shape])
        img2 = img2 * np.outer(*[tukey(dim, window) for dim in shape])

    f1, f2 = fft(img1), fft(img2)
    max_dim = np.min(img1.shape) // 2

    num = np.real(f1 * f2.conj())
    f1 = np.abs(f1) ** 2
    f2 = np.abs(f2) ** 2
    r = np.floor(np.sqrt(np.sum(np.array(create_mesh(*img1.shape)) ** 2 / ring_size, axis=0)))

    fsc = np.zeros(max_dim)

    for ii in range(max_dim):
        mask = r == ii
        fsc[ii] = np.sum(num[mask]) / np.sqrt(np.sum(f1[mask]) * np.sum(f2[mask]))

    return fsc

