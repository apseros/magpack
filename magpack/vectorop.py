from typing import Union, Any

import numpy as np
from itertools import combinations

from numpy import ndarray, dtype, signedinteger

DEFAULT_INTERPOLATION_ORDER = 1


def scalar_gradient(scalar_field: np.ndarray) -> np.ndarray:
    """Compute the gradient of a scalar field.

    :param scalar_field: Scalar field to compute the gradient.
    :return:             (Vector) gradient of scalar field.
    """
    return np.stack(np.gradient(scalar_field))


def vector_laplacian(vector_field: np.ndarray) -> np.ndarray:
    """Compute the laplacian of a vector field (cartesian).

    :param vector_field: Input vector field.
    :return:             Vector laplacian of input field.
    """
    return np.stack([scalar_laplacian(i) for i in vector_field])


def scalar_laplacian(scalar_field: np.ndarray) -> np.ndarray:
    """Calculates the scalar Laplacian of a scalar field.

    :param scalar_field: Scalar field to calculate the Laplacian.
    :return: (Vector)    Laplacian of scalar field."""
    # get all spatial dimension gradients, with the first index being the component
    return divergence(scalar_gradient(scalar_field))


def divergence(vector_field: np.ndarray) -> np.ndarray:
    """Calculates the divergence of a vector field.

    :param vector_field: Input vector field.
    :return: (Vector)    Divergence of input field."""
    return np.add.reduce([np.gradient(vector_field[i], axis=i) for i in range(vector_field.shape[0])])


def levi_civita_nd(*args) -> np.ndarray:
    """Calculate N-Dimensional Levi Civita tensor

    :param args:    Dimensions of tensor.
    :return:        N-Dimensional Levi Civita tensor."""

    def __levi_civita(*sub_args) -> Union[ndarray[Any, dtype[Any]], signedinteger[Any]]:
        if len(sub_args) != len(set(sub_args)):
            return np.array(0)
        combs = combinations(reversed(sub_args), 2)
        signs = [np.sign(x - y) for x, y in combs]
        return np.prod(signs)

    vec_lc = np.vectorize(__levi_civita)
    return vec_lc(*args)


def _levi_civita(i: np.ndarray, j: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Calculates the value of (i,j,k) element of the Levi-Civita tensor.

    :param i: Index i of the array.
    :param j: Index j of the array.
    :param k: Index k of the array.
    :return:  Corresponding Levi-Civita tensor."""
    return (i - j) * (j - k) * (k - i) / 2


def vorticity(vector_field: np.ndarray) -> np.ndarray:
    r"""Calculates the magnetic vorticity vector field for the given vector field.

    :param vector_field: Three-dimensional vector field shaped (3, nx, ny, nz) for which the vorticity will be
                         calculated.
    :return:             Three-dimensional vorticity vector field shaped (3, nx, ny, nz).

    The magnetic vorticity, :math:`\mathbf{\Omega}`, of the magnetization vector field, :math:`\mathbf{m}`, is given by
    [1]_, [2]_:

    .. math::

        \Omega_i = \frac{1}{8\pi}\epsilon_{abc}\epsilon_{ijk}m_{i}\partial_{b}m_{j}\partial_{c}m_{k}

    References
    ----------
    .. [1] Papanicolaou, N. , Tomaras, T. N. Nucl. Phys. B 360, 2-3
    .. [2] Cooper, N. R. Phys. Rev. Lett. 82, 1554

    """
    epsilon = _levi_civita(*np.indices((3, 3, 3)))
    diffs = np.stack([np.gradient(vector_field[i]) for i in range(vector_field.shape[0])])
    vort = np.einsum('ijk,abc,jbxyz,kcxyz,ixyz->axyz', epsilon, epsilon, diffs, diffs, vector_field)
    return vort / (8 * np.pi)


def skyrmion_number(vector_field: np.ndarray) -> np.ndarray:
    r"""Calculates the skyrmion topological number.

    The skyrmion topological number for the vector field :math:`\mathbf{m}` is given by evaluating integral:

    .. math::

        \frac{1}{4\pi} \int \mathbf{m} \cdot \left( \frac{\partial \mathbf{m}}{\partial x} \times \frac{\partial
        \mathbf{m}}{\partial y} \right)\,\text{d}x \text{d}y


    :param vector_field:    The vector field shape (n, x, y) from which the skyrmion number will be calculated.
    :return:                The skyrmion number.
    """
    components = vector_field.shape[0]
    spatial_dimensions = vector_field.ndim - 1

    if components != 3:
        raise ValueError("Vector field must have 3 components.")
    if spatial_dimensions != 2:
        raise ValueError("Field must have 2 spatial dimensions. ")

    vector_field = normalize(vector_field)
    epsilon = _levi_civita(*np.indices((3, 3, 3)))

    # differentials for x and y components
    diffs = np.stack([np.gradient(vector_field[i]) for i in range(3)])
    sk_n = np.einsum('ixy,ijk,jxy,kxy->xy', vector_field, epsilon, diffs[:, 0], diffs[:, 1])

    return sk_n.sum() / (4 * np.pi)


def curl(vector_field: np.ndarray) -> np.ndarray:
    """Calculates the curl of a vector field.

    :param vector_field: Vector field shape (3, nx, ny, nz) from which the curl will be calculated.
    :return:             Curl of vector field."""

    epsilon = _levi_civita(*np.indices((3, 3, 3)))
    diffs = np.stack([np.gradient(vector_field[i]) for i in range(vector_field.shape[0])])
    return np.einsum('ijk,jkabc->iabc', epsilon, diffs)


def magnitude(vector_field: np.ndarray) -> np.ndarray:
    """Calculates the magnitude of a vector field.

    :param vector_field:    Vector field shape (3, nx, ny, nz).
    :return:                Magnitude of vector field."""

    return np.sqrt(np.sum(vector_field ** 2, axis=0))


def scale_range(values: np.ndarray, norm_list: list = None, mask_zero: bool = True) -> np.ndarray:
    """Scales all values in the array such that they lie within the specified range.

    :param values:      Array to be scaled.
    :param norm_list:   List of length 2 with lower and upper bound (defaults to [-1, 1]).
    :param mask_zero:   If True, zero values will remain zero..
    :return:            Scaled values."""
    if norm_list is None:
        norm_list = [-1, 1]
    norm_range = norm_list[1] - norm_list[0]

    vmin, vmax = np.min(values), np.max(values)
    vrange = vmax - vmin
    values = (values - vmin) / vrange * norm_range + norm_list[0]

    if mask_zero:
        mask = np.where(values == 0, 0, 1)
        values = values * mask
    return values


def normalize(vector_field: np.ndarray) -> np.ndarray:
    """Scales all vectors in the array to unit length.

    :param vector_field:    The vector field to normalize.
    :return:                The normalized vector field."""
    mag = magnitude(vector_field)
    return np.divide(vector_field, mag, where=mag != 0)


def cart2sph(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Converts cartesian coordinates to spherical coordinates.

    All arrays must have the same shape (a, b, c). The resulting array has shape (3, a, b, c) and can be unpacked to the
    radial, elevation and azimuthal components.

    :param x:   The x cartesian coordinates.
    :param y:   The y cartesian coordinates.
    :param z:   The z cartesian coordinates.
    :return:    Corresponding spherical coordinates."""

    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    th = np.arccos(z / r, where=r != 0)
    ph = np.arctan2(y, x, where=np.logical_and(x != 0, y != 0))
    return np.stack([r, th, ph])


def sph2cart(r: np.ndarray, th: np.ndarray, ph: np.ndarray) -> np.ndarray:
    """Converts spherical coordinates to cartesian coordinates.

    All arrays must have the same shape (x, y, z). The resulting array has shape (3, a, b, c) and can be unpacked to the
    x, y and z components.

    :param r:       The spherical radial coordinates.
    :param th:      The spherical elevation coordinates.
    :param ph:      The spherical azimuthal coordinates.
    :return:        Corresponding cartesian coordinates."""

    x = r * np.sin(th) * np.cos(ph)
    y = r * np.sin(th) * np.sin(ph)
    z = r * np.cos(th)
    return np.stack([x, y, z])


def cart2pol(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Converts 2D cartesian coordinates to polar coordinates.

    All arrays must have the same shape (a, b). The resulting array has shape (2, a, b) and can be unpacked to the
    radial and azimuthal components.

    :param x:   The x cartesian coordinates.
    :param y:   The y cartesian coordinates.
    :return:    Corresponding polar coordinates."""

    r = np.sqrt(x ** 2 + y ** 2)
    az = np.arctan2(y, x)
    return np.stack([r, az])


def angular_gradient(data: np.ndarray, axial=False) -> np.ndarray:
    """Calculates the angular gradient of a vector field.
    
    :param data:    The vector field to calculate the angular gradient from.
    :param axial:   True for orientation fields, False for directional fields.
    :return:        The angular gradient scalar field."""
    delta_theta = [angular_difference(data, np.roll(data, 1, axis=i), axial=axial) for i in range(1, data.ndim)]
    delta_theta = np.sum(np.array(delta_theta), axis=0) / 3
    return delta_theta


def angular_difference(in_a: np.ndarray, in_b: np.ndarray, axial=False) -> np.ndarray:
    """Calculates the angular difference between vectors at the same positions of array a and b.

    :param in_a:    The first vector field to compare.
    :param in_b:    The second vector field to compare.
    :param axial:   True for orientation fields, False for directional fields.
    :return:        The scalar field of angular difference between the two inputs."""

    if in_a.shape != in_b.shape:
        raise ValueError("Inputs must have the same shape.")
    mag_mul = magnitude(in_a) * magnitude(in_b)
    dot_product = np.sum(in_a * in_b, axis=0)
    dot_product = np.where(dot_product > mag_mul, mag_mul, dot_product)  # normalise
    if axial:
        dot_product = np.abs(dot_product)
    np.seterr(invalid='ignore')
    ang_diff = np.arccos(dot_product / mag_mul, where=mag_mul != 0, out=np.zeros_like(dot_product))
    # replace nans with 0
    ang_diff = np.where(np.isnan(ang_diff), 0, ang_diff)
    return ang_diff


def magnitude_difference(in_a: np.ndarray, in_b: np.ndarray, percent=True) -> np.ndarray:
    """Calculates the magnitude difference between vectors at the same positions of array a and b.

    :param in_a:    The first vector field to compare.
    :param in_b:    The second vector field to compare.
    :param percent: True to calculate percentage difference, False to calculate absolute difference.
    :return:        The scalar field of magnitude difference between the two inputs.
    """

    m_a = magnitude(in_a)
    m_b = magnitude(in_b)
    if in_a.shape != in_b.shape:
        raise ValueError("Inputs must have the same shape.")
    if percent:  # if percent is used then in_b is considered to be the reference
        return 100 * np.divide(m_a - m_b, m_b, where=m_b != 0, out=np.zeros_like(m_a))
    else:
        return m_a - m_b


def stokes_to_jones(stokes: np.ndarray) -> np.ndarray:
    """Converts Stokes polarization vector to Jones polarization vector.

    :param stokes:  The Stokes polarization vector, shape (4,).
    :return:        The Jones polarization vector.
    """
    dop = np.sqrt(np.sum(np.power(stokes[1:], 2)))
    horizontal = stokes[1] / dop
    diagonal = stokes[2] / dop
    circular = stokes[3] / dop

    a = np.sqrt((1 + horizontal) / 2)
    if a == 0:
        b = 1
    else:
        b = diagonal / (2 * a) - 1j * circular / (2 * a)
    return np.sqrt(dop) * np.array([a, b])


def jones_to_stokes(jones: np.ndarray) -> np.ndarray:
    """ Converts Jones polarization vector to Stokes polarization vector.

    :param jones:  The Jones polarization vector, shape (2,).
    :return:        The Stokes polarization vector."""
    jones = jones / np.sqrt(np.sum(np.abs(jones) ** 2))
    m = jones[0] * jones[1].conjugate()
    return np.array([1, np.abs(jones[0]) ** 2 - np.abs(jones[1]) ** 2, 2 * np.real(m), 2 * np.imag(m)])
