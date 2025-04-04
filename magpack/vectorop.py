import numpy as np
from itertools import combinations


def scalar_gradient(scalar_field):
    """Compute the gradient of a scalar field.

    Parameters
    ----------
    scalar_field : np.ndarray
        Scalar field to compute the gradient.

    Returns
    -------
    np.ndarray
        Gradient of scalar field (vector field).
    """
    return np.stack(np.gradient(scalar_field))


def vector_laplacian(vector_field):
    """Compute the laplacian of a vector field (cartesian).

    Parameters
    ----------
    vector_field : np.ndarray
        Input vector field.

    Returns
    -------
    np.ndarray
        Vector Laplacian of input field.
    """
    return np.stack([scalar_laplacian(i) for i in vector_field])


def scalar_laplacian(scalar_field):
    """Calculates the scalar Laplacian of a scalar field.

    Parameters
    ----------
    scalar_field : np.ndarray
        Input scalar field.

    Returns
    -------
    np.ndarray
        Scalar Laplacian of input field.
    """
    # get all spatial dimension gradients, with the first index being the component
    return divergence(scalar_gradient(scalar_field))


def divergence(vector_field):
    """Calculates the divergence of a vector field.

    Parameters
    ----------
    vector_field : np.ndarray
        Input vector field.

    Returns
    -------
        Divergence of input field (scalar field).
    """
    return np.add.reduce([np.gradient(vector_field[i], axis=i) for i in range(vector_field.shape[0])])


def levi_civita_nd(*args):
    """Calculate N-Dimensional Levi Civita tensor

    Parameters
    ----------
    args : array_like
        Dimensions of the Levi Civita tensor.

    Returns
    -------
    np.ndarray
        N-dimensional Levi Civita tensor.
    """

    def __levi_civita(*sub_args):
        """Calculation of the Levi Civita value for the member ijk... in the matrix.

        Parameters
        ----------
        sub_args : list of int | np.ndarray
            Indices of the current matrix member (i, j, k, ...)

        Returns
        -------
        np.ndarray
            Value of the Levi Civita tensor at the given indices.
        """
        if len(sub_args) != len(set(sub_args)):
            return np.array(0)
        combs = combinations(reversed(sub_args), 2)
        signs = [np.sign(x - y) for x, y in combs]
        return np.prod(signs)

    vec_lc = np.vectorize(__levi_civita)
    return vec_lc(*args)


def _levi_civita(i, j, k):
    """Calculates the value of (i,j,k) element of the Levi-Civita tensor.

    Parameters
    ----------
    i, j, k : np.ndarray
        Indices of the array.

    Returns
    -------
    np.ndarray
        Corresponding 3D Levi Civita tensor.
    """
    return (i - j) * (j - k) * (k - i) / 2


def vorticity(vector_field):
    r"""Calculates the magnetic vorticity vector field for the given vector field.

    Parameters
    ----------
    vector_field : np.ndarray
        Three-dimensional vector field shaped (3, nx, ny, nz) for which the vorticity will be calculated.

    Returns
    -------
    np.ndarray
        Three-dimensional vorticity vector field shaped (3, nx, ny, nz).

    Notes
    -----
    The magnetic vorticity, :math:`\mathbf{\Omega}`, of the magnetization vector field, :math:`\mathbf{m}`, is given by
    [1]_, [2]_:

    .. math::

        \Omega_i = \frac{1}{8\pi}\epsilon_{abc}\epsilon_{ijk}m_{i}\partial_{b}m_{j}\partial_{c}m_{k}

    .. warning:: This calculation is slow and can be improved by writing out the expression explicitly.

    References
    ----------
    .. [1] Papanicolaou, N. , Tomaras, T. N., *Nucl. Phys. B* **360**, 2-3
    .. [2] Cooper, N. R., *Phys. Rev. Lett.* **82**, 1554

    """
    epsilon = _levi_civita(*np.indices((3, 3, 3)))
    diffs = np.stack([np.gradient(vector_field[i]) for i in range(vector_field.shape[0])])
    v = np.einsum('ijk,abc,jbxyz,kcxyz,ixyz->axyz', epsilon, epsilon, diffs, diffs, vector_field)
    return v / (8 * np.pi)


def skyrmion_number(vector_field):
    r"""Calculates the skyrmion topological number.

    Parameters
    ----------
    vector_field : np.ndarray
        The vector field shape (n, x, y) from which the skyrmion number will be calculated.

    Returns
    -------
    np.ndarray
        The skyrmion topological number.

    Notes
    -----
    The skyrmion topological number for the vector field :math:`\mathbf{m}` is given by evaluating integral:

    .. math::

        \frac{1}{4\pi} \int \mathbf{m} \cdot \left( \frac{\partial \mathbf{m}}{\partial x} \times \frac{\partial
        \mathbf{m}}{\partial y} \right)\,\text{d}x \text{d}y

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


def curl(vector_field):
    """Calculates the curl of a vector field.

    Parameters
    ----------
    vector_field : np.ndarray
        Vector field shape (3, nx, ny, nz) from which the curl will be calculated.

    Returns
    -------
    np.ndarray
        Curl (vector field) of the input vector field.
    """

    epsilon = _levi_civita(*np.indices((3, 3, 3)))
    diffs = np.stack([np.gradient(vector_field[i]) for i in range(vector_field.shape[0])])
    return np.einsum('ijk,jkabc->iabc', epsilon, diffs)


def magnitude(vector_field):
    """Calculates the magnitude of a vector field.

    Parameters
    ----------
    vector_field : np.ndarray
        Vector field shape (3, nx, ny, nz)

    Returns
    -------
    np.ndarray
        Magnitude (scalar field) of the input vector field.
    """

    return np.sqrt(np.sum(vector_field ** 2, axis=0))


def scale_range(values, norm_list=None, mask_zero=True):
    """Scales all values in the array such that they lie within the specified range.

    Parameters
    ----------
    values : np.ndarray
        Array of values to scale.
    norm_list : list of float (optional)
        List of length 2 with lower and upper bound (defaults to [-1, 1]).
    mask_zero : bool (optional)
        If True, zero values will remain zero..

    Returns
    -------
    np.ndarray
        Scaled values.
    """
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


def normalize(vector_field):
    """Scales all vectors in the array to unit length.

    Parameters
    ----------
    vector_field : np.ndarray
        Vector field shape (3, nx, ny, nz)

    Returns
    -------
    np.ndarray
        Unit-normalized vector field.
    """
    mag = magnitude(vector_field)
    return np.divide(vector_field, mag, where=mag != 0, out=np.zeros_like(vector_field))


def cart2sph(x, y, z):
    """Converts cartesian coordinates to spherical coordinates.

    Parameters
    ----------
    x, y, z : np.ndarray
        The x, y and z cartesian coordinates.

    Returns
    -------
    np.ndarray
        The resulting spherical coordinates as (radius, elevation, azimuth).
    """

    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    th = np.arccos(z / r, where=r != 0)
    ph = np.arctan2(y, x, where=np.logical_and(x != 0, y != 0))
    return np.stack([r, th, ph])


def sph2cart(r, th, ph):
    """Converts spherical coordinates to cartesian coordinates.

    Parameters
    ----------
    r, th, ph : np.ndarray
        The radial, elevation and azimuthal spherical coordinates.

    Returns
    -------
    np.ndarray
        The resulting cartesian coordinates as (x, y, z).
    """

    x = r * np.sin(th) * np.cos(ph)
    y = r * np.sin(th) * np.sin(ph)
    z = r * np.cos(th)
    return np.stack([x, y, z])


def cart2pol(x, y):
    """Converts 2D cartesian coordinates to polar coordinates.

    Parameters
    ----------
    x, y : np.ndarray
        The x and y cartesian coordinates.

    Returns
    -------
    np.ndarray
        Corresponding polar coordinates as (r, theta).

    """

    r = np.sqrt(x ** 2 + y ** 2)
    az = np.arctan2(y, x)
    return np.stack([r, az])


def angular_gradient(data, axial=False):
    """Calculates the angular gradient of a vector field.

    Parameters
    ----------
    data : np.ndarray
        The vector field to calculate the angular gradient from.
    axial : bool (optional)
        True for orientation fields, False for directional fields.

    Returns
    -------
    np.ndarray
        The angular gradient (scalar field) of the input vector field.
    """
    delta_theta = [angular_difference(data, np.roll(data, 1, axis=i), axial=axial) for i in range(1, data.ndim)]
    delta_theta = np.sum(np.array(delta_theta), axis=0) / 3
    return delta_theta


def angular_difference(in_a, in_b, axial=False):
    """Calculates the angular difference between vectors at the same positions of array a and b.

    Parameters
    ----------
    in_a, in_b : np.ndarray
        The first and second vector fields to compare..
    axial : bool (optional)
        True for orientation fields, False for directional fields.

    Returns
    -------
    np.ndarray
        The angular difference (as a scalar field) between the two vector fields.
    """

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


def magnitude_difference(in_a, in_b, percent=True):
    """Calculates the magnitude difference between vectors at the same positions of array a and b.

    Parameters
    ----------
    in_a, in_b : np.ndarray
        The first and second vector fields to compare.
    percent : bool (optional)
        Set to True to calculate percentage difference, False to calculate absolute difference.

    Returns
    -------
    np.ndarray
        The magnitude difference (as a scalar field) between the two vector fields.
    """

    m_a = magnitude(in_a)
    m_b = magnitude(in_b)
    if in_a.shape != in_b.shape:
        raise ValueError("Inputs must have the same shape.")
    if percent:  # if percent is used then in_b is considered to be the reference
        return 100 * np.divide(m_a - m_b, m_b, where=m_b != 0, out=np.zeros_like(m_a))
    else:
        return m_a - m_b


def stokes_to_jones(stokes):
    """Converts Stokes polarization vector to Jones polarization vector.

    Parameters
    ----------
    stokes : np.ndarray
        Stokes polarization vector, shape (4,).

    Returns
    -------
    np.ndarray
        Jones polarization vector.

    See Also
    --------
    jones_to_stokes
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


def jones_to_stokes(jones):
    """ Converts Jones polarization vector to Stokes polarization vector.

    Parameters
    ----------
    jones : np.ndarray
        Jones polarization vector, shape (2,).

    Returns
    -------
    np.ndarray
        Stokes polarization vector.

    See Also
    --------
    stokes_to_jones
    """
    jones = jones / np.sqrt(np.sum(np.abs(jones) ** 2))
    m = jones[0] * jones[1].conjugate()
    return np.array([1, np.abs(jones[0]) ** 2 - np.abs(jones[1]) ** 2, 2 * np.real(m), 2 * np.imag(m)])
