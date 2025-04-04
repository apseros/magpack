import numpy as np
import logging
import re
from functools import reduce
from scipy.ndimage import affine_transform


def _validate_rotation(matrix_shape, vf_shape):
    """Checks if the matrix shape is congruent with the shape of the vector field.

    Parameters
    ----------
    matrix_shape : tuple
        Shape of the vector field.
    vf_shape : tuple
        Shape of the vector field.
    """
    if not len(vf_shape) > 1:
        raise ValueError(f"Vector field must be at least two dimensional. Given field has shape {vf_shape}.")
    if len(matrix_shape) != 2:
        raise ValueError(f"Rotation matrix must be two-dimensional. Given matrix has shape {matrix_shape}.")
    elif matrix_shape[1] != vf_shape[0]:
        raise ValueError(f"Matrix dimensions do not match the number of components. Matrix has shape {matrix_shape},"
                         f" field has shape {vf_shape}.")
    if not matrix_shape[0] == matrix_shape[1]:
        logging.warning(f"Non-square rotation matrix, information will be lost.")


def rotx(theta, degrees=True):
    """Generates the 3x3 rotation matrix for a rotation about the x-axis.

    Parameters
    ----------
    theta : float
        Rotation angle in degrees.
    degrees : bool
        True for degrees, False for radians.

    Returns
    -------
    np.ndarray
        Rotation matrix with shape(3, 3).
    """
    theta = np.deg2rad(theta) if degrees else theta

    c, s = np.cos(theta), np.sin(theta)
    r = np.array([[1, 0, 0],
                  [0, c, -s],
                  [0, s, c]])
    return r


def roty(theta, degrees=True):
    """Generates the 3x3 rotation matrix for a rotation about the y-axis.

    Parameters
    ----------
    theta : float
        Rotation angle in degrees.
    degrees : bool
        True for degrees, False for radians.

    Returns
    -------
    np.ndarray
        Rotation matrix with shape(3, 3).
    """
    theta = np.deg2rad(theta) if degrees else theta

    c, s = np.cos(theta), np.sin(theta)
    r = np.array([[c, 0, s],
                  [0, 1, 0],
                  [-s, 0, c]])
    return r


def rotz(theta, degrees=True):
    """Generates the 3x3 rotation matrix for a rotation about the z-axis.

    Parameters
    ----------
    theta : float
        Rotation angle in degrees.
    degrees : bool
        True for degrees, False for radians.

    Returns
    -------
    np.ndarray
        Rotation matrix with shape(3, 3).
    """
    theta = np.deg2rad(theta) if degrees else theta

    c, s = np.cos(theta), np.sin(theta)
    r = np.array([[c, -s, 0],
                  [s, c, 0],
                  [0, 0, 1]])
    return r


def rot(theta, degrees=True):
    """Generates the 2x2 rotation matrix for an in-plane rotation.

    Parameters
    ----------
    theta : float
        Rotation angle in degrees.
    degrees : bool
        True for degrees, False for radians.

    Returns
    -------
    np.ndarray
        Rotation matrix with shape(2, 2).
    """
    theta = np.deg2rad(theta) if degrees else theta
    c, s = np.cos(theta), np.sin(theta)

    r = np.array([[+c, -s],
                  [+s, +c]])
    return r


def eul2rot(seq, *args, degrees=True, reverse_order=False):
    r"""Returns a stack of rotation matrices following the sequence and the corresponding angles.

    Parameters
    ----------
    reverse_order
    seq : str
        Sequence of rotation operations (first operation listed on the right, follows matrix multiplication rule).
    args : array_like
        Unpacked list of angles (or array of angles) matching the length of the sequence.
    degrees : bool
        True for degrees, False for radians.
    reverse_order : bool
        By default, the rightmost arguement is expanded first.

    Returns
    -------
    np.ndarray
        Stack of rotation matrices with shape(n, 3, 3). `n` is the product of the length of all args.

    See Also
    --------
    tomo_rot, lamni_rot

    Examples
    --------
    Dual-axis tomography: Describes full 180° rotations for each of the two tilts: 0° and 30°.

    >>> eul2rot('yz', np.linspace(1, 180), [0,30])

    Laminography: Full 360° rotation around an axis that is slanted by 45° around the y-axis.

    >>> eul2rot('zy', 45, np.linspace(1,360))

    The regular order of producing the rotation matrix array assumes that the first array should be traversed before
    moving to the second array. By setting ``reverse_order=True``, the last array is traversed first, before traversing
    the second to last array, and so on. The resulting operations describe the same rotations and geometry, but are
    ordered diffently.Example of the change of order shown below: (not true outputs)

    >>> eul2rot('yzx', [0, 1, 2], [0,30], 45)
    [[ 0  0 45]     # roty(0) @ rotz(0)  @ rotx(45)
     [ 1  0 45]     # roty(1) @ rotz(0)  @ rotx(45)
     [ 2  0 45]     # roty(2) @ rotz(0)  @ rotx(45)
     [ 3  0 45]     # roty(3) @ rotz(0)  @ rotx(45)
     [ 0 30 45]     # roty(0) @ rotz(30) @ rotx(45)
     [ 1 30 45]     # roty(1) @ rotz(30) @ rotx(45)
     [ 2 30 45]     # roty(2) @ rotz(30) @ rotx(45)
     [ 3 30 45]]    # roty(3) @ rotz(30) @ rotx(45)

    >>> eul2rot('yzx', [0, 1, 2], [0,30], 45, reverse_order=True)
    [[ 0  0 45]     # roty(0) @ rotz(0)  @ rotx(45)
     [ 0 30 45]     # roty(0) @ rotz(30) @ rotx(45)
     [ 1  0 45]     # roty(1) @ rotz(0)  @ rotx(45)
     [ 1 30 45]     # roty(1) @ rotz(30) @ rotx(45)
     [ 2  0 45]     # roty(2) @ rotz(0)  @ rotx(45)
     [ 2 30 45]     # roty(2) @ rotz(30) @ rotx(45)
     [ 3  0 45]     # roty(3) @ rotz(0)  @ rotx(45)
     [ 3 30 45]]    # roty(3) @ rotz(30) @ rotx(45)
    """
    seq_regex = re.compile("^[xyz]+$")
    if not (seq_regex.match(seq)):
        raise ValueError("Sequence must be a combination of 'xyz' only.")

    if not degrees:
        args = [np.rad2deg(arg) for arg in args]

    len_seq = len(seq)
    n_args = len(args)

    if len(args) != len(seq):
        raise ValueError(f"Sequence of rotations must match input arguments. Sequence has {len_seq} operations but"
                         f" {n_args} were given.")
    all_rotations = np.meshgrid(*args, indexing='ij')

    if not reverse_order:
        table = np.array(all_rotations).reshape(n_args, -1, order='F').T
    else:
        table = np.array(all_rotations).T.reshape(-1, n_args, order='F')

    # creates a 2D table with all operations. Each row is one measurement orientation.
    rot_dict = {'x': rotx, 'y': roty, 'z': rotz}
    rot_all = map(lambda z: reduce(lambda x, y: x @ y, [rot_dict[axis](angle) for axis, angle in zip(seq, z)]), table)
    return np.array(list(rot_all)).squeeze()


def tomo_rot(angles, tilts=0):
    r"""Generates a stack of rotation matrices describing tomography with tilting.

    Parameters
    ----------
    angles : np.ndarray
        Angles of rotation within each tilt series.
    tilts : array_like
        Tilt axis as a list of number (if single tilt axis).

    Returns
    -------
    np.ndarray
        Stack of rotation matrices shaped (n, 3, 3). `n` is the product of the size of `angles` and `tilts`.

    See Also
    --------
    eul2rot, lamni_rot
    """
    return eul2rot('yz', angles, tilts)


def lamni_rot(angles, lamni_tilt=45, lamni_skew=0):
    r"""Generates a stack of rotation matrices describing laminography.

    Parameters
    ----------
    angles : np.ndarray
        Angles of rotation within each tilt series.
    lamni_tilt : array_like
        Array of laminography tilts.
    lamni_skew : array_like
        Array of skewness angles in laminography.

    Returns
    -------
    np.ndarray
        Stack of rotation matrices shaped (n, 3, 3). `n` is the product of the size of `angles`, `lamni_tilt` and
        `lamni_skew`.

    See Also
    --------
    eul2rot, tomo_rot
    """
    return eul2rot('zxy', lamni_skew, lamni_tilt, angles, reverse_order=True)


def transform_field(vector_field, rot_matrix):
    """Rotates each element in the vector field according to the provided rotation matrix.

    The vector field must have the form (n, x, y, ..., z) where n indexes the component.

    Parameters
    ----------
    vector_field : np.ndarray
        N-dim vector field, shaped (n, x, y, ..., z).
    rot_matrix : np.ndarray
        Rotation matrix, shape (m, n).

    Returns
    -------
    np.ndarray
        Rotated N-dimensional vector field, shape (m, x, y, ..., z).
    """

    _validate_rotation(rot_matrix.shape, vector_field.shape)
    return np.einsum('ij,j...->i...', rot_matrix, vector_field)


def rotate_vector_field(vector_field, rot_matrix, order=1):
    """Rotates a vector field according to the provided rotation matrix.

    Parameters
    ----------
    vector_field : np.ndarray
        Vector field to be rotated, shape (3, x, y, z).
    rot_matrix : np.ndarray
        Rotation matrix, shape (3, 3).
    order : int
        Interpolation order (0 to 5, default is 1).

    Returns
    -------
    np.ndarray
        Rotated vector field, shape (3, x, y, z).

    See Also
    --------
    transform_field
    """
    # add another dimension to the rotation matrix and send to scalar recons
    n_spatial_dims = vector_field.ndim - 1
    vector_field = transform_field(vector_field, rot_matrix)

    new_rot_matrix = np.zeros((n_spatial_dims + 1, n_spatial_dims + 1))
    new_rot_matrix[0, 0] = 1
    new_rot_matrix[1:, 1:] = rot_matrix[0:n_spatial_dims, 0:n_spatial_dims]
    return rotate_scalar_field(vector_field, new_rot_matrix, order=order)


def rotate_scalar_field(field, rot_matrix, order=1):
    """Rotates a scalar field according to the provided rotation matrix.

    Parameters
    ----------
    field : np.ndarray
        Scalar field to be rotated, shape (x, y, z).
    rot_matrix : np.ndarray
        Rotation matrix, shape (3, 3).
    order : int
        Interpolation order (0 to 5, default is 1).

    Returns
    -------
    np.ndarray
        Rotated scalar field, shape (x, y, z).
    """
    if not (0 <= order <= 5):
        raise ValueError(f"Interpolation order must be an integer between 0 and 5. Given order is {order}.")
    vf_shape = np.asarray(field.shape)

    rot_matrix = rot_matrix.T  # the rotation matrix is transposed to match the affine_transform convention

    out_center = rot_matrix @ (vf_shape - 1) / 2
    in_center = (vf_shape - 1) / 2
    offset = in_center - out_center
    output = affine_transform(field, rot_matrix, offset=offset, order=order)
    return output
