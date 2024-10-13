import numpy as np
import logging
import re
from functools import reduce
from typing import Union


def _validate_rotation(matrix_shape: tuple, vf_shape: tuple) -> None:
    if not len(vf_shape) > 1:
        raise ValueError(f"Vector field must be at least two dimensional. Given field has shape {vf_shape}.")
    if len(matrix_shape) != 2:
        raise ValueError(f"Rotation matrix must be two-dimensional. Given matrix has shape {matrix_shape}.")
    elif matrix_shape[1] != vf_shape[0]:
        raise ValueError(f"Matrix dimensions do not match the number of components. Matrix has shape {matrix_shape},"
                         f" field has shape {vf_shape}.")
    if not matrix_shape[0] == matrix_shape[1]:
        logging.warning(f"Non-square rotation matrix, information will be lost.")


def rotx(theta: float, degrees: bool = True):
    theta = np.deg2rad(theta) if degrees else theta

    c, s = np.cos(theta), np.sin(theta)
    r = np.array([[1, 0, 0],
                  [0, c, -s],
                  [0, s, c]])
    return r


def roty(theta: float, degrees: bool = True):
    theta = np.deg2rad(theta) if degrees else theta

    c, s = np.cos(theta), np.sin(theta)
    r = np.array([[c, 0, s],
                  [0, 1, 0],
                  [-s, 0, c]])
    return r


def rotz(theta: float, degrees: bool = True):
    theta = np.deg2rad(theta) if degrees else theta

    c, s = np.cos(theta), np.sin(theta)
    r = np.array([[c, -s, 0],
                  [s, c, 0],
                  [0, 0, 1]])
    return r


def rot(theta: float, degrees: bool = True):
    theta = np.deg2rad(theta) if degrees else theta
    c, s = np.cos(theta), np.sin(theta)

    r = np.array([[+c, -s],
                  [+s, +c]])
    return r


def eul2rot(seq, *args, degrees: bool = True):
    """Returns a stack of rotation matrices following the sequence and the corresponding angles.

    Example for dual axis tomography:   eul2rot('zy',[0,30],np.linspace(1,180))
    Example for laminography:           eul2rot('yz',np.linspace(1,180),45)
    """
    seq_regex = re.compile("^[xyz]+$")
    if not (seq_regex.match(seq)):
        raise ValueError("Sequence must contain a combination of 'xyz' only.")

    if not degrees:
        args = [np.rad2deg(arg) for arg in args]

    len_seq = len(seq)
    n_args = len(args)

    if len(args) != len(seq):
        raise ValueError(f"Sequence of rotations must match input arguments. Sequence has {len_seq} operations but"
                         f" {n_args} were given.")
    all_rotations = np.meshgrid(*args, indexing='ij')
    table = np.array(all_rotations).T.reshape(-1, n_args, order='F')

    # creates a 2D table with all operations. Each row is one measurement orientation.
    rot_dict = {'x': rotx, 'y': roty, 'z': rotz}
    rot_all = map(lambda z: reduce(lambda x, y: x @ y, [rot_dict[axis](angle) for axis, angle in zip(seq, z)]), table)
    return np.array(list(rot_all)).squeeze()


def tomo_rot(angles: np.ndarray, tilts: Union[float, np.ndarray] = 0) -> np.ndarray:
    return eul2rot('zy', tilts, angles)


def lamni_rot(angles: np.ndarray, lamni_tilt: Union[float, np.ndarray] = 45, lamni_skew: np.ndarray = 0) -> np.ndarray:
    return eul2rot('yzx', angles, lamni_tilt, lamni_skew)


def rotate_field(vector_field: np.ndarray, rot_matrix: np.ndarray) -> np.ndarray:
    """Rotates the vector field according to the provided rotation matrix.

    The vector field must have the form (n, x, y, ..., z) where n indexes the component.

    :param vector_field:        N-dim vector field, shape (n, x, y, ..., z)
    :param rot_matrix:          Rotation matrix, shape (m, n)
    :return:                    Rotated N-dim vector field, shape (m, x, y, ..., z)
    """
    _validate_rotation(rot_matrix.shape, vector_field.shape)
    return np.einsum('ij,j...->i...', rot_matrix, vector_field)


def global_phase_rotation(vector_field: np.ndarray, rot_matrix: np.ndarray) -> np.ndarray:
    """Rotates the vector field according to the provided rotation matrix.

    The vector field must have the form (n, x, y, ..., z) where n indexes the component.

    :param vector_field:        N-dim vector field, shape (n, x, y, ..., z)
    :param rot_matrix:          Rotation matrix, shape (m, n)
    :return:                    Rotated N-dim vector field, shape (m, x, y, ..., z)
    """

    _validate_rotation(rot_matrix.shape, vector_field.shape)
    return np.einsum('ij,j...->i...', rot_matrix, vector_field)


def rotate_vector_field(vector_field: np.ndarray, rot_matrix: np.ndarray,
                        order: int = 1) -> np.ndarray:
    # add another dimension to the rotation matrix and send to scalar recons
    n_spatial_dims = vector_field.ndim - 1
    vector_field = global_phase_rotation(vector_field, rot_matrix)

    new_rot_matrix = np.zeros((n_spatial_dims + 1, n_spatial_dims + 1))
    new_rot_matrix[0, 0] = 1
    new_rot_matrix[1:, 1:] = rot_matrix[0:n_spatial_dims, 0:n_spatial_dims]

    return rotate_scalar_field(vector_field, new_rot_matrix, order=order)


def rotate_scalar_field(field: np.ndarray, rot_matrix: np.ndarray,
                        order: int = 1) -> np.ndarray:
    from scipy.ndimage import affine_transform
    if not (0 <= order <= 5):
        raise ValueError(f"Interpolation order must be an integer between 0 and 5. Given order is {order}.")
    vf_shape = np.asarray(field.shape)

    rot_matrix = rot_matrix.T  # the rotation matrix is transposed to match the affine_transform convention

    out_center = rot_matrix @ (vf_shape - 1) / 2
    in_center = (vf_shape - 1) / 2
    offset = in_center - out_center
    output = affine_transform(field, rot_matrix, offset=offset, order=order)
    return output
