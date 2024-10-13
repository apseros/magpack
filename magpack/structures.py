import logging
import numpy as np
from typing import Union
from magpack.vectorop import cart2pol, sph2cart, cart2sph
import itertools
import random


def checkerboard(shape):
    """Creates a checkerboard array of alternating +1 and -1 for the given shape."""
    return (np.indices(shape).sum(axis=0) % 2) * 2 - 1


def circ_mask(nx: int, ny: int) -> np.ndarray:
    """Defines a circular binary mask."""
    xx, yy = create_mesh(nx, ny)
    circ = np.sqrt(xx ** 2 + yy ** 2) < np.min([nx, ny]) // 2
    return circ


def vortex(nx: int, ny: int, winding: float = 1) -> np.ndarray:
    """Creates a 2D magnetic vortex."""
    xx, yy = create_mesh(nx, ny)
    rad, azimuth = cart2pol(yy, xx)

    mx = -np.cos(winding * azimuth)
    my = np.sin(winding * azimuth)
    mz = np.zeros_like(azimuth)

    return np.array([mx, my, mz])


def domain_generator(shape: list, points: int, cart: bool = False) -> tuple[np.ndarray, list[tuple]]:
    """Generates Voronoi domains."""
    if np.prod(shape) <= points:
        raise ValueError('Number of seeds is greater than the number of points.')
    elements = list(map(np.arange, shape))
    grid = np.meshgrid(*elements, indexing='ij')
    all_randoms = itertools.product(*elements)
    randoms = sorted(all_randoms, key=lambda k: random.random())[:points]

    azimuths = np.random.rand(points) * 2 * np.pi
    elevations = np.random.rand(points) * np.pi
    min_distance_field = np.ones(tuple(shape)) * max(shape) * 2
    orientation_field = np.zeros((2,) + tuple(shape))
    for az, el, p_vect in zip(azimuths, elevations, randoms):
        difference_field = np.sqrt(sum([pow(dist - p, 2) for dist, p in zip(grid, p_vect)]))
        loc = np.where(difference_field < min_distance_field)
        orientation_field[0][loc] = az
        orientation_field[1][loc] = el
        min_distance_field[loc] = difference_field[loc]

    if cart:
        orientation_field = sph2cart(np.ones_like(orientation_field[0]), orientation_field[0], orientation_field[1])
    return orientation_field, randoms


def skyrmion(nx, ny, number=1, helicity=0, polarity=1, neel: bool = False) -> np.ndarray:
    """Creates a skyrmion texture of size (nx, ny).

    :param nx:          Number of points in the x-direction.
    :param ny:          Number of points in the y-direction.
    :param number:      The skyrmion topological number.
    :param helicity:    Global angular offset.
    :param polarity:    Direction of center spin (±1).
    :param neel:        Neel or Bloch skyrmion.
    :return:
    """
    xx, yy = create_mesh(nx, ny)
    rad, azimuth = cart2pol(xx, -yy)

    # Normalize polarity to ±1 and neel to ±1
    polarity = np.sign(polarity)
    if neel and helicity:
        logging.warning("Neel skyrmions should not have a helicity.")

    theta = 2 * rad / (np.min([nx, ny]) - 1) * np.pi
    my, mx, mz = sph2cart(np.ones_like(theta), theta, number * (azimuth + helicity))

    if neel:
        my, mx = -mx, my

    mx = np.where(theta > np.pi, 0, mx)
    my = np.where(theta > np.pi, 0, my)
    mz = np.where(theta > np.pi, -polarity, mz) * polarity
    return np.array([mx, my, mz])


def meron(nx: int, ny: int, number: float = 1) -> np.ndarray:
    """Creates a magnetic meron."""
    xx, yy = create_mesh(nx, ny)
    rad, azimuth = cart2pol(xx, -yy)
    theta = rad / (np.min([nx, ny]) - 1) * np.pi
    my, mx, mz = sph2cart(np.ones_like(theta), theta, number * azimuth)
    mz = np.where(theta > np.pi / 2, 0, mz)
    return np.array([mx, my, mz])


def domain_wall(nx: int, ny: int, neel=False, width=2):
    """Creates a 2D magnetic domain wall."""

    xx, yy = create_mesh(nx, ny)
    angle = np.arctan(xx / width)
    mx = np.cos(angle)
    my = -np.sin(angle)
    mz = np.zeros_like(angle)

    if neel:
        mx, mz = mz, mx
    return np.array([mx, my, mz])


def bloch_point(nx, ny, nz, inwards=False, winding=1):
    """Creates a Bloch point topological defect."""
    xx, yy, zz = create_mesh(nx, ny, nz)
    _, t, p = cart2sph(xx, yy, zz)
    mz = -np.cos(t) if inwards else np.cos(t)
    mx = np.sin(t) * np.sin(winding * p)
    my = -np.sin(t) * np.cos(winding * p)
    return np.array([mx, my, mz])


def meron_pair(nx: int, ny: int):
    """Creates a magnetic meron pair."""
    mer = meron(nx, ny, 1)
    anti_mer = meron(nx, ny, -1)
    return np.hstack([mer, -anti_mer])


def stack_config(config, repeat, axis=-1):
    """Stacks 2D slices to form a 3D structure."""
    return np.stack([config] * repeat, axis=axis)


def _create_mesh_ints(*args: list[int]):
    """Creates an ND mesh centered at the origin."""
    if any(not isinstance(v, int) for v in args) or min(args) < 1:
        raise ValueError("Only positive integer values can be converted to a mesh.")
    vectors = map(lambda x: np.arange(x) - (x - 1) / 2, args)
    return np.meshgrid(*vectors, indexing='ij')


def create_mesh(*args: Union[int, float]):
    """Creates an ND mesh with the specified dimensions."""
    if len(args) < 1:
        raise ValueError("Need more than one dimensions.")
    if all(isinstance(v, int) for v in args):
        logging.debug("all ints")
        return _create_mesh_ints(*args)
    return np.meshgrid(*args, indexing='ij')
