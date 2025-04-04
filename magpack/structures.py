import logging
import numpy as np
from magpack.vectorop import cart2pol, sph2cart, cart2sph
import itertools
import random


def checkerboard(shape):
    """Creates a checkerboard array of alternating +1 and -1 for the given shape.

    Parameters
    ----------
    shape : tuple of int
        Shape of the checkerboard.

    Returns
    -------
    np.ndarray
        Array of alternating +1 and -1 for the given shape.
    """
    return (np.indices(shape).sum(axis=0) % 2) * 2 - 1


def circ_mask(nx, ny):
    """Defines a circular binary mask.

    Parameters
    ----------
    nx, ny : int
        x and y dimensions of the circular mask.

    Returns
    -------
    np.ndarray
        Circular binary mask.
    """
    xx, yy = create_mesh(nx, ny)
    circ = np.sqrt(xx ** 2 + yy ** 2) < np.min([nx, ny]) // 2
    return circ


def vortex(nx, ny, winding = 1):
    """Creates a 2D magnetization vortex.

    Parameters
    ----------
    nx, ny : int
        x and y dimensions of the vortex.
    winding : float (optional)
        Winding number of the vortex

    Returns
    -------
    np.ndarray
        Magnetization vector of a 2D vortex with specified winding number.
    """
    xx, yy = create_mesh(nx, ny)
    rad, azimuth = cart2pol(yy, xx)

    mx = -np.cos(winding * azimuth)
    my = np.sin(winding * azimuth)
    mz = np.zeros_like(azimuth)

    return np.array([mx, my, mz])


def domain_generator(shape, points, cart = False):
    """Generates Voronoi domains with random orientations.

    Parameters
    ----------
    shape : array_like
        Shape of the final array.
    points : int
        Number of seed points.
    cart : bool (optional)
        Return cartesian coordinates if True, spherical coordinates if False.

    Returns
    -------
    np.ndarray
        Array with orientations (3, shape)
    list
        Coordinates of seed points.
    """
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


def skyrmion(nx, ny, number=1, helicity=0, polarity=1, neel = False):
    """Creates a skyrmion texture of size (nx, ny).

    Parameters
    ----------
    nx, ny : int
        Size of the skyrmion texture in the x and y directions.
    number : float (optional)
        Skyrmion topological number.
    helicity : float (optional)
        Helicity of the skyrmion (angular offset at each lattice site in radians).
    polarity : int (optional)
        Direction of central spin (±1).
    neel : bool (optional)
        Neel or Bloch skyrmion.

    Returns
    -------
    np.ndarray
        Magnetization vector field of a skyrmion.
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


def meron(nx, ny, number = 1):
    """Creates a magnetic meron.

    Parameters
    ----------
    nx, ny : int
        Size of the meron texture in the x and y directions.
    number : float (optional)
        Meron topological number.

    Returns
    -------
    np.ndarray
        Magnetization vector field of a meron.
    """
    xx, yy = create_mesh(nx, ny)
    rad, azimuth = cart2pol(yy, -xx)
    theta = np.clip(rad / (np.min([nx, ny]) - 1) * np.pi, 0, np.pi / 2)
    mx, my, mz = sph2cart(np.ones_like(theta), theta, number * azimuth)
    mz = np.where(theta > np.pi / 2, 0, mz)
    return np.array([mx, my, mz])


def domain_wall(nx, ny, neel=False, width=2):
    """Creates a 2D magnetic domain wall.

    Parameters
    ----------
    nx, ny : int
        Size of the domain wall along the x and y directions.
    neel : bool (optional)
        Neel domain wall if True, Bloch domain wall if False.
    width : int
        Width of the magnetic domain wall.

    Returns
    -------
    np.ndarray
        Magnetization vector field of the domain wall.
    """

    xx, yy = create_mesh(nx, ny)
    angle = np.arctan(xx / width)
    mx = np.cos(angle)
    my = -np.sin(angle)
    mz = np.zeros_like(angle)

    if neel:
        mx, mz = mz, mx
    return np.array([mx, my, mz])


def bloch_point(nx, ny, nz, inwards=False, winding=1):
    """Creates a Bloch point topological defect.

    Parameters
    ----------
    nx, ny, nz : int
        Size of the Bloch point along the x, y and z directions.
    inwards : bool (optional)
        If True, magnetization points towards the defect's core.
    winding : float (optional)
        Winding number of the Bloch point.

    Returns
    -------
    np.ndarray
        Magnetization vector field of a bloch point.
    """
    xx, yy, zz = create_mesh(nx, ny, nz)
    _, t, p = cart2sph(xx, yy, zz)
    mz = -np.cos(t) if inwards else np.cos(t)
    mx = np.sin(t) * np.sin(winding * p)
    my = -np.sin(t) * np.cos(winding * p)
    return np.array([mx, my, mz])


def meron_pair(nx, ny):
    """Creates a meron-antimeron pair. The two meron structures are combined side-by-side along the x-axis.

    Parameters
    ----------
    nx, ny : int
        Size of a single meron texture in the x and y directions.

    Returns
    -------
    np.ndarray
        Magnetization vector field of a meron-antimeron pair.
    """
    mer = meron(nx, ny, 1)
    anti_mer = meron(nx, ny, -1)
    return np.hstack([mer, -anti_mer])


def stack_config(config, repeat, axis=-1):
    """Stacks 2D slices to form a 3D structure.

    Parameters
    ----------
    config : np.ndarray
        Two-dimensional magnetic configuration.
    repeat : int
        Number of times to stack slices.
    axis : int
        Axis along which to stack slices.

    Returns
    -------
    np.ndarray
        Stacked 3D structure.
    """
    return np.stack([config] * repeat, axis=axis)


def _create_mesh_ints(*args):
    """Creates an ND mesh centered at the origin.

    Parameters
    ----------
    args : list of int
        The integer dimensions of the mesh.

    Returns
    -------
    np.ndarray
        N-dimensional mesh centered at the origin with unit spacing.
    """
    if any(not isinstance(v, int) for v in args) or min(args) < 1:
        raise ValueError("Only positive integer values can be converted to a mesh.")
    vectors = map(lambda x: np.arange(x) - (x - 1) / 2, args)
    return np.meshgrid(*vectors, indexing='ij')


def create_mesh(*args):
    """Creates an N-dimensional mesh with the specified dimensions.

    The number `N` is the number of arguments.

    Parameters
    ----------
    args : int | float | list of int | list of float | np.ndarray
        The integer dimensions of the mesh or all points along the dimension.

    Returns
    -------
    np.ndarray
        N-dimensional mesh for the corresponding array."""
    if len(args) < 1:
        raise ValueError("Need more than one dimensions.")
    if all(isinstance(v, int) for v in args):
        logging.debug("all ints")
        return _create_mesh_ints(*args)
    return np.meshgrid(*args, indexing='ij')
