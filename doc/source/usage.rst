Usage
=====

Installation
------------

To use magpack, first install it using pip:

.. code-block:: console

   (.venv)$ pip install magpack


Examples
--------
magpack contains tools for:

- loading and saving data in formats such as ``.mat``, ``.ovf`` and ``.vtk`` (:mod:`magpack.io`)
- performing vector operations (:mod:`magpack.vectorop`, see :ref:`vector_ex` section)
- calculating and evaluating scalar and vector field rotations (:mod:`magpack.rotations`, see
  :ref:`rotations_ex` section)
- removing image distortions, coloring complex arrays (e.g. Fourier transforms) and evaluating image resolutions using
  Fourier shell correlation (:mod:`magpack.image_utils`)

.. _vector_ex:

Vector Operations
^^^^^^^^^^^^^^^^^
magpack provides vector operations that help analyse both simulations and reconstructions. The convention used here is
such that index *0* corresponds to vector components, such that a vector field :math:`\vec{m} = (m_x,\,m_y,\,m_z)`
is created by combining the three constituent scalar fields os shape ``(nx, ny, nz)``

.. code-block::

    vector_field = np.stack([mx, my, mz])

thus the ``vector_field`` variable will have shape ``(3, nx, ny, nz)``. Operations that are not specific to 3D, (e.g.
curl) are generalized for M-component fields for N-dimensions ``(M, N1, N2, ..., Nn)``

Some of the available operations include:

- Gradient, :math:`\nabla f` :func:`magpack.vectorop.scalar_gradient()`
- Divergence, :math:`\mathbf{\nabla} \cdot \mathbf{v}` :func:`magpack.vectorop.divergence()`
- Curl, :math:`\mathbf{\nabla} \times \mathbf{v}` :func:`magpack.vectorop.curl()`
- Laplacian, :math:`\nabla^2 \times f` :func:`magpack.vectorop.scalar_laplacian()`
- Vector Laplacian, :math:`\mathbf{\nabla} \mathbf{v}` :func:`magpack.vectorop.vector_laplacian()`
- Vorticity, :math:`\frac{1}{8\pi}\epsilon_{abc}\epsilon_{ijk}m_{i}\partial_{b}m_{j}\partial_{c}m_{k}`
  :func:`magpack.vectorop.vorticity()`

.. _rotations_ex:

Rotations
^^^^^^^^^
:mod:`magpack.rotations` provides functions for rotating scalar and vector fields. It can also be used to calculate
rotation matrices for different 3D imaging geometries such as tomography and laminography. For arbitrary rotation
operations, the function :func:`magpack.rotations.eul2rot()` can be used.

In tomographic imaging (assuming the incident beam is along the *z* axis), the sample is first tilted and then rotated
from 0° to 180° about an axis perpendicular to the propagation of the beam. A sequence of rotation matrices describing
this operation can be obtained using either of the two functions

.. code-block::

    angles = np.linspace(0, 179, 180)
    rot = magpack.rotations.tomo_rot(angles, 30)
    rot_eul = magpack.rotations.eul2rot('yz', angles, 30)

Both functions return the same matrices, but :func:`magpack.rotations.eul2rot()` allows for more control over the
order of operations and axes. The convention employed here, using the example ``yz``, is such that the tilting (``z``
rotation by 30°) is performed first, followed by the ``y`` rotation (from 0° to 179°). This is similar to matrix
multiplication :math:`\mathbf{R}_y(\theta) \mathbf{R}_z(\phi) \vec{m}` where :math:`\vec{m}` represents a vector field.