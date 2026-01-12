Mesh symmetries
===============

Defining a symmetric mesh
~~~~~~~~~~~~~~~~~~~~~~~~~

Mesh symmetries can be used to speed up the computation.
Four kind of symmetries are supported by Capytaine:

* **Single plane symmetry** with respect to the :math:`x = 0` or :math:`y = 0` plane.
  This is the symmetry of most ship hulls and is thus implemented in almost all linear sea-keeping codes.

    A mesh with such a symmetry is stored by Capytaine with the
    :class:`~capytaine.new_meshes.symmetric_meshes.ReflectionSymmetricMesh` class.
    It is defined with an other mesh of the half and a plane (and optionally a name
    like the usual meshes)::

        half_mesh = cpt.load_mesh(...)
        mesh = cpt.ReflectionSymmetricMesh(half_mesh, plane="xOz", name="my full mesh")

    When loading a file in a format that support defining a symmetry (`gdf`,
    `hst`, `mar`, `pnl`), the ``ReflectionSymmetricMesh`` is returned
    automatically by ``load_mesh``.

* **Two plane symmetries** with respect to both the :math:`x = 0` and :math:`y = 0` plane.

    Two vertical plane symmetries can be nested to be used by Capytaine (assuming
    that the two planes are orthogonal)::

        quarter_mesh = cpt.load_mesh(...)
        half_mesh = cpt.ReflectionSymmetricMesh(half_mesh, plane="yOz")
        mesh = cpt.ReflectionSymmetricMesh(half_mesh, plane="x0z")

* **Rotation symmetry** of a shape repeated by rotation around the :math:`z`-axis any number of times.

    It can be defined either from the repetition of an existing mesh::

        wedge_mesh = cpt.load_mesh(...)
        mesh = cpt.RotationSymmetricMesh(wedge=wedge_mesh, n=4)

    or for an axysymmetric geometry from a list of points along a "meridian" of the shape::

        meridian_points = np.array([(np.sqrt(1-z**2), 0.0, z) for z in np.linspace(-1.0, 1.0, 10)])
        sphere = cpt.RotationSymmetricMesh.from_profile_points(meridian_points, n=10)

..
    * **Dihedral symmetry** is the combination of a plane symmetry and the rotation symmetry.

        It is defined by nesting a ``RotationSymmetricMesh`` into a ``ReflectionSymmetricMesh``::

            half_wedge = cpt.load_mesh(...)
            inner_mesh = cpt.RotationSymmetricMesh(half_wedge, n=4)
            mesh = cpt.ReflectionSymmetricMesh(half=inner_mesh, plane="xOz")

        The nesting in the other order is supported, but not as efficient.

Manipulating a symmetric mesh
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All the methods defined in the :doc:`mesh` section can be used on the ``ReflectionSymmetricMesh`` and ``RotationSymmetricMesh``.

It the resulting object is not symmetric anymore, the symmetry is lost and a bar ``Mesh`` of the whole surface is returned::

    mesh = cpt.ReflectionSymmetricMesh(..., plane="xOz")
    x_translated_mesh = mesh.translated_x(1.0)
    print(x_translated_mesh.__class__)  # ReflectionSymmetricMesh
    y_translated_mesh = mesh.translated_y(1.0)
    print(y_translated_mesh.__class__)  # Mesh

In particular, joining meshes with ``+`` or ``join_meshes`` conserves the symmetry assuming all meshes have the same symmetry.
Clipping the part of the mesh above the free surface with ``immersed_part`` should always conserve the symmetry.


Using a symmetric mesh to define a floating body
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The symmetric mesh can be used to setup a floating body::

    mesh = cpt.ReflectionSymmetricMesh(..., plane="xOz")
    lid_mesh = cpt.ReflectionSymmetricMesh(..., plane="xOz")
    body = cpt.FloatingBody(
                mesh=mesh,
                lid_mesh=lid_mesh,
                dofs=cpt.rigid_body_dofs()
                )

.. warning::
   When using a lid mesh for irregular frequencies removal, the lid mesh and
   the hull mesh should have the same symmetry, otherwise the symmetry will be
   ignored when solving the BEM problem.

.. note::
   For all the symmetries described above, the **mesh** (and the lid mesh)
   needs to be symmetric, but the full problem might not be.
   In other word, even when working with a symmetric mesh, it is possible to
   consider incoming waves from any directions, or generalized degrees of
   freedom that are not symmetric.


Expected performance gain
~~~~~~~~~~~~~~~~~~~~~~~~~

Resolution time
---------------

Considering a 1 plane symmetry problem, the size of the problem goes from :math:`N` 
to 2 blocks of size :math:`N/2`.
For a 2 plane symmetry problem, we end up with 4 blocks size :math:`N/4`.

More generally for an :math:`n` rotation symmetry problem, the problem is decomposed 
into :math:`n` blocks of size :math:`N/n`.

Thus, asymptotically and for large problems the overall gain factor in the resolution time 
is approximately :math:`1/n`. 
However, by decomposing the solution time into its main computational steps,  
we can give a more precise estimate  of the gain factor for each step.

This gain depends on the chosen linear solver. There is no gain for the GMRES solver.
Since the LU decomposition has a computational complexity of :math:`O(N^3)`, 
the expected gain factor is :math:`1/n^2`. 
The time spent computing the Green's function is reduced by a factor :math:`1/n`.
There is no gain for the matrix-vector product.

Estimating the overall performance gain remains difficult,  
as it strongly depends on the relative proportion of each computational step,  
which is problem-dependent. 

+------------------------+----------------+-----------+--------------+------------------------+
| Symmetry               | Green function | LU solver | GMRES solver |  Matrix-vector product |
+------------------------+----------------+-----------+--------------+------------------------+
|| 1 plane symmetry or   || 1/2           || 1/4      || 1           || 1                     |
|| 2 rotation symmetry   ||               ||          ||             ||                       |
+------------------------+----------------+-----------+--------------+------------------------+
|| 2 plane symmetries or || 1/4           || 1/16     || 1           || 1                     |
|| 2 rotation symmetry   ||               ||          ||             ||                       |
+------------------------+----------------+-----------+--------------+------------------------+
| n rotation symmetry    | 1/n            | 1/n²      | 1            | 1                      |
+------------------------+----------------+-----------+--------------+------------------------+


RAM usage 
---------

Regarding RAM usage, there is also, asymptotically for large problems, 
a gain factor of :math:`1/n`.
This gain is exact for GMRES solver.
However, it is slightly lower for the LU solver due to intermediate step. 

Solving a problem of size :math:`N` with an LU solver requires 3 matrices 
(2 for the resolution of the problem itself, and 1 for the LU decomposition),
which corresponds to :math:`3N^2` memory allocations.

Considering a n rotation symmetry problem there is still 2 matrices 
for the resolution of the problem but made of n blocks of size :math:`N/n`.
There are :math:`n` LU decomposition of size :math:`N/n` 
and there is also one intermediate step made of n blocks of size :math:`N/n`.
Finally there are :math:`2n(N/n)^2 + n(N/n)^2 + n(N/n)^2 = 4N^2/n` allocations, 
thus the gain factor is :math:`4/3n`. 

For the 1 plane symmetry problem the reasoning is exactly the same. 

The 2 plane symmetries case is a bit different. 
Since the symmetries are nested, there are actually three intermediate steps: 
one of size 4 blocks :math:`N/4`, and two more of size 2 blocks :math:`N/4`. 
Finally there are :math:`2*4(N/4)^2 + 4(N/4)^2 + 4(N/4)^2 + 2*2(N/4)^2 = 5N^2/4` allocations, 
thus the gain factor is :math:`5/12`.


+---------------------+-----------+--------------+
| Symmetry            | LU solver | GMRES solver |
+---------------------+-----------+--------------+
| 1 plane symmetry    | 2/3       | 1/2          |
+---------------------+-----------+--------------+
| 2 plane symmetries  | 5/12      | 1/4          |
+---------------------+-----------+--------------+
| n rotation symmetry | 4/3n      | 1/n          |
+---------------------+-----------+--------------+


Note that the theoretical performance gain described above might not be reached
in practice, especially for smaller problems.
For instance, the threading parallelisation is currently less efficient on
highly symmetric meshes.
