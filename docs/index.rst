.. pythonworley documentation master file, created by
   sphinx-quickstart on Tue Aug  8 00:07:23 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PythonWorley
============

Python implementation of Worley noise - to seamlessly tile in any dimensions


Installation
------------

::

   pip install pythonworley



Quick start
-----------

::

   import pylab as plt
   from pythonworley import worley

   dens = 64
   shape = (8,4)
   w, c = worley(shape, dens=dens, seed=0)
   plt.imshow(w[0].T, cmap=plt.get_cmap('Greys'))

.. toctree::
   :maxdepth: 2
   :caption: Examples

   notebook/example
