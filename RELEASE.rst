Release 0.2.1
=============

Minor release bringing bug fixes and performance improvements.

Major Features and Improvements
-------------------------------

- Miscellaneous improvements including verification that ``coeff`` is a prime, improvements to the Python bindings (see "Backwards-Incompatible Changes" below), updated C++ preprocessor directives and code style fixes (`#50 <https://github.com/giotto-ai/giotto-ph/pull/50>`_).
- An unnecessary binary search for 0-dimensional simplices is now avoided in the C++ backend, leading to faster runtimes (`#51 <https://github.com/giotto-ai/giotto-ph/pull/51>`_).
- ``scikit-learn``'s ``NearestNeighbors`` is now used for computing sparse thresholded distance matrices, leading to large benefits on memory consumption and runtime in many cases of interest (`#54 <https://github.com/giotto-ai/giotto-ph/pull/54>`_).
- General improvements to the documentation (`#54 <https://github.com/giotto-ai/giotto-ph/pull/54>`_ and `#58 <https://github.com/giotto-ai/giotto-ph/pull/58>`_).

Bug Fixes
---------

- A bug in computing sparse thresholded distance matrices has been fixed (`#54 <https://github.com/giotto-ai/giotto-ph/pull/54>`_).

Backwards-Incompatible Changes
------------------------------

- Barcodes are now returned as (lists of) arrays of dtype ``numpy.float32`` instead of ``numpy.float64``, since single-precision floats are used internally by the C++ backend (`#50 <https://github.com/giotto-ai/giotto-ph/pull/50>`_).

Thanks to our Contributors
--------------------------

This release contains contributions from many people:

Julian Burella Pérez and Umberto Lupo.

We are also grateful to all who filed issues or helped resolve them, asked and answered questions, and were part of inspiring discussions.

Release 0.2.0
=============

Major release bringing new features, performance enhancements, and bug fixes.

Major Features and Improvements
-------------------------------

- ``ripser_parallel`` can now return "flag persistence generators", i.e. vertices and edges creating/destroying bars in the barcode (`#29 <https://github.com/giotto-ai/giotto-ph/pull/29>`_). A new Jupyter notebook illustrates usage of this new feature.
- Wheels for Python 3.10 have been added (`#47 <https://github.com/giotto-ai/giotto-ph/pull/47>`_).
- The computation of the enclosing radius has been sped up (`#28 <https://github.com/giotto-ai/giotto-ph/pull/28>`_).
- ``ripser_parallel`` can now be imported directly from ``gph`` (`#36 <https://github.com/giotto-ai/giotto-ph/pull/36>`_).
- A garbage collection step which was found to negatively impact runtimes with little memory benefit has been removed (`#33 <https://github.com/giotto-ai/giotto-ph/pull/33>`_).

Bug Fixes
---------

- Sparse input with (signed) 64-bit row and column indices is now correctly dealt with by the Python interface (`#34 <https://github.com/giotto-ai/giotto-ph/pull/34>`_).
- A bug causing segfaults to occur when ``maxdim=0`` was passed to the C++ backend has been fixed (`#40 <https://github.com/giotto-ai/giotto-ph/pull/40>`_).
- An algorithmic error in dealing with edges with zero weight in the 0-dimensional computation has been fixed (`#43 <https://github.com/giotto-ai/giotto-ph/pull/43>`_).

Backwards-Incompatible Changes
------------------------------

None.

Thanks to our Contributors
--------------------------

This release contains contributions from many people:

Umberto Lupo and Julian Burella Pérez.

We are also grateful to all who filed issues or helped resolve them, asked and answered questions, and were part of inspiring discussions.

Release 0.1.0
=============

Initial release of ``giotto-ph``.

Major Features and Improvements
-------------------------------

The following methods where added:

-  ``ripser`` computes the persistent homology for Vietoris-Rips filtrations with parallel computation.

Bug Fixes
---------


Backwards-Incompatible Changes
------------------------------


Thanks to our Contributors
--------------------------

This release contains contributions from many people:

Julian Burella Pérez, Sydney Hauke and Umberto Lupo.

We are also grateful to all who filed issues or helped resolve them, asked and answered questions, and were part of inspiring discussions.
