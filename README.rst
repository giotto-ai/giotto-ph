=========
giotto-ph
=========

``giotto-ph`` is a high-performance implementation of Vietoris–Rips (VR) persistence, and is distributed under the GNU AGPLv3 license.
It consists of an improved reimplementation of `Morozov and Nigmetov's "lock-free Ripser" <https://dl.acm.org/doi/10.1145/3350755.3400244>`_,
integrated with an improved reimplementation of `GUDHI's Edge Collapse (EC) algorithm <https://hal.inria.fr/hal-02395227>`_ and with support
for weighted VR filtrations. See also https://github.com/Ripser/ripser/pull/31, https://github.com/grey-narn/oineus (lock-free Ripser), and http://gudhi.gforge.inria.fr/doc/latest/group__edge__collapse.html (GUDHI's EC implementation).

``giotto-ph`` is part of the `Giotto <https://github.com/giotto-ai>`_ family of open-source projects and designed for tight integration with
the `giotto-tda <https://github.com/giotto-ai/giotto-tda>`_ and `pyflagser <https://github.com/giotto-ai/giotto-tda>`_ libraries.

Project genesis
===============

``giotto-ph`` is the result of a collaborative effort between `L2F SA <https://www.l2f.ch/>`_,
the `Laboratory for Topology and Neuroscience <https://www.epfl.ch/labs/hessbellwald-lab/>`_ at EPFL,
and the `Institute of Reconfigurable & Embedded Digital Systems (REDS) <https://heig-vd.ch/en/research/reds>`_ of HEIG-VD.

License
=======

.. _L2F team: business@l2f.ch

``giotto-ph`` is distributed under the AGPLv3 `license <https://github.com/giotto-ai/giotto-tda/blob/master/LICENSE>`_.
If you need a different distribution license, please contact the `L2F team`_.

Documentation
=============

Please visit https://giotto-ai.github.io/giotto-ph and navigate to the version you are interested in.

Installation
============

Dependencies
------------

The latest stable version of ``giotto-ph`` requires:

- Python (>= 3.6)
- NumPy (>= 1.19.1)
- SciPy (>= 1.5.0)
- scikit-learn (>= 0.23.1)

User installation
-----------------

The simplest way to install ``giotto-ph`` is using ``pip``   ::

    python -m pip install -U giotto-ph

If necessary, this will also automatically install all the above dependencies. Note: we recommend
upgrading ``pip`` to a recent version as the above may fail on very old versions.

Developer installation
----------------------

Please consult the `dedicated page <https://giotto-ai.github.io/gtda-docs/latest/installation.html#developer-installation>`_
for detailed instructions on how to build ``giotto-ph`` from sources across different platforms.

.. _contributing-section:

Contributing
============

We welcome new contributors of all experience levels. The Giotto community goals are to be helpful, welcoming,
and effective. To learn more about making a contribution to ``giotto-ph``, please consult `the relevant page
<https://giotto-ai.github.io/gtda-docs/latest/contributing/index.html>`_.

Testing
-------

After installation, you can launch the test suite from outside the
source directory   ::

    pytest

Important links
===============

- Issue tracker: https://github.com/giotto-ai/giotto-ph/issues


Community
=========

giotto-ai Slack workspace: https://slack.giotto.ai/

Contacts
========

maintainers@giotto.ai
