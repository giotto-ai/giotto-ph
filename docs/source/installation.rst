############
Installation
############

.. _installation:

************
Dependencies
************

The latest stable version of ``giotto-ph`` requires:

- Python (>= 3.6)
- NumPy (>= 1.19.1)
- SciPy (>= 1.5.0)
- scikit-learn (>= 0.23.1)

To run the examples, ``jupyter`` is required.


*****************
User installation
*****************

The simplest way to install ``giotto-ph`` is using ``pip``   ::

    python -m pip install -U giotto-ph

If necessary, this will also automatically install all the above dependencies. Note: we recommend
upgrading ``pip`` to a recent version as the above may fail on very old versions.

**********************
Developer installation
**********************

.. _dev_installation:

Installing both the PyPI release and source of ``giotto-ph`` in the same environment is not recommended since it is
known to cause conflicts with the C++ bindings.

The developer installation requires two important C++ dependencies:

-  A C++14 compatible compiler
-  CMake >= 3.9

Please refer to your system's instructions and to the `CMake <https://cmake.org/>`_ website for definitive guidance on how to install this dependency. The instructions below are unofficial, please follow them at your own risk.

Linux
=====

Most Linux systems should come with a suitable compiler pre-installed. For the other two dependencies, you may consider using your distribution's package manager, e.g. by running

.. code-block:: bash

    sudo apt-get install cmake

if ``apt-get`` is available in your system.

macOS
=====

On macOS, you may consider using ``brew`` (https://brew.sh/) to install the dependencies as follows:

.. code-block:: bash

    brew install gcc cmake

Windows
=======

On Windows, you will likely need to have `Visual Studio <https://visualstudio.microsoft.com/>`_ installed. At present,
it appears to be important to have a recent version of the VS C++ compiler. One way to check whether this is the case
is as follows:

1. open the VS Installer GUI;
2. under the "Installed" tab, click on "Modify" in the relevant VS version;
3. in the newly opened window, select "Individual components" and ensure that v14.24 or above of the MSVC "C++ x64/x86 build tools" is selected. The CMake dependency are best installed using the latest binary executable from the official website.

Source code
===========

You can obtain the latest state of the source code with the command::

    git clone https://github.com/giotto-ai/giotto-ph.git


To install:
===========

.. code-block:: bash

   cd giotto-ph
   python -m pip install -e ".[dev]"

This way, you can pull the library's latest changes and make them immediately available on your machine.
Note: we recommend upgrading ``pip`` and ``setuptools`` to recent versions before installing in this way.

Testing
=======

After installation, you can launch the test suite from inside the
source directory::

    pytest

