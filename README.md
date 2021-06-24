ConnectiviPy
=========================
Python module for connectivity analysis. You may find here implementation
of MVAR based estimators like DTF, PDC and also Coherency, PSI. For more
information visit documentation page.

Project was supported by [Google Summer of Code](http://www.google-melange.com/gsoc/project/details/google/gsoc2015/dokato/5649050225344512)
and made under [International Neuroinformatics Coordination Facility](http://incf.org/) wings.

[![Build Status](https://travis-ci.org/dokato/connectivipy.svg?branch=master)](https://travis-ci.org/dokato/connectivipy)
[![codecov](https://codecov.io/gh/dokato/connectivipy/branch/master/graph/badge.svg)](https://codecov.io/gh/dokato/connectivipy)


## Content

* Data - data object that makes data handling easier
* Connectivity - class with connectivity estimators
* Mvar fitting - this submodule includes some MVAR algorithms

## License
BSD 2-clause

## Documentation

Visit [ReadTheDocs](http://connectivipy.readthedocs.org/) for detailed
documentation and tutorials.

## Installation

Option 1: PIP (stable release)

```
pip install connectivipy
```

Option 1: PIP+GIT (latest release)

```
pip install git+https://github.com/dokato/connectivipy.git
```


Option 3: using GIT (the most recent version)

```
git clone https://github.com/dokato/connectivipy.git
cd connectivipy
python setup.py install
```

## Authors
* Dominik Krzemiński
* Maciej Kamiński (scientific lead)
