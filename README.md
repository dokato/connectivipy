ConnectiviPy
=========================
Python connectivity module.
It is a part of [GSOC 2015](http://www.google-melange.com/gsoc/project/details/google/gsoc2015/dokato/5649050225344512) project.
Project blog is [here](http://dokato.github.io/connpy-blog/).

## Content

* Data - wrapper of your data
* Connectivity - classes with connectivity estimators
* Mvar fitting - this submodule includes some MVAR algorithms

## License
BSD 2-clause

## Documentation

Visit [ReadTheDocs](http://connectivipy.readthedocs.org/).

## Installation

Option 1: using GIT

```
git clone https://github.com/dokato/connectivipy.git
cd connectivipy
python setup.py install
```

Option 2: Download ZIP from the menu on the right, unzip it and go
in terminal to that folder. Than just execute:
```
python setup install 
```

###Changelog

#### 0.34
* short-time statistics
* documentation in sphinx ready on readthedocs
* visualization improved
* conversion to trans3d
* more examples

#### 0.31
* connectivity methods: gDTF, gPDC, Coherency, PSI, GCI
* new mvar estimation criterion: FPE
* statistics for multitrial (bootstrap) and normal case (surrogate data)
* fitting mvar for multitrial
* short-time versions of estimation
* data plotting
* working example

#### 0.2
* connectivity methods: DTF, PDC, Partial Coherence, iPDC
* mvar estimation criterions
* mvar class static

#### 0.1
* data class with simple preprocessing methods
* mvar class almost done

#### 0.05
* project structure
* basic fitting
