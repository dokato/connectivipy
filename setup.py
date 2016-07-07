 #! /usr/bin/env python
 
# python installer will be inlcuded here ...
from __future__ import print_function
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
import connectivipy 

requirements = [
    'numpy>=1.7.1',
    'scipy>=0.11.0'
]

setup(  name='connectivipy',
        version= connectivipy.__version__,
        description='Python Connectivity Module',
        license='bsd',
        author='Dominik Krzeminski',
        install_requires=requirements,
        packages=['connectivipy', 'connectivipy.load', 'connectivipy.mvar'],
        include_package_data=True,
        platforms='any',
        keywords=['connectivity','mvar', 'biosignals', 'eeg', 'autoregressive model', 'ar model'],
        url = 'https://github.com/dokato/connectivipy',
        download_url = 'https://github.com/dokato/connectivipy/releases/tag/v0.37'
     )
