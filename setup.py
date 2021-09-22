 #! /usr/bin/env python
 
try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages
import connectivipy 

requirements = [
    'numpy>=1.7.1',
    'scipy>=0.11.0'
]

from distutils.core import setup

setup(
    name='connectivipy',
    version= connectivipy.__version__,
    description='Python Connectivity Module',
    license='bsd',
    author='Dominik Krzeminski',
    install_requires=requirements,
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(include=['connectivipy', 'connectivipy.*']),
    include_package_data=True,
    platforms='any',
    keywords=['connectivity','mvar', 'biosignals', 'eeg', 'autoregressive model', 'ar model'],
    url = 'https://github.com/dokato/connectivipy',
    download_url = 'https://github.com/dokato/connectivipy/releases/tag/v0.37'
)
