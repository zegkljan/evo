# -*- coding: utf8 -*-
from codecs import open
from os import path

from setuptools import setup, find_packages

from evo import __version__ as evo_version

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with open(path.join(here, 'DESCRIPTION.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    # core descriptors
    name='evo',
    version=evo_version,
    packages=find_packages(where=here, exclude=('tests', 'tests.*')),
    install_requires=[
        'setuptools',
        'numpy',
        'PyYAML'
    ],
    package_data={
    },
    # metadata
    url='https://gitlab.fel.cvut.cz/zegkljan/evo',
    license='Academic Free License (AFL)',
    author='Jan Žegklitz',
    author_email='zegkljan@fel.cvut.cz',
    description='Framework for evolutionary computation',
    long_description=long_description,
    classifiers=[
        'Development Status :: 1 - Planning',

        'Intended Audience :: Science/Research',

        'Topic :: Scientific/Engineering :: Artificial Intelligence',

        'License :: OSI Approved :: Academic Free License (AFL)',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5'
    ],
    keywords='evolutionary optimization ai'
)
