#!/usr/bin/env python
# -*- coding: utf-8 -*-


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

setup(
    name='tntorch',
    version='1.0.0',
    description="Tensor Network Learning with PyTorch",
    long_description="tntorch is a PyTorch-powered modeling and learning library using tensor networks. Features include basic and fancy indexing of tensors, broadcasting, assignment, etc.; tensor decomposition and arithmetics; adaptive sampling (cross-approximation); global optimization of tensors; sensitivity analysis; and more.",
    url='https://github.com/rballester/tntorch',
    author="Rafael Ballester-Ripoll",
    author_email='rafael.ballester@ie.edu',
    packages=[
        'tntorch',
    ],
    include_package_data=True,
    install_requires=[
        'numpy',
        'scipy',
        'torch'
    ],
    license="LGPL",
    zip_safe=False,
    keywords='tntorch',
    classifiers=[
        'License :: OSI Approved :: ISC License (ISCL)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    test_suite='tests',
    tests_require='pytest'
)
