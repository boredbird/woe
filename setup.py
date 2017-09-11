# -*- coding:utf-8 -*-
__author__ = 'boredbird'

from setuptools import setup, find_packages

setup(
    name='woe',
    version='0.0.1',
    description=(
        'Tools for WoE Transformation mostly used in ScoreCard Model for credit rating'
    ),
    long_description=open('README.rst').read(),
    author='boredbird',
    author_email='1002937942@qq.com',
    maintainer='boredbird',
    maintainer_email='1002937942@qq.com',
    license='NJUPT',
    packages=find_packages(),
    platforms=["all"],
    url='https://github.com/boredbird/woe',
    classifiers=[
        'Development Status :: 1 - Alpha',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: NJUPT',
        'Programming Language :: Python',
        'Programming Language :: Python :: Implementation',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Topic :: Software Development :: Libraries'
    ],
    keywords = ["math","finance","scorecard","woe",'iv'],
    install_requires = [
        'pandas>=0.19.2',
        'numpy>=1.11.3',
        'scipy>=0.18.1',
        'matplotlib>=2.0.0',
    ]
)
