
woe
==========

.. image:: https://travis-ci.org/justdoit0823/pywxclient.svg?branch=master
    :target: https://travis-ci.org/justdoit0823/pywxclient

version: 0.0.7

Tools for WoE Transformation mostly used in ScoreCard Model for credit rating

Here we go:


.. code-block:: pycon

   >>> import woe.config as config

   >>> import woe.feature_process as fp


Features
========

  * Split tree with IV criterion

  * Rich and plentiful model eval methods

  * Unified format and easy for output

  * Storage of IV tree for follow-up use


**woe aims to only support Python 2.7, so there is no guarantee for Python 3.**


Installation
============

We can simply use pip to install, as the following:

.. code-block:: bash

   $ pip install woe

or installing from git

.. code-block:: bash

   $ pip install git+https://github.com/boredbird/woe


Examples
========

In the examples directory, there is a simple woe transformation program as tutorials.

Or you can write a more complex program with this `woe` package.

Version Records
================

2017-09-19

	* Fix bug: eval.eval_feature_detail raises ValueError('arrays must all be same length')
	* Add parameter interface: alpha specified step learning rate ,default 0.01

How to Contribute
=================

Email me,1002937942@qq.com.
