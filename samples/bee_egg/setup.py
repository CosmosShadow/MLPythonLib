#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
	name = 'bee',
	version = '0.0.1',
	keywords = ('bee', 'egg'),
	description = 'a simple egg',
	license = 'MIT License',

	url = 'http://liluo.org',
	author = 'liluo',
	author_email = 'i@liluo.org',

	packages = find_packages(),
	include_package_data = True,
	platforms = 'any',
	install_requires = [],
)