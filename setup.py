# Fern
#
# Author: Jason Lin
# Email: jason.m.lin@outlook.com
#
# =============================================================================
"""setup file for packaging"""
import re
import os
import setuptools


REF = os.getenv('GITHUB_REF', '')   # refs/tags/1.1.1.dev5
VERSION = REF.split('/')[-1]
# check version, 1.1.1rc2.post1.dev1
if not VERSION or not re.match(r'^\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?(?:\.post\d+)?(?:\.dev\d+)?$', VERSION):
    raise ValueError(f'Version check failed: {VERSION}')

with open('README.md', 'r') as f:
    LONG_DESCRIPTION = f.read()

with open('requirements.txt', 'r') as f:
    INSTALL_REQUIRES = [line.strip() for line in f if line.strip()]

setuptools.setup(
    name='Fern2',
    version=VERSION,
    author='Jason, Lin',
    author_email='jason.m.lin@outlook.com',
    license='Apache 2.0',
    description='NLP text processing toolkit for Deep Learning',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    url='https://github.com/Jasonsey/Fern',
    packages=setuptools.find_packages(),
    install_requires=INSTALL_REQUIRES,
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Operating System :: MacOS',
        'Operating System :: Microsoft',
        'Operating System :: POSIX :: Linux',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Natural Language :: Chinese (Simplified)',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.6',
)
