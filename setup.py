#!/usr/bin/env python3

import os
from setuptools import setup, find_packages

# Read the contents of your README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements from requirements.txt
def read_requirements() -> list[str]:
    """Read requirements from requirements.txt file."""
    with open('requirements.txt') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='openai2civitai',
    version='1.0.0',
    author='FarisHijazi',
    description='A proxy server that translates between OpenAI Images API and CivitAI API for image generation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/FarisHijazi/openai2civitai',
    packages=find_packages(),
    py_modules=['app', 'prompt_parser', 'prompt_reviser'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Internet :: WWW/HTTP :: HTTP Servers',
    ],
    python_requires='>=3.8',
    install_requires=read_requirements(),
    extras_require={
        'dev': [
            'pytest',
            'requests-mock',
        ],
    },
    entry_points={
        'console_scripts': [
            'openai2civitai=openai2civitai.server:main',
        ],
    },
    include_package_data=True,
    package_data={
        '': ['*.md', '*.txt', '*.yml', '*.yaml'],
    },
)