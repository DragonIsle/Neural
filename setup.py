from os.path import join, dirname

from setuptools import setup, find_packages

setup(
    name='conv-network',
    version='1.0',
    packages=find_packages(),
    long_description=open(join(dirname(__file__), 'README.rst')).read(),
    install_requires=[
        'flask',
        'python-mnist',
        'pillow',
        'numpy',
        'opencv-python',
        'scikit-image',
        'scipy',
        'matplotlib'
    ]
)
