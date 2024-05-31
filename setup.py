from setuptools import setup

setup(
    name='iter-inla',
    version='0.3.0',
    description='A Python implementation of iterated INLA for state and parameter estimation in nonlinear dynamical systems.',
    url='https://github.com/rafaelanderka/iter-inla',
    author='Rafael Anderka',
    author_email='rafael.anderka.22@ucl.ac.uk',
    license='BSD 2-clause',
    packages=['iinla'],
    install_requires=[
                      'scikit-learn',
                      'scipy',
                      'numpy',
                      'matplotlib',
                      'tqdm',
                      'sdeint',
                      ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3.10',
    ],
)
