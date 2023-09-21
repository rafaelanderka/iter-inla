from setuptools import setup

setup(
    name='spdeinf',
    version='0.2.0',    
    description='Framework for Bayesian inference with nonlinear SPDEs',
    url='https://github.com/rafaelanderka/spde-inference',
    author='Rafael Anderka',
    author_email='rafael.anderka.22@ucl.ac.uk',
    license='BSD 2-clause',
    packages=['spdeinf'],
    install_requires=['findiff @ git+ssh://git@github.com:rafaelanderka/findiff.git',
                      'scikit-sparse @ git+ssh://git@github.com:rafaelanderka/scikit-sparse.git',',
                      'scikit-learn',
                      'scipy',
                      'numpy',
                      'matplotlib',
                      'tqdm',
                      ],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3.10',
    ],
)
