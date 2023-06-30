from setuptools import setup

setup(
    name='spdeinf',
    version='0.1.0',    
    description='A example Python package',
    url='https://github.com/rafaelanderka/spde-inference',
    author='Rafael Anderka',
    author_email='rafael.anderka.22@ucl.ac.uk',
    license='BSD 2-clause',
    packages=['spdeinf'],
    install_requires=['numpy',
                      'jax',
                      'scipy',
                      'scikit-sparse',
                      'matplotlib',
                      'findiff',
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3.10',
    ],
)
