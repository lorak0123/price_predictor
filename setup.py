from setuptools import setup

setup(
    name='price_predictor',
    version='0.0.1',
    py_modules=['predict'],
    url='https://github.com/lorak0123/price_predictor.git',
    license='',
    author='Tomasz Janus, Karol Pilot',
    author_email='',
    description='Price prdictor',
    install_requires=[
        'pandas',
        'numpy',
        'sklearn'
    ],
)