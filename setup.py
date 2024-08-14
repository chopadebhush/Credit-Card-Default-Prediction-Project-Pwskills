from setuptools import find_packages,setup

setup(
    name='Credit-Card-Default-Prediction',
    version='0.0.1',
    author='Deepak Singh',
    author_email='itsdeepaksingh00@gmail.com',
    packages= find_packages()
)
classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
python_requires='>=3.6',
install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'tensorflow',
        'ipykernel',
    ]