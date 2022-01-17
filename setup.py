from setuptools import setup

setup(
    name='pytorch_blocks',
    version='0.0.1',
    description='All sorts of high-level pytorch blocks',
    url='https://github.com/dongkyuk/Pytorch-Blocks.git',
    author='Dongkyun Kim',
    author_email='dongkyuk.andrew.cmu.edu',
    license='MIT',
    packages=['pytorch_blocks'],
    zip_safe=False,
    install_requires=[
        'torch',
        'torchvision',
        'einops',
    ]
)