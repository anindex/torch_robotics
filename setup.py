from setuptools import setup
from codecs import open
from os import path


ext_modules = []

here = path.abspath(path.dirname(__file__))
requires_list = []
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    for line in f:
        requires_list.append(str(line))


setup(name='torch_kinematics_tree',
      description='Full Differentiable Kinematics Tree Implementation constructed from URDF or MJCF',
      author='An Thai Le',
      author_email='an@robot-learning.de',
      packages=['torch_kinematics_tree'],
      install_requires=requires_list,
      )
