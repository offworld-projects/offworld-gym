from setuptools import setup, find_packages
from offworld_gym import version 

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='offworld-gym',
      version=version.__version__,
      packages=find_packages(),
      install_requires=requirements,
      description='A suite of realistic environment to develop reinforcement learning algorithms and compare results.',
      url='https://github.com/offworld-projects/offworld-real-gym',
      author='ashish.kumar@offworld.ai',
      tests_require=['pytest', 'mock'],
)