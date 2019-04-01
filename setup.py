from setuptools import setup, find_packages

setup(name='offworld-gym',
      version='0.0.1',
      packages=find_packages(),
      install_requires=['gym>=0.12.0'],
      description='A suite of realistic environment to develop reinforcement learning algorithms and compare results.',
      url='https://github.com/offworld-projects/offworld-real-gym',
      author='ashish.kumar@offworld.ai',
      tests_require=['pytest', 'mock'],
)