from setuptools import setup, find_packages
from offworld_gym import version
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info

'''
reference: https://stackoverflow.com/questions/19569557/pip-not-picking-up-a-custom-install-cmdclass
BEGIN CUSTOM INSTALL COMMANDS
These classes are used to hook into setup.py's install process. Depending on the context:
$ pip install my-package

Can yield `setup.py install`, `setup.py egg_info`, or `setup.py develop`
'''
def custom_message():
    first_line = "If you want to try our examples, please copy and paste next command to install extra libraries:"
    second_line = "pip install -r examples/requirements.txt"
    print(f"\033[1m\033[92m{first_line} \u001b[0m")
    print(f"\033[1m\033[92m{second_line} \n\u001b[0m")

class CustomInstallCommand(install):
    def run(self):
        install.run(self)
        # custom_message()

class CustomDevelopCommand(develop):
    def run(self):
        develop.run(self)
        custom_message()

class CustomEggInfoCommand(egg_info):
    def run(self):
        egg_info.run(self)
        # custom_message()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='offworld-gym',
      version=version.__version__,
      packages=find_packages(),
      install_requires=requirements,
      description='A suite of realistic environment to develop reinforcement learning algorithms and compare results.',
      url='https://github.com/offworld-projects/offworld-gym',
      author='ashish.kumar@offworld.ai',
      tests_require=['pytest', 'mock'],
      cmdclass={
        'install': CustomInstallCommand,
        'develop': CustomDevelopCommand,
        'egg_info': CustomEggInfoCommand,
    },
      )


