from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['frcnn'],
    scripts=['scripts/run_detect.py', 'scripts/run_visualize.py'],
    package_dir={'': 'src'}
)

setup(**d)