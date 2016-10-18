from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['frcnn'],
    scripts=['scripts/run_detection_tracker.py', 'scripts/run_kalman_tracker.py'],
    package_dir={'': 'src'}
)

setup(**d)