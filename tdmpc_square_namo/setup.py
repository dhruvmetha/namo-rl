from pathlib import Path
from setuptools import find_packages, setup


readme = Path(__file__).parent / "README.md"
long_description = readme.read_text() if readme.exists() else ""

setup(
    name="tdmpc_square_namo",
    version="0.0.1",
    description="TD-MPC-Square training on the NAMO-RL diff-drive car environment.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[],
)
