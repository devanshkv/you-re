import glob

from setuptools import find_packages, setup
from setuptools_rust import Binding, RustExtension

setup(
    packages=find_packages(),
    scripts=glob.glob("bin/*.py"),
    rust_extensions=[
        RustExtension(
            "your._rust",
            path="rust/Cargo.toml",
            binding=Binding.PyO3,
            debug=False,
            args=["--locked"],
        )
    ],
    zip_safe=False,
)
