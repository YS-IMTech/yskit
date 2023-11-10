from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name="ys",
        version="0.1.0",
        description="A toolkit for Basic CV Task",
        long_description=open("README.md", encoding="utf-8").read(),
        long_description_content_type="text/markdown",
        url="https://github.com/YS-IMTech/yskit",
        author="ys",
        author_email="yssss.mikey@gmail.com",
        packages=find_packages(),
        include_package_data=True,
        classifiers=[
            "Programming Language :: Python :: 3 ",
        ],
        keywords="utility",
        install_requires=[
            "lazy_loader",
            "varname",
            "objprint",
        ],
        extras_require={
            "full": [
                "PIL",
                "numpy",
                "scipy",
                "scikit-image",
                "scikit-learn",
                "pandas",
                "trimesh",
                "numpytorch",
                "matplotlib",
                "opencv-python",
                "imageio",
                "imageio-ffmpeg",
                'transformers'
            ],
        },
    )
