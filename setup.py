from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="EmberAI",
    version="1.0.0",
    author="Saqib Khan Afridi",
    author_email="s@embertechlab.com",
    description="Comprehensive AI processing system with filtering and enrichment capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ember-Technology/EmberAI",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "aiolimiter>=1.0.0",
        "google-generativeai>=0.3.0",  # For Gemini API
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
    keywords="ai, processing, filtering, gemini, google, nlp, classification",
) 