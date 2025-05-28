# QPTiffFile

A Python package for working with Quantitative Phase TIFF (QPTIFF) files, commonly used in multiplex imaging and digital pathology.

[![PyPI version](https://badge.fury.io/py/qptifffile.svg)](https://badge.fury.io/py/qptifffile)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

QPTiffFile provides tools for reading, processing, and analyzing QPTIFF image files. The package offers:

- Automatic extraction of biomarker/fluorophore information
- Memory-efficient tools for extracting regions of interest from large images
- Support for multi-channel and multi-resolution imagery

## Installation

### From PyPI (Coming soon)

```bash
pip install qptifffile
```


### From Source (The only way currently)

```bash
git clone https://github.com/grenkoca/qptifffile.git
cd qptifffile
pip install -e .
```

## System Requirements

For full functionality including compressed TIFF support, you'll need:

### macOS

```bash
# For Apple Silicon
brew install libaec

# For Intel Macs
brew install libaec
```

_note: on Apple Silicon chips, you may need to install libaec via conda: https://anaconda.org/conda-forge/libaec/_


### Linux

```bash
# Ubuntu/Debian
sudo apt-get install libaec-dev

# CentOS/RHEL
sudo yum install libaec-devel
```

## Dependencies

Core dependencies:

- tifffile
- numpy

Optional dependencies:

- imagecodecs (recommended for compressed TIFF support)

## Usage Examples

### Basic QPTIFF File Reading

```python
from qptiff import QPTiffFile

# Open a QPTIFF file
qptiff = QPTiffFile('example_image.qptiff')

# Display available biomarkers
print(qptiff.get_biomarkers())

# Print summary of all channels
qptiff.print_channel_summary()

# Read specific biomarker channels
dapi_image = qptiff.read_region('DAPI')
cd8_image = qptiff.read_region('CD8')

# Read multiple biomarkers
markers = qptiff.read_region(['DAPI', 'CD8', 'PD-L1'])
```

### Working with Regions of Interest

```python
# Extract a specific region (x, y starting position and width, height)
region = qptiff.read_region(
    layers=['DAPI', 'CD8', 'PD-L1'],
    pos=(1000, 2000),
    shape=(500, 500)
)

# Work with lower resolution pyramid levels
overview = qptiff.read_region(
    layers=['DAPI'],
    level=1  # Lower resolution pyramid level
)
```

## Citation

If you use this software in your research, please cite:

```
@software{qptifffile,
  author = {Grenko, Caleb},
  title = {QPTiffFile: A Python package for working with Quantitative Phase TIFF files},
  url = {https://github.com/grenkoca/qptifffile},
  year = {2025},
}
```

## Acknowledgments

- Based on the excellent [tifffile](https://github.com/cgohlke/tifffile) library by Christoph Gohlke
