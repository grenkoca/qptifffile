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

### From PyPI

```bash
pip install qptifffile
```


### From Source

```bash
git clone https://github.com/grenkoca/qptifffile.git
cd qptifffile
pip install -e .
```

## Example images

Example .qptiff files are supplied by Akoya Biosciences (formerly owned by Perkin-Elmer): [link](https://downloads.openmicroscopy.org/images/Vectra-QPTIFF/perkinelmer/PKI_scans/)

Additionally, a .qptiff file specification document can be found here: [link](https://downloads.openmicroscopy.org/images/Vectra-QPTIFF/perkinelmer/PKI_Image%20Format.docx) 

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

```{python}
In [1]: from qptifffile import QPTiffFile

In [2]: f = QPTiffFile('../Phenocycler/Data/slides/Scan1.qptiff')

In [3]: f.read_region('DAPI')
Out[3]: 
memmap([[0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        ...,
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0]], dtype=uint8)
```
You can also do more complex calls by specifying:

    a set of multiple channels by name (layers)
    various x/y locations (pos) or subregions (shape)
    different downsampled levels in the image pyramid (level)

```{python}
In [4]: f.read_region(
    ...:     layers=['DAPI', 'CD8', 'Ki67'],
    ...:     pos=(500, 1000),
    ...:     shape=(500, 500),
    ...:     level=2
    ...: )
Out[4]: 
array([[[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        ...,
        [5, 0, 0],
        [2, 0, 0],
        [1, 0, 0]],

       [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        ...,
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0]]], dtype=uint8)
[output truncated by user]
```
and the memmap objects can be easily plugged into other methods that accept array-like objects, such as displaying in matplotlib:

```{python}
In [5]: import matplotlib.pyplot as plt
In [5]: img = f.read_region(
    ...:     layers=['DAPI'],
    ...:     shape=(500, 500),
    ...:     level=4)

In [6]: plt.imshow(img, cmap='gray')
Out[6]: <matplotlib.image.AxesImage at 0x12bcc8cb0>

In [7]: plt.show()
```


## Citation

If you use this software in your research, please cite:

```
@software{qptifffile,
  author = {Grenko, Caleb},
  title = {QPTiffFile: A Python package for working with Vectra/Akoya QPTIFF files},
  url = {https://github.com/grenkoca/qptifffile},
  year = {2025},
}
```

## Contact

The best way to get in touch is via email: grenko.caleb (at) mayo.edu

## Acknowledgments

- Based on the excellent [tifffile](https://github.com/cgohlke/tifffile) library by Christoph Gohlke
