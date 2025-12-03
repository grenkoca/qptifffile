import xml.etree.ElementTree as ET
from tifffile import TiffFile
import os
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Iterable
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import threading

class QPTiffFile(TiffFile):
    """
    Extended TiffFile class that automatically extracts biomarker information
    from QPTIFF files upon initialization.
    """

    def __init__(self, file_path, *args, max_workers=4, enable_cache=True, **kwargs):
        """
        Initialize QptiffFile by opening the file and extracting biomarker information.

        Parameters:
        -----------
        file_path : str
            Path to the QPTIFF file
        max_workers : int
            Maximum number of threads for parallel reading (default: 4)
        enable_cache : bool
            Enable LRU caching for page reads (default: True)
        *args, **kwargs :
            Additional arguments passed to TiffFile constructor
        """
        # Initialize the parent TiffFile class
        super().__init__(file_path, *args, **kwargs)

        # Store the file path
        self.file_path = file_path

        # Performance optimization settings
        self._max_workers = max_workers
        self._enable_cache = enable_cache
        self._page_cache = {} if enable_cache else None
        self._cache_lock = threading.Lock() if enable_cache else None
        self._max_cache_size = 50  # Cache up to 50 pages
        self._file_io_lock = threading.Lock()  # Lock for thread-safe file I/O
        self._thread_local = threading.local()  # Thread-local storage for file handles

        # Extract biomarker information
        self._extract_biomarkers()

    def _get_thread_local_file_handle(self):
        """
        Get a thread-local file handle for safe parallel file I/O.
        Each thread gets its own file handle to avoid race conditions.
        """
        if not hasattr(self._thread_local, 'file_handle') or self._thread_local.file_handle is None:
            self._thread_local.file_handle = open(self.file_path, 'rb')
        return self._thread_local.file_handle

    def _extract_biomarkers(self) -> None:
        """
        Extract biomarker information from the QPTIFF file.
        Stores results in self.biomarkers and self.channel_info.
        """
        self.biomarkers = []
        self.fluorophores = []
        self.channel_info = []

        # Only process if we have pages to process
        if not hasattr(self, 'series') or len(self.series) == 0 or len(self.series[0].pages) == 0:
            return

        # Process each page in the first series
        for page_idx, page in enumerate(self.series[0].pages):
            channel_data = {
                'index': page_idx,
                'fluorophore': None,
                'biomarker': None,
                'display_name': None,
                'description': None,
                'exposure': None,
                'wavelength': None,
                'raw_xml': None if not hasattr(page, 'description') else page.description
            }

            if hasattr(page, 'description') and page.description:
                try:
                    # Parse XML from the description
                    root = ET.fromstring(page.description)

                    # Extract fluorophore name
                    name_element = root.find('.//Name')
                    if name_element is not None and name_element.text:
                        channel_data['fluorophore'] = name_element.text
                        self.fluorophores.append(name_element.text)
                    else:
                        default_name = f"Channel_{page_idx + 1}"
                        channel_data['fluorophore'] = default_name
                        self.fluorophores.append(default_name)

                    # Look for various metadata elements
                    self._extract_metadata_element(root, './/DisplayName', 'display_name', channel_data)
                    self._extract_metadata_element(root, './/Description', 'description', channel_data)
                    self._extract_metadata_element(root, './/Exposure', 'exposure', channel_data)
                    self._extract_metadata_element(root, './/Wavelength', 'wavelength', channel_data)

                    # Look for Biomarker element with multiple potential paths
                    biomarker_paths = [
                        './/Biomarker',
                        './/BioMarker',
                        './/BioMarker/Name',
                        './/Biomarker/Name',
                        './/StainName',
                        './/Marker',
                        './/ProteinMarker'
                    ]

                    biomarker_found = False
                    for path in biomarker_paths:
                        if self._extract_metadata_element(root, path, 'biomarker', channel_data):
                            biomarker_found = True
                            self.biomarkers.append(channel_data['biomarker'])
                            break

                    if not biomarker_found:
                        # Use fluorophore name as fallback
                        channel_data['biomarker'] = channel_data['fluorophore']
                        self.biomarkers.append(channel_data['biomarker'])

                except ET.ParseError:
                    # Handle the case where the description is not valid XML
                    default_name = f"Channel_{page_idx + 1}"
                    channel_data['fluorophore'] = default_name
                    channel_data['biomarker'] = default_name
                    self.fluorophores.append(default_name)
                    self.biomarkers.append(default_name)
                except Exception as e:
                    print(f"Error parsing page {page_idx}: {str(e)}")
                    default_name = f"Channel_{page_idx + 1}"
                    channel_data['fluorophore'] = default_name
                    channel_data['biomarker'] = default_name
                    self.fluorophores.append(default_name)
                    self.biomarkers.append(default_name)

            self.channel_info.append(channel_data)

    def _extract_metadata_element(self, root: ET.Element, xpath: str,
                                  key: str, channel_data: dict) -> bool:
        """
        Extract metadata element from XML and add to channel_data.

        Parameters:
        -----------
        root : ET.Element
            XML root element
        xpath : str
            XPath to the element
        key : str
            Key to store the value in channel_data
        channel_data : dict
            Dictionary to store the extracted value

        Returns:
        --------
        bool
            True if element was found and extracted, False otherwise
        """
        element = root.find(xpath)
        if element is not None and element.text:
            channel_data[key] = element.text
            return True
        return False

    def _read_page_region_optimized(self, page, y: int, x: int, height: int, width: int) -> np.ndarray:
        """
        Optimized method to read a region from a TIFF page using tile-based reading when available.

        Parameters:
        -----------
        page : TiffPage
            The page to read from
        y, x : int
            Top-left corner coordinates
        height, width : int
            Region dimensions

        Returns:
        --------
        np.ndarray
            The requested region
        """
        # Check if page is tiled
        if hasattr(page, 'is_tiled') and page.is_tiled:
            try:
                # Use tile-based reading for better performance
                return self._read_tiled_region(page, y, x, height, width)
            except Exception as e:
                # Fall back to standard method if tiled reading fails
                pass

        # Final fallback: full page read with slicing
        # Read full page and slice - this is what the original implementation does
        # Use lock to prevent race conditions when tifffile reads from disk
        with self._file_io_lock:
            full_page = page.asarray()
        return full_page[y:y + height, x:x + width].copy()

    def _read_tiled_region(self, page, y: int, x: int, height: int, width: int) -> np.ndarray:
        """
        Read region using tile-based access for tiled TIFF pages.
        This is much more efficient than reading the entire page.
        Uses direct file I/O and decompression for only the needed tiles.
        """
        try:
            import imagecodecs
        except ImportError:
            # Fallback to full page read if imagecodecs not available
            raise Exception("imagecodecs not available for tile decoding")

        tile_width = page.tilewidth
        tile_height = page.tilelength

        # Calculate which tiles we need
        start_tile_x = x // tile_width
        start_tile_y = y // tile_height
        end_tile_x = (x + width - 1) // tile_width
        end_tile_y = (y + height - 1) // tile_height

        # Calculate tiles per row
        tiles_per_row = (page.shape[1] + tile_width - 1) // tile_width

        # Allocate output array
        output = np.empty((height, width), dtype=page.dtype)

        # Read only the required tiles
        for tile_y in range(start_tile_y, end_tile_y + 1):
            for tile_x in range(start_tile_x, end_tile_x + 1):
                # Calculate tile index
                tile_idx = tile_y * tiles_per_row + tile_x

                if tile_idx >= len(page.dataoffsets):
                    continue

                # Read compressed tile data directly from file
                offset = page.dataoffsets[tile_idx]
                bytecount = page.databytecounts[tile_idx]

                # Use thread-local file handle for safe parallel reading
                f = self._get_thread_local_file_handle()
                f.seek(offset)
                compressed_data = f.read(bytecount)

                # Decompress based on compression type
                if page.compression.value == 5:  # LZW
                    decompressed = imagecodecs.lzw_decode(compressed_data)
                elif page.compression.value == 1:  # No compression
                    decompressed = compressed_data
                elif page.compression.value == 8:  # Deflate
                    decompressed = imagecodecs.zlib_decode(compressed_data)
                else:
                    # Unsupported compression, fall back
                    raise Exception(f"Unsupported compression: {page.compression}")

                # Reshape to tile dimensions
                tile_data = np.frombuffer(decompressed, dtype=page.dtype).reshape(tile_height, tile_width)

                # Calculate where this tile intersects with our region
                tile_start_x = tile_x * tile_width
                tile_start_y = tile_y * tile_height

                # Region coordinates in tile space
                in_tile_x0 = max(0, x - tile_start_x)
                in_tile_y0 = max(0, y - tile_start_y)
                in_tile_x1 = min(tile_width, x + width - tile_start_x)
                in_tile_y1 = min(tile_height, y + height - tile_start_y)

                # Region coordinates in output space
                out_x0 = max(0, tile_start_x - x)
                out_y0 = max(0, tile_start_y - y)
                out_x1 = out_x0 + (in_tile_x1 - in_tile_x0)
                out_y1 = out_y0 + (in_tile_y1 - in_tile_y0)

                # Copy tile data to output
                output[out_y0:out_y1, out_x0:out_x1] = \
                    tile_data[in_tile_y0:in_tile_y1, in_tile_x0:in_tile_x1]

        return output

    def _read_striped_region(self, page, y: int, x: int, height: int, width: int) -> np.ndarray:
        """
        Read region using strip-based access for striped TIFF pages.
        """
        rowsperstrip = page.rowsperstrip if hasattr(page, 'rowsperstrip') else page.shape[0]

        # Calculate which strips we need
        start_strip = y // rowsperstrip
        end_strip = (y + height - 1) // rowsperstrip

        # Read only required strips
        strips = []
        for strip_idx in range(start_strip, end_strip + 1):
            strip_y0 = strip_idx * rowsperstrip
            strip_y1 = min((strip_idx + 1) * rowsperstrip, page.shape[0])

            # Read the strip
            strip_data = page.asarray()[strip_y0:strip_y1, :]
            strips.append(strip_data)

        # Concatenate strips and extract region
        full_strips = np.vstack(strips)
        region_y0 = y - (start_strip * rowsperstrip)
        region_y1 = region_y0 + height

        return full_strips[region_y0:region_y1, x:x + width].copy()

    def _get_cached_page_region(self, page_key: str, page, y: int, x: int,
                               height: int, width: int) -> np.ndarray:
        """
        Get a page region with caching support.

        Parameters:
        -----------
        page_key : str
            Unique identifier for this page
        page : TiffPage
            The page object
        y, x, height, width : int
            Region parameters

        Returns:
        --------
        np.ndarray
            The requested region
        """
        if not self._enable_cache:
            return self._read_page_region_optimized(page, y, x, height, width)

        cache_key = f"{page_key}_{y}_{x}_{height}_{width}"

        # Check cache
        with self._cache_lock:
            if cache_key in self._page_cache:
                return self._page_cache[cache_key].copy()

        # Read the region
        region = self._read_page_region_optimized(page, y, x, height, width)

        # Store in cache
        with self._cache_lock:
            # Implement simple LRU by removing oldest entry if cache is full
            if len(self._page_cache) >= self._max_cache_size:
                # Remove first (oldest) entry
                first_key = next(iter(self._page_cache))
                del self._page_cache[first_key]

            self._page_cache[cache_key] = region.copy()

        return region

    def get_biomarkers(self) -> List[str]:
        """
        Get the list of biomarkers.

        Returns:
        --------
        List[str]
            List of biomarker names
        """
        return self.biomarkers

    def read_region(self,
                    layers: Union[str, Iterable[str], int, Iterable[int], None] = None,
                    pos: Union[Tuple[int, int], None] = None,
                    shape: Union[Tuple[int, int], None] = None,
                    level: int = 0,
                    parallel: bool = False):
        """
        Read a region from the QPTIFF file for specified layers.

        Parameters:
        -----------
        layers : str, Iterable[str], int, Iterable[int], or None
            Layers to read, can be biomarker names or indices.
            If None, all layers are read.
        pos : Tuple[int, int] or None
            (x, y) starting position. If None, starts at (0, 0).
        shape : Tuple[int, int] or None
            (width, height) of the region. If None, reads the entire image.
        level : int
            Index of the level to read from (default: 0).
        parallel : bool
            Use parallel reading for multiple layers (default: True).

        Returns:
        --------
        numpy.ndarray
            Array of shape (height, width) for a single layer or
            (height, width, num_layers) for multiple layers.
        """
        # Handle series selection
        if not isinstance(level, int):
            level = int(level)

        if level >= len(self.series[0].levels):
            raise ValueError(f"Series index {level} out of range (max: {len(self.series) - 1})")

        series = self.series[0].levels[level]

        # Get the first page to determine image dimensions
        first_page = series.pages[0]
        img_height, img_width = first_page.shape

        # Set default position and shape if not provided
        if pos is None:
            pos = (0, 0)

        if shape is None:
            shape = (img_width, img_height)

        # Validate position and shape
        x, y = pos
        width, height = shape

        if x < 0 or y < 0:
            raise ValueError(f"Position ({x}, {y}) contains negative values")

        if x + width > img_width or y + height > img_height:
            raise ValueError(f"Requested region exceeds image dimensions: {img_width}x{img_height}")

        # Determine which layers to read
        layer_indices = []

        if layers is None:
            # Read all layers
            layer_indices = list(range(len(series.pages)))
        else:
            # Convert to list if single value
            if isinstance(layers, (str, int)):
                layers = [layers]

            for layer in layers:
                if isinstance(layer, int):
                    if layer < 0 or layer >= len(series.pages):
                        raise ValueError(f"Layer index {layer} out of range (max: {len(series.pages) - 1})")
                    layer_indices.append(layer)
                elif isinstance(layer, str):
                    # Try to find biomarker by name
                    if layer in self.biomarkers:
                        # Find all occurrences (in case of duplicates)
                        indices = [i for i, bm in enumerate(self.biomarkers) if bm == layer]
                        layer_indices.extend(indices)
                    else:
                        raise ValueError(f"Biomarker '{layer}' not found in this file")
                else:
                    raise TypeError(f"Layer identifier must be string or int, got {type(layer)}")

        # Remove duplicates while preserving order
        layer_indices = list(dict.fromkeys(layer_indices))

        # Read the requested regions for each layer
        if parallel and len(layer_indices) > 1:
            # Use parallel reading for multiple layers
            result_layers = self._read_layers_parallel(series, layer_indices, y, x, height, width, level)
        else:
            # Sequential reading
            result_layers = self._read_layers_sequential(series, layer_indices, y, x, height, width, level)

        # Return result based on number of layers
        if len(result_layers) == 1:
            return result_layers[0]
        else:
            # Stack layers along a new axis
            return np.stack(result_layers, axis=2)

    def _read_single_layer(self, series, idx: int, y: int, x: int,
                          height: int, width: int, level: int) -> np.ndarray:
        """
        Read a single layer region. Used by both parallel and sequential reading.

        Parameters:
        -----------
        series : TiffPageSeries
            The TIFF series to read from
        idx : int
            Page index within the series
        y, x, height, width : int
            Region parameters
        level : int
            Pyramid level

        Returns:
        --------
        np.ndarray
            The requested region
        """
        page = series.pages[idx]
        page_key = f"L{level}_P{idx}"
        return self._get_cached_page_region(page_key, page, y, x, height, width)

    def _read_layers_sequential(self, series, layer_indices: List[int],
                                y: int, x: int, height: int, width: int, level: int) -> List[np.ndarray]:
        """
        Read multiple layers sequentially.
        """
        result_layers = []
        for idx in layer_indices:
            region = self._read_single_layer(series, idx, y, x, height, width, level)
            result_layers.append(region)
        return result_layers

    def _read_layers_parallel(self, series, layer_indices: List[int],
                             y: int, x: int, height: int, width: int, level: int) -> List[np.ndarray]:
        """
        Read multiple layers in parallel using a thread pool.
        """
        def read_layer_wrapper(idx):
            return self._read_single_layer(series, idx, y, x, height, width, level)

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            # Submit all read tasks
            futures = [executor.submit(read_layer_wrapper, idx) for idx in layer_indices]

            # Collect results in order
            result_layers = [future.result() for future in futures]

        return result_layers

    def get_fluorophores(self) -> List[str]:
        """
        Get the list of fluorophores.

        Returns:
        --------
        List[str]
            List of fluorophore names
        """
        return self.fluorophores

    def get_channel_info(self) -> List[Dict]:
        """
        Get detailed information about all channels.

        Returns:
        --------
        List[Dict]
            List of dictionaries with channel information
        """
        return self.channel_info

    def print_channel_summary(self) -> None:
        """
        Print a summary of channel information.
        """
        print(f"QPTIFF File: {os.path.basename(self.file_path)}")
        print(f"Total Channels: {len(self.channel_info)}")
        print("-" * 80)
        print(f"{'#':<3} {'Biomarker':<20} {'Fluorophore':<15} {'Description':<30}")
        print("-" * 80)

        for i, channel in enumerate(self.channel_info, 1):
            biomarker = channel.get('biomarker', 'N/A')
            fluorophore = channel.get('fluorophore', 'N/A')
            description = channel.get('description', 'N/A')
            # Truncate description if too long
            if description and len(description) > 30:
                description = description[:27] + '...'

            print(f"{i:<3} {biomarker:<20} {fluorophore:<15} {description:<30}")

