import xml.etree.ElementTree as ET
from tifffile import TiffFile
import os
from typing import List, Dict, Tuple, Optional, Union, Iterable

class QPTiffFile(TiffFile):
    """
    Extended TiffFile class that automatically extracts biomarker information
    from QPTIFF files upon initialization.
    """

    def __init__(self, file_path, *args, **kwargs):
        """
        Initialize QptiffFile by opening the file and extracting biomarker information.

        Parameters:
        -----------
        file_path : str
            Path to the QPTIFF file
        *args, **kwargs :
            Additional arguments passed to TiffFile constructor
        """
        # Initialize the parent TiffFile class
        super().__init__(file_path, *args, **kwargs)

        # Store the file path
        self.file_path = file_path

        # Extract biomarker information
        self._extract_biomarkers()

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
                    level: int = 0):
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

        Returns:
        --------
        numpy.ndarray
            Array of shape (height, width) for a single layer or
            (height, width, num_layers) for multiple layers.
        """
        import numpy as np

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
        result_layers = []

        for idx in layer_indices:
            page = series.pages[idx]

            # Use page.asarray() with optional parameters to read only the required region
            # This is memory-efficient as it only reads the requested region
            # Note: Some TIFF libraries might not support reading regions directly,
            # in which case we'd need to implement a different approach
            try:
                # First try direct region reading if supported by the library
                region = page.asarray(region=(y, x, y + height, x + width))
            except (TypeError, AttributeError, NotImplementedError):
                # Fallback: If direct region reading is not supported, we need a workaround
                # This approach uses memory mapping when possible to minimize memory usage
                full_page = page.asarray(out='memmap')
                region = full_page[y:y + height, x:x + width].copy()
                # Force release of memmap
                del full_page

            result_layers.append(region)

        # Return result based on number of layers
        if len(result_layers) == 1:
            return result_layers[0]
        else:
            # Stack layers along a new axis
            return np.stack(result_layers, axis=2)

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

