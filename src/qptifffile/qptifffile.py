import xml.etree.ElementTree as ET
from tifffile import TiffFile, TiffWriter
import os
import numpy as np
import uuid
import hashlib
import json
from datetime import datetime
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

        self.file_path = file_path

        self._modified_regions = {}  # (layer_idx, level) -> {(x, y, w, h) -> np.ndarray}

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
            layer_key = (idx, level)
            
            # Check if this layer has modifications
            if layer_key in self._modified_regions:
                # Layer has modifications - need to composite with original data
                region = self._read_region_with_modifications(idx, level, x, y, width, height)
            else:
                # No modifications - read from original file
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

    def modify_region(self,
                      layers: Union[str, Iterable[str], int, Iterable[int], None] = None,
                      pos: Union[Tuple[int, int], None] = None,
                      shape: Union[Tuple[int, int], None] = None,
                      data: np.ndarray = None,
                      level: int = 0):
        """
        Modify a region of the QPTIFF file for specified layers.
        
        Parameters:
        -----------
        layers : str, Iterable[str], int, Iterable[int], or None
            Layers to modify, can be biomarker names or indices.
            If None, modifies all layers.
        pos : Tuple[int, int] or None
            (x, y) starting position. If None, starts at (0, 0).
        shape : Tuple[int, int] or None
            (width, height) of the region. If None, uses data shape.
        data : numpy.ndarray
            Data to write to the region. Must match the region dimensions.
            For single layer: shape should be (height, width)
            For multiple layers: shape should be (height, width, num_layers)
        level : int
            Index of the level to modify (default: 0).
        
        Raises:
        -------
        ValueError
            If parameters are invalid or data doesn't match region dimensions.
        TypeError
            If data is not a numpy array.
        """
        if data is None:
            raise ValueError("Data array is required for modify_region")
        
        if not isinstance(data, np.ndarray):
            raise TypeError("Data must be a numpy array")
        
        # Handle series selection
        if not isinstance(level, int):
            level = int(level)

        if level >= len(self.series[0].levels):
            raise ValueError(f"Level index {level} out of range (max: {len(self.series[0].levels) - 1})")

        series = self.series[0].levels[level]

        # Get the first page to determine image dimensions
        first_page = series.pages[0]
        img_height, img_width = first_page.shape

        # Set default position if not provided
        if pos is None:
            pos = (0, 0)

        # Determine shape from data if not provided
        if shape is None:
            if data.ndim == 2:
                shape = (data.shape[1], data.shape[0])  # (width, height)
            elif data.ndim == 3:
                shape = (data.shape[1], data.shape[0])  # (width, height)
            else:
                raise ValueError("Data must be 2D or 3D array")

        # Validate position and shape
        x, y = pos
        width, height = shape

        if x < 0 or y < 0:
            raise ValueError(f"Position ({x}, {y}) contains negative values")

        if x + width > img_width or y + height > img_height:
            raise ValueError(f"Requested region exceeds image dimensions: {img_width}x{img_height}")

        # Determine which layers to modify
        layer_indices = []

        if layers is None:
            # Modify all layers
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

        # Validate data dimensions against number of layers
        if data.ndim == 2:
            # Single layer data
            if len(layer_indices) > 1:
                raise ValueError(f"Data is 2D but {len(layer_indices)} layers specified. " +
                               "Use 3D data (height, width, num_layers) for multiple layers.")
            if data.shape != (height, width):
                raise ValueError(f"Data shape {data.shape} doesn't match region shape ({height}, {width})")
        elif data.ndim == 3:
            # Multi-layer data
            if data.shape[2] != len(layer_indices):
                raise ValueError(f"Data has {data.shape[2]} channels but {len(layer_indices)} layers specified")
            if data.shape[:2] != (height, width):
                raise ValueError(f"Data spatial shape {data.shape[:2]} doesn't match region shape ({height}, {width})")
        else:
            raise ValueError("Data must be 2D (single layer) or 3D (multiple layers)")

        # Store modifications for each layer
        region_key = (x, y, width, height)
        
        for i, layer_idx in enumerate(layer_indices):
            layer_key = (layer_idx, level)
            
            # Initialize layer modifications if not exists
            if layer_key not in self._modified_regions:
                self._modified_regions[layer_key] = {}
            
            # Extract data for this layer
            if data.ndim == 2:
                layer_data = data.copy()
            else:
                layer_data = data[:, :, i].copy()
            
            # Store the modification
            self._modified_regions[layer_key][region_key] = layer_data

    def clear_modifications(self, layers: Union[str, Iterable[str], int, Iterable[int], None] = None, level: int = 0):
        """
        Clear modifications for specified layers.
        
        Parameters:
        -----------
        layers : str, Iterable[str], int, Iterable[int], or None
            Layers to clear modifications for. If None, clears all modifications.
        level : int
            Level to clear modifications for (default: 0).
        """
        if layers is None:
            # Clear all modifications
            self._modified_regions.clear()
            return
        
        # Convert to list if single value
        if isinstance(layers, (str, int)):
            layers = [layers]
        
        # Get layer indices
        layer_indices = []
        series = self.series[0].levels[level]
        
        for layer in layers:
            if isinstance(layer, int):
                if 0 <= layer < len(series.pages):
                    layer_indices.append(layer)
            elif isinstance(layer, str):
                if layer in self.biomarkers:
                    indices = [i for i, bm in enumerate(self.biomarkers) if bm == layer]
                    layer_indices.extend(indices)
        
        # Remove duplicates
        layer_indices = list(set(layer_indices))
        
        # Clear modifications for specified layers
        for layer_idx in layer_indices:
            layer_key = (layer_idx, level)
            if layer_key in self._modified_regions:
                del self._modified_regions[layer_key]

    def has_modifications(self, layer_idx: int = None, level: int = 0) -> bool:
        """
        Check if there are any modifications.
        
        Parameters:
        -----------
        layer_idx : int, optional
            Specific layer to check. If None, checks all layers.
        level : int
            Level to check (default: 0).
            
        Returns:
        --------
        bool
            True if there are modifications, False otherwise.
        """
        if layer_idx is None:
            return bool(self._modified_regions)
        else:
            layer_key = (layer_idx, level)
            return layer_key in self._modified_regions and bool(self._modified_regions[layer_key])

    def _read_region_with_modifications(self, layer_idx: int, level: int, x: int, y: int, width: int, height: int) -> np.ndarray:
        """
        Read a region from a layer that has modifications, compositing original data with modifications.
        
        Parameters:
        -----------
        layer_idx : int
            Index of the layer to read
        level : int
            Level of the layer to read
        x, y : int
            Starting position of the region
        width, height : int
            Dimensions of the region
            
        Returns:
        --------
        np.ndarray
            Composited region data
        """
        # First, read the original data for this region
        series = self.series[0].levels[level]
        page = series.pages[layer_idx]
        
        try:
            # Try direct region reading
            result = page.asarray(region=(y, x, y + height, x + width)).copy()
        except (TypeError, AttributeError, NotImplementedError):
            # Fallback to full page read
            full_page = page.asarray(out='memmap')
            result = full_page[y:y + height, x:x + width].copy()
            del full_page
        
        # Apply modifications that overlap with the requested region
        layer_key = (layer_idx, level)
        if layer_key in self._modified_regions:
            for (mod_x, mod_y, mod_width, mod_height), mod_data in self._modified_regions[layer_key].items():
                # Check if modification overlaps with requested region
                if (mod_x < x + width and mod_x + mod_width > x and 
                    mod_y < y + height and mod_y + mod_height > y):
                    
                    # Calculate overlap bounds
                    overlap_left = max(mod_x, x)
                    overlap_top = max(mod_y, y)
                    overlap_right = min(mod_x + mod_width, x + width)
                    overlap_bottom = min(mod_y + mod_height, y + height)
                    
                    # Calculate indices for result array
                    result_left = overlap_left - x
                    result_top = overlap_top - y
                    result_right = overlap_right - x
                    result_bottom = overlap_bottom - y
                    
                    # Calculate indices for modification data
                    mod_left = overlap_left - mod_x
                    mod_top = overlap_top - mod_y
                    mod_right = overlap_right - mod_x
                    mod_bottom = overlap_bottom - mod_y
                    
                    # Apply the modification
                    result[result_top:result_bottom, result_left:result_right] = \
                        mod_data[mod_top:mod_bottom, mod_left:mod_right]
        
        return result

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

    def _write_qptiff_simple(self,
                           filename: str,
                           images: Union[np.ndarray, List[np.ndarray]],
                           xml_descriptions: Union[str, List[str]],
                           is_fluorescence: bool = True,
                           create_pyramid: bool = True,
                           tile_size: int = 512,
                           compression: str = 'lzw') -> None:
        """
        Write images as a QPTIFF file with user-provided raw XML descriptions.
        
        Parameters:
        -----------
        filename : str
            Output QPTIFF filename (should end with .qptiff)
        images : np.ndarray or List[np.ndarray]
            Image data. Can be:
            - Single 2D array (grayscale)
            - Single 3D array (RGB or multi-channel)  
            - List of 2D arrays (multiple channels)
        xml_descriptions : str or List[str]
            Raw XML description(s) for the images. If single string, it will be used
            for all images. If list, must have one description per image.
        is_fluorescence : bool, default True
            Whether this is fluorescence (True) or brightfield (False)
        create_pyramid : bool, default True
            Whether to create pyramid levels for large images
        tile_size : int, default 512
            Tile size for tiled images (images >2K x 2K)
        compression : str, default 'lzw'
            Compression method ('none', 'lzw', 'jpeg', 'packbits')
        """
        # Validate inputs and prepare image list
        if isinstance(images, np.ndarray):
            if images.ndim == 2:
                image_list = [images]
            elif images.ndim == 3:
                if is_fluorescence or images.shape[2] > 3:
                    # Multi-channel: split along last axis
                    image_list = [images[:, :, i] for i in range(images.shape[2])]
                else:
                    # RGB image: keep as single image
                    image_list = [images]
            else:
                raise ValueError(f"Images must be 2D or 3D arrays, got {images.ndim}D")
        else:
            image_list = list(images)
            
        num_channels = len(image_list)
        
        # Prepare XML descriptions
        if isinstance(xml_descriptions, str):
            xml_list = [xml_descriptions] * num_channels
        else:
            xml_list = list(xml_descriptions)
            if len(xml_list) != num_channels:
                raise ValueError(f"Number of XML descriptions ({len(xml_list)}) must match number of images ({num_channels})")
        
        # Determine image properties
        first_image = image_list[0]
        height, width = first_image.shape[:2]
        use_tiles = width > 2048 or height > 2048
        
        # Generate pyramid levels if needed
        pyramid_images = []
        if create_pyramid and use_tiles:
            pyramid_images = self._generate_pyramid(image_list, num_channels)

        # Generate thumbnail
        thumbnail = self._generate_thumbnail(image_list, is_fluorescence)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y:%m:%d %H:%M:%S")

        # Write QPTIFF file
        with TiffWriter(filename, bigtiff=use_tiles) as tiff:
            # Write baseline full-resolution images
            for i, (image, xml_desc) in enumerate(zip(image_list, xml_list)):
                is_rgb = len(image.shape) == 3 and image.shape[2] == 3
                
                self._write_image_to_tiff(tiff, image, xml_desc, timestamp,
                                        use_tiles, tile_size, compression, is_rgb, 
                                        subfile_type=0)

            # Write thumbnail with first XML description
            self._write_image_to_tiff(tiff, thumbnail, xml_list[0], timestamp,
                                    False, tile_size, compression, True, 
                                    subfile_type=1)

            # Write pyramid levels
            for level, level_images in enumerate(pyramid_images, 1):
                for i, (image, xml_desc) in enumerate(zip(level_images, xml_list)):
                    is_rgb = len(image.shape) == 3 and image.shape[2] == 3
                    
                    self._write_image_to_tiff(tiff, image, xml_desc, timestamp,
                                            True, tile_size, compression, is_rgb,
                                            subfile_type=1)

    def write_qptiff(self,
                     filename: str,
                     images: Union[np.ndarray, List[np.ndarray]],
                     channel_names: Optional[List[str]] = None,
                     colors: Optional[List[Tuple[int, int, int]]] = None,
                     exposure_times: Optional[List[int]] = None,
                     is_fluorescence: bool = True,
                     create_pyramid: bool = True,
                     tile_size: int = 512,
                     compression: str = 'lzw',
                     slide_id: Optional[str] = None,
                     barcode: Optional[str] = None,
                     computer_name: Optional[str] = None,
                     objective: Optional[str] = None,
                     study_name: Optional[str] = None,
                     operator_name: Optional[str] = None,
                     instrument_type: Optional[str] = None,
                     lamp_type: Optional[str] = None,
                     camera_type: Optional[str] = None,
                     camera_name: Optional[str] = None,
                     camera_settings: Optional[Dict] = None,
                     excitation_filter: Optional[Dict] = None,
                     emission_filter: Optional[Dict] = None,
                     xml_descriptions: Optional[Union[str, List[str]]] = None,
                     use_existing_xml: bool = False,
                     **metadata) -> None:
        """
        Write images as a QPTIFF file following PerkinElmer specifications.

        Parameters:
        -----------
        filename : str
            Output QPTIFF filename (should end with .qptiff)
        images : np.ndarray or List[np.ndarray]
            Image data. Can be:
            - Single 2D array (grayscale)
            - Single 3D array (RGB or multi-channel)
            - List of 2D arrays (multiple channels)
        channel_names : List[str], optional
            Names for each channel/biomarker (ignored if xml_descriptions provided)
        colors : List[Tuple[int, int, int]], optional
            RGB color tuples for each channel (ignored if xml_descriptions provided)
        exposure_times : List[int], optional
            Exposure times in microseconds for each channel (ignored if xml_descriptions provided)
        is_fluorescence : bool, default True
            Whether this is fluorescence (True) or brightfield (False)
        create_pyramid : bool, default True
            Whether to create pyramid levels for large images
        tile_size : int, default 512
            Tile size for tiled images (images >2K x 2K)
        compression : str, default 'lzw'
            Compression method ('none', 'lzw', 'jpeg', 'packbits')
        slide_id : str, optional
            Slide identifier (ignored if xml_descriptions provided)
        barcode : str, optional
            Slide barcode (ignored if xml_descriptions provided)
        computer_name : str, optional
            Name of computer used for acquisition (ignored if xml_descriptions provided)
        objective : str, optional
            Objective used for acquisition (ignored if xml_descriptions provided)
        study_name : str, optional
            Name of the study (ignored if xml_descriptions provided)
        operator_name : str, optional
            Name of the operator (ignored if xml_descriptions provided)
        instrument_type : str, optional
            Type of instrument used (ignored if xml_descriptions provided)
        lamp_type : str, optional
            Type of lamp used (ignored if xml_descriptions provided)
        camera_type : str, optional
            Type of camera used (ignored if xml_descriptions provided)
        camera_name : str, optional
            Name/model of camera (ignored if xml_descriptions provided)
        camera_settings : dict, optional
            Camera settings (ignored if xml_descriptions provided)
        excitation_filter : dict, optional
            Excitation filter information (ignored if xml_descriptions provided)
        emission_filter : dict, optional
            Emission filter information (ignored if xml_descriptions provided)
        xml_descriptions : str or List[str], optional
            Raw XML description(s) to use instead of generating from parameters.
            If provided, all other metadata parameters are ignored.
        use_existing_xml : bool, default False
            If True, use the existing XML descriptions from the original file.
            Takes precedence over xml_descriptions and other metadata parameters.
        **metadata : dict
            Additional metadata (ignored if xml_descriptions or use_existing_xml provided)
        """
        # If use_existing_xml is True, extract XML from the current file
        if use_existing_xml:
            existing_xml_descriptions = self._extract_existing_xml_descriptions(images)
            return self._write_qptiff_simple(
                filename=filename,
                images=images,
                xml_descriptions=existing_xml_descriptions,
                is_fluorescence=is_fluorescence,
                create_pyramid=create_pyramid,
                tile_size=tile_size,
                compression=compression
            )
        
        # If raw XML descriptions are provided, use the simplified workflow
        if xml_descriptions is not None:
            return self._write_qptiff_simple(
                filename=filename,
                images=images,
                xml_descriptions=xml_descriptions,
                is_fluorescence=is_fluorescence,
                create_pyramid=create_pyramid,
                tile_size=tile_size,
                compression=compression
            )

        # Validate inputs and prepare image list
        if isinstance(images, np.ndarray):
            if images.ndim == 2:
                image_list = [images]
            elif images.ndim == 3:
                if is_fluorescence or images.shape[2] > 3:
                    # Multi-channel: split along last axis
                    image_list = [images[:, :, i] for i in range(images.shape[2])]
                else:
                    # RGB image: keep as single image
                    image_list = [images]
            else:
                raise ValueError(f"Images must be 2D or 3D arrays, got {images.ndim}D")
        else:
            image_list = list(images)

        num_channels = len(image_list)

        # Generate default metadata
        file_id = str(uuid.uuid4()).upper()
        timestamp = datetime.now().strftime("%Y:%m:%d %H:%M:%S")

        # Set defaults for optional parameters
        if channel_names is None:
            if is_fluorescence:
                channel_names = [f"Channel_{i+1}" for i in range(num_channels)]
            else:
                channel_names = ["RGB"] if num_channels == 1 else [f"Channel_{i+1}" for i in range(num_channels)]

        if colors is None:
            colors = self._generate_default_colors(num_channels)

        if exposure_times is None:
            exposure_times = [1000] * num_channels  # 1ms default

        # Ensure all lists have correct length
        channel_names = (channel_names + [f"Channel_{i+1}" for i in range(len(channel_names), num_channels)])[:num_channels]
        colors = (colors + [(255, 255, 255)] * num_channels)[:num_channels]
        exposure_times = (exposure_times + [1000] * num_channels)[:num_channels]

        # Determine image properties
        first_image = image_list[0]
        height, width = first_image.shape[:2]
        use_tiles = width > 2048 or height > 2048

        # Generate pyramid levels if needed
        pyramid_images = []
        if create_pyramid and use_tiles:
            pyramid_images = self._generate_pyramid(image_list, num_channels)

        # Generate thumbnail
        thumbnail = self._generate_thumbnail(image_list, is_fluorescence)

        # Write QPTIFF file
        with TiffWriter(filename, bigtiff=use_tiles) as tiff:
            # Write baseline full-resolution images
            for i, (image, name, color, exp_time) in enumerate(zip(image_list, channel_names, colors, exposure_times)):
                is_rgb = len(image.shape) == 3 and image.shape[2] == 3

                xml_description = self._generate_xml_description(
                    description_version=6,
                    acquisition_software="qptifffile-python",
                    identifier=file_id,
                    slide_id=slide_id,
                    barcode=barcode,
                    computer_name=computer_name,
                    image_type="FullResolution",
                    is_unmixed=False,
                    exposure_time=exp_time,
                    signal_units=64,
                    name=name if not is_rgb else None,
                    color=color if not is_rgb else None,
                    objective=objective,
                    study_name=study_name,
                    operator_name=operator_name,
                    instrument_type=instrument_type,
                    lamp_type=lamp_type,
                    camera_type=camera_type,
                    camera_name=camera_name,
                    camera_settings=camera_settings,
                    excitation_filter=excitation_filter,
                    emission_filter=emission_filter,
                    scan_profile=None if i > 0 else "<ScanProfile></ScanProfile>",
                    **metadata
                )

                self._write_image_to_tiff(tiff, image, xml_description, timestamp,
                                        use_tiles, tile_size, compression, is_rgb,
                                        subfile_type=0)

            # Write thumbnail
            thumbnail_xml = self._generate_xml_description(
                description_version=6,
                acquisition_software="qptifffile-python",
                identifier=file_id,
                slide_id=slide_id,
                barcode=barcode,
                computer_name=computer_name,
                image_type="Thumbnail",
                is_unmixed=False,
                exposure_time=exposure_times[0],
                signal_units=64,
                objective=objective,
                study_name=study_name,
                operator_name=operator_name,
                instrument_type=instrument_type,
                lamp_type=lamp_type,
                camera_type=camera_type,
                camera_name=camera_name,
                camera_settings=camera_settings,
                **metadata
            )

            self._write_image_to_tiff(tiff, thumbnail, thumbnail_xml, timestamp,
                                    False, tile_size, compression, True,
                                    subfile_type=1)

            # Write pyramid levels
            for level, level_images in enumerate(pyramid_images, 1):
                for i, (image, name, color, exp_time) in enumerate(zip(level_images, channel_names, colors, exposure_times)):
                    is_rgb = len(image.shape) == 3 and image.shape[2] == 3

                    xml_description = self._generate_xml_description(
                        description_version=6,
                        acquisition_software="qptifffile-python",
                        identifier=file_id,
                        slide_id=slide_id,
                        barcode=barcode,
                        computer_name=computer_name,
                        image_type="ReducedResolution",
                        is_unmixed=False,
                        exposure_time=exp_time,
                        signal_units=64,
                        name=name if not is_rgb else None,
                        color=color if not is_rgb else None,
                        objective=objective,
                        study_name=study_name,
                        operator_name=operator_name,
                        instrument_type=instrument_type,
                        lamp_type=lamp_type,
                        camera_type=camera_type,
                        camera_name=camera_name,
                        camera_settings=camera_settings,
                        excitation_filter=excitation_filter,
                        emission_filter=emission_filter,
                        **metadata
                    )

                    self._write_image_to_tiff(tiff, image, xml_description, timestamp,
                                            True, tile_size, compression, is_rgb,
                                            subfile_type=1)

    def _generate_default_colors(self, num_channels: int) -> List[Tuple[int, int, int]]:
        """Generate default colors for channels."""
        default_colors = [
            (0, 0, 255),    # Blue
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
        ]

        colors = []
        for i in range(num_channels):
            if i < len(default_colors):
                colors.append(default_colors[i])
            else:
                # Generate pseudo-random colors for additional channels
                np.random.seed(i)
                colors.append(tuple(np.random.randint(0, 256, 3)))

        return colors

    def _generate_thumbnail(self, images: List[np.ndarray], is_fluorescence: bool) -> np.ndarray:
        """Generate RGB thumbnail image (~500x500)."""
        first_image = images[0]
        height, width = first_image.shape[:2]

        # Calculate thumbnail size maintaining aspect ratio
        max_dim = 500
        if width > height:
            new_width = max_dim
            new_height = int(height * max_dim / width)
        else:
            new_height = max_dim
            new_width = int(width * max_dim / height)

        if len(images) == 1 and len(first_image.shape) == 3:
            from skimage.transform import resize
            return (resize(first_image, (new_height, new_width), preserve_range=True, anti_aliasing=True)).astype(first_image.dtype)
        else:
            # Create RGB composite from multiple channels
            from skimage.transform import resize

            # Resize first few channels and combine
            channels_to_use = min(3, len(images))
            rgb_channels = []

            for i in range(channels_to_use):
                resized = resize(images[i], (new_height, new_width), preserve_range=True, anti_aliasing=True)
                # Normalize to 0-255 range
                img_min, img_max = resized.min(), resized.max()
                if img_max > img_min:
                    normalized = ((resized - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                else:
                    normalized = np.zeros_like(resized, dtype=np.uint8)
                rgb_channels.append(normalized)

            # Pad with zeros if needed
            while len(rgb_channels) < 3:
                rgb_channels.append(np.zeros_like(rgb_channels[0]))

            return np.stack(rgb_channels[:3], axis=2)

    def _generate_pyramid(self, images: List[np.ndarray]) -> List[List[np.ndarray]]:
        """Generate pyramid levels until images are ≤2K x 2K."""
        from skimage.transform import resize

        pyramid_levels = []
        current_images = images

        while True:
            height, width = current_images[0].shape[:2]
            if width <= 2048 and height <= 2048:
                break

            # Generate next level at half resolution
            next_level = []
            new_height = height // 2
            new_width = width // 2

            for image in current_images:
                resized = resize(image, (new_height, new_width), preserve_range=True, anti_aliasing=True)
                next_level.append(resized.astype(image.dtype))

            pyramid_levels.append(next_level)
            current_images = next_level

        return pyramid_levels

    def _generate_xml_description(self, **kwargs) -> str:
        """Generate XML description following QPTIFF specification."""
        # Create root element
        root = ET.Element("PerkinElmer-QPI-ImageDescription")

        # Core required fields in proper order
        core_fields = [
            ("DescriptionVersion", kwargs.get("description_version", 6)),
            ("AcquisitionSoftware", kwargs.get("acquisition_software", "qptifffile-python")),
            ("ImageType", kwargs.get("image_type", "FullResolution")),
            ("Identifier", kwargs.get("identifier")),
            ("SlideID", kwargs.get("slide_id")),
            ("Barcode", kwargs.get("barcode", "")),
            ("ComputerName", kwargs.get("computer_name", "")),
            ("IsUnmixedComponent", "True" if kwargs.get("is_unmixed", False) else "False"),
            ("ExposureTime", kwargs.get("exposure_time", 1000)),
            ("SignalUnits", kwargs.get("signal_units", 64)),
        ]

        # Add core fields
        for tag, value in core_fields:
            if value is not None:
                elem = ET.SubElement(root, tag)
                elem.text = str(value) if value != "" else ""

        # Add ExposureTimeArray
        exposure_time = kwargs.get("exposure_time", 1000)
        exp_array = ET.SubElement(root, "ExposureTimeArray")
        value_elem = ET.SubElement(exp_array, "Value")
        value_elem.text = str(exposure_time)

        # Add Name and Color for channels (not for RGB/thumbnail images)
        name = kwargs.get("name")
        if name:
            name_elem = ET.SubElement(root, "Name")
            name_elem.text = name

        color = kwargs.get("color")
        if color and isinstance(color, (list, tuple)):
            color_elem = ET.SubElement(root, "Color")
            color_elem.text = f"{color[0]},{color[1]},{color[2]}"

        # Add Responsivity section if this is a fluorescence channel
        if name and kwargs.get("image_type") == "FullResolution":
            responsivity = ET.SubElement(root, "Responsivity")
            filter_elem = ET.SubElement(responsivity, "Filter")

            filter_name = ET.SubElement(filter_elem, "Name")
            filter_name.text = name

            response = ET.SubElement(filter_elem, "Response")
            response.text = kwargs.get("responsivity_response", "100.0")

            date_elem = ET.SubElement(filter_elem, "Date")
            date_elem.text = kwargs.get("responsivity_date", datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"))

            filter_id = ET.SubElement(filter_elem, "FilterID")
            filter_id.text = kwargs.get("filter_id", f"{name}_Filter")

        # Add Objective
        objective = kwargs.get("objective")
        if objective:
            obj_elem = ET.SubElement(root, "Objective")
            obj_elem.text = objective

        # Add Biomarker (same as Name for most cases)
        if name:
            biomarker_elem = ET.SubElement(root, "Biomarker")
            biomarker_elem.text = kwargs.get("biomarker", name)

        # Add advanced optional fields
        advanced_fields = [
            ("AutofluorescenceSubtracted", kwargs.get("autofluorescence_subtracted", "True")),
            ("ScaleFactor", kwargs.get("scale_factor", "1.0")),
            ("StudyName", kwargs.get("study_name")),
            ("OperatorName", kwargs.get("operator_name")),
        ]

        for tag, value in advanced_fields:
            if value is not None:
                elem = ET.SubElement(root, tag)
                elem.text = str(value)

        # Add ExcitationFilter if provided
        excitation_filter = kwargs.get("excitation_filter")
        if excitation_filter and isinstance(excitation_filter, dict):
            exc_filter = ET.SubElement(root, "ExcitationFilter")

            if "name" in excitation_filter:
                name_elem = ET.SubElement(exc_filter, "Name")
                name_elem.text = excitation_filter["name"]

            if "manufacturer" in excitation_filter:
                mfg_elem = ET.SubElement(exc_filter, "Manufacturer")
                mfg_elem.text = excitation_filter["manufacturer"]

            if "part_no" in excitation_filter:
                part_elem = ET.SubElement(exc_filter, "PartNo")
                part_elem.text = excitation_filter["part_no"]

            if "bands" in excitation_filter:
                bands_elem = ET.SubElement(exc_filter, "Bands")
                for band_info in excitation_filter["bands"]:
                    band_elem = ET.SubElement(bands_elem, "Band")

                    if "cuton" in band_info:
                        cuton_elem = ET.SubElement(band_elem, "Cuton")
                        cuton_elem.text = str(band_info["cuton"])

                    if "cutoff" in band_info:
                        cutoff_elem = ET.SubElement(band_elem, "Cutoff")
                        cutoff_elem.text = str(band_info["cutoff"])

                    if "active" in band_info:
                        active_elem = ET.SubElement(band_elem, "Active")
                        active_elem.text = "true" if band_info["active"] else "false"

                    if "name" in band_info:
                        name_elem = ET.SubElement(band_elem, "Name")
                        name_elem.text = band_info["name"]

        # Add EmissionFilter if provided
        emission_filter = kwargs.get("emission_filter")
        if emission_filter and isinstance(emission_filter, dict):
            em_filter = ET.SubElement(root, "EmissionFilter")

            if "name" in emission_filter:
                name_elem = ET.SubElement(em_filter, "Name")
                name_elem.text = emission_filter["name"]

            if "manufacturer" in emission_filter:
                mfg_elem = ET.SubElement(em_filter, "Manufacturer")
                mfg_elem.text = emission_filter["manufacturer"]

            if "part_no" in emission_filter:
                part_elem = ET.SubElement(em_filter, "PartNo")
                part_elem.text = emission_filter["part_no"]

            if "bands" in emission_filter:
                bands_elem = ET.SubElement(em_filter, "Bands")
                for band_info in emission_filter["bands"]:
                    band_elem = ET.SubElement(bands_elem, "Band")

                    if "cuton" in band_info:
                        cuton_elem = ET.SubElement(band_elem, "Cuton")
                        cuton_elem.text = str(band_info["cuton"])

                    if "cutoff" in band_info:
                        cutoff_elem = ET.SubElement(band_elem, "Cutoff")
                        cutoff_elem.text = str(band_info["cutoff"])

        # Add CameraSettings if provided
        camera_settings = kwargs.get("camera_settings")
        if camera_settings and isinstance(camera_settings, dict):
            cam_settings = ET.SubElement(root, "CameraSettings")

            camera_fields = ["gain", "offset_counts", "binning", "bit_depth", "orientation"]
            for field in camera_fields:
                if field in camera_settings:
                    field_name = "".join(word.capitalize() for word in field.split("_"))
                    if field == "offset_counts":
                        field_name = "OffsetCounts"
                    elif field == "bit_depth":
                        field_name = "BitDepth"

                    elem = ET.SubElement(cam_settings, field_name)
                    elem.text = str(camera_settings[field])

            # Add ROI if provided
            if "roi" in camera_settings and isinstance(camera_settings["roi"], dict):
                roi_elem = ET.SubElement(cam_settings, "ROI")
                roi_info = camera_settings["roi"]

                for roi_field in ["x", "y", "width", "height"]:
                    if roi_field in roi_info:
                        field_elem = ET.SubElement(roi_elem, roi_field.upper())
                        field_elem.text = str(roi_info[roi_field])

        # Add instrument information
        instrument_fields = [
            ("InstrumentType", kwargs.get("instrument_type")),
            ("LampType", kwargs.get("lamp_type")),
            ("CameraType", kwargs.get("camera_type")),
            ("CameraName", kwargs.get("camera_name")),
        ]

        for tag, value in instrument_fields:
            if value is not None:
                elem = ET.SubElement(root, tag)
                elem.text = str(value)

        # Add ScanProfile if provided
        scan_profile = kwargs.get("scan_profile")
        if scan_profile:
            profile_elem = ET.SubElement(root, "ScanProfile")
            profile_elem.text = scan_profile

        # Generate validation code (MD5 hash of content)
        temp_xml = ET.tostring(root, encoding='unicode')
        validation_code = hashlib.md5(temp_xml.encode('utf-8')).hexdigest().upper()

        # Add validation code
        validation_elem = ET.SubElement(root, "ValidationCode")
        validation_elem.text = validation_code

        # Format XML with proper declaration (UTF-16 to match example)
        xml_str = '<?xml version="1.0" encoding="utf-16"?>\r\n'
        xml_str += ET.tostring(root, encoding='unicode').replace('\n', '\r\n')

        return xml_str

    def _extract_existing_xml_descriptions(self, images: Union[np.ndarray, List[np.ndarray]] = None) -> List[str]:
        """
        Extract existing XML descriptions from the current file and adjust dimensions if needed.
        
        Parameters:
        -----------
        images : np.ndarray or List[np.ndarray], optional
            New image data to check dimensions against
        
        Returns:
        --------
        List[str]
            List of XML descriptions from each page/channel with adjusted dimensions
        """
        xml_descriptions = []
        
        # Convert images to list format for dimension checking
        if images is not None:
            if isinstance(images, np.ndarray):
                if images.ndim == 2:
                    image_list = [images]
                elif images.ndim == 3:
                    # Multi-channel: split along last axis (channels x height x width)
                    image_list = [images[:, :, i] for i in range(images.shape[2])]
                else:
                    raise ValueError(f"Images must be 2D or 3D arrays, got {images.ndim}D")
            else:
                image_list = list(images)
        else:
            image_list = None
        
        # Process each page in the first series (full resolution images only)
        for page_idx, page in enumerate(self.series[0].pages):
            if hasattr(page, 'description') and page.description:
                xml_desc = page.description
                
                # If images provided, check and adjust dimensions in XML
                if image_list is not None and page_idx < len(image_list):
                    xml_desc = self._adjust_xml_dimensions(xml_desc, image_list[page_idx], page_idx)
                
                xml_descriptions.append(xml_desc)
            else:
                # If no description, create a minimal one
                xml_descriptions.append(f'<?xml version="1.0" encoding="utf-16"?>\r\n<PerkinElmer-QPI-ImageDescription><Name>Channel_{page_idx + 1}</Name></PerkinElmer-QPI-ImageDescription>')
        
        return xml_descriptions
    
    def _adjust_xml_dimensions(self, xml_description: str, image: np.ndarray, series_index: int) -> str:
        """
        Adjust dimensions in XML description to match the provided image.
        
        Parameters:
        -----------
        xml_description : str
            Original XML description
        image : np.ndarray
            Image data to match dimensions to
        series_index : int
            Index of the series/channel
            
        Returns:
        --------
        str
            XML description with adjusted dimensions
        """
        try:
            # Parse the XML
            root = ET.fromstring(xml_description)
            
            # Find or create the series element structure
            # Look for existing ScanProfile or similar structure
            scan_profile = root.find('.//ScanProfile')
            
            # Get image dimensions (height, width for 2D image)
            if image.ndim == 2:
                height, width = image.shape
                channels = 1
            elif image.ndim == 3:
                height, width, channels = image.shape
            else:
                raise ValueError(f"Unsupported image dimensions: {image.ndim}")
            
            # If no ScanProfile exists, create one with the dimension information
            if scan_profile is None:
                scan_profile = ET.SubElement(root, "ScanProfile")
                scan_profile.text = f"<ScanProfile><Series><Index>{series_index}</Index><Shape>{channels},{height},{width}</Shape></Series></ScanProfile>"
            else:
                # Try to parse existing scan profile and update dimensions
                try:
                    # Extract existing scan profile content
                    profile_content = scan_profile.text if scan_profile.text else ""
                    
                    # Parse the inner XML if it exists
                    if profile_content.strip():
                        try:
                            profile_root = ET.fromstring(profile_content)
                            # Find series element
                            series_elem = profile_root.find('.//Series')
                            if series_elem is not None:
                                # Update shape
                                shape_elem = series_elem.find('Shape')
                                if shape_elem is not None:
                                    shape_elem.text = f"{channels},{height},{width}"
                                else:
                                    shape_elem = ET.SubElement(series_elem, "Shape")
                                    shape_elem.text = f"{channels},{height},{width}"
                                
                                # Update index if needed
                                index_elem = series_elem.find('Index')
                                if index_elem is None:
                                    index_elem = ET.SubElement(series_elem, "Index")
                                    index_elem.text = str(series_index)
                            else:
                                # Create series element
                                series_elem = ET.SubElement(profile_root, "Series")
                                index_elem = ET.SubElement(series_elem, "Index")
                                index_elem.text = str(series_index)
                                shape_elem = ET.SubElement(series_elem, "Shape")
                                shape_elem.text = f"{channels},{height},{width}"
                            
                            # Update the scan profile text
                            scan_profile.text = ET.tostring(profile_root, encoding='unicode')
                        except ET.ParseError:
                            # If parsing fails, create new content
                            scan_profile.text = f"<ScanProfile><Series><Index>{series_index}</Index><Shape>{channels},{height},{width}</Shape></Series></ScanProfile>"
                    else:
                        # No existing content, create new
                        scan_profile.text = f"<ScanProfile><Series><Index>{series_index}</Index><Shape>{channels},{height},{width}</Shape></Series></ScanProfile>"
                        
                except Exception:
                    # Fallback: replace with new content
                    scan_profile.text = f"<ScanProfile><Series><Index>{series_index}</Index><Shape>{channels},{height},{width}</Shape></Series></ScanProfile>"
            
            # Return the modified XML
            xml_str = '<?xml version="1.0" encoding="utf-16"?>\r\n'
            xml_str += ET.tostring(root, encoding='unicode').replace('\n', '\r\n')
            return xml_str
            
        except ET.ParseError:
            # If XML parsing fails, return original
            return xml_description
        except Exception:
            # For any other errors, return original XML
            return xml_description

    def _write_image_to_tiff(self, tiff_writer: TiffWriter, image: np.ndarray,
                           description: str, timestamp: str, use_tiles: bool,
                           tile_size: int, compression: str, is_rgb: bool,
                           subfile_type: int = 0) -> None:
        """Write single image to TIFF with proper metadata."""

        # Determine image properties
        if is_rgb:
            photometric = 'rgb'
            planarconfig = 'contig'
            samples_per_pixel = 3
        else:
            photometric = 'minisblack'
            planarconfig = 'contig'
            samples_per_pixel = 1

        # Set resolution (default to 96 DPI if unknown)
        resolution = (96, 96)
        resolution_unit = 'inch'

        # Write the image
        tiff_writer.write(
            image,
            photometric=photometric,
            planarconfig=planarconfig,
            compression=compression,
            tile=(tile_size, tile_size) if use_tiles else None,
            subfiletype=subfile_type,
            software="PerkinElmer-QPI-qptifffile-python",
            description=description,
            datetime=timestamp,
            resolution=resolution,
            resolutionunit=resolution_unit
        )

