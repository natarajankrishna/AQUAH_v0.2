import os
import requests
import gzip
import shutil
from datetime import datetime, timedelta
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import rasterio
import numpy as np
import glob
import geopandas as gpd
from osgeo import gdal
from rasterio.windows import from_bounds




def download_mrms_precipitation(start_date, end_date, download_folder='../MRMS_precipitation', time_step='1d'):
    """
    Downloads and extracts MRMS hourly precipitation data for a given date range.
    
    Parameters:
    -----------
    start_date : datetime
        The start date and time for data download
    end_date : datetime
        The end date and time for data download
    download_folder : str, optional
        Directory to save the downloaded and extracted files (default: '../MRMS_precipitation')
    """
    # hourly data
    if time_step == '1h':
        # Create directories if they do not exist
        if not os.path.exists(download_folder):
            os.makedirs(download_folder)
        else:
            # Clear all files in the destination folder before downloading
            for file_path in glob.glob(os.path.join(download_folder, '*')):
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            print(f"Cleared all existing files in {download_folder}")

        # Base URL for data
        base_url = "https://mtarchive.geol.iastate.edu/"

        # Calculate total number of hours for progress bar
        total_hours = int((end_date - start_date).total_seconds() / 3600) + 1
        
        print(f"Downloading MRMS precipitation data from {start_date} to {end_date}")
        print(f"Files will be saved to: {os.path.abspath(download_folder)}")
        
        # Date threshold for file format change (October 15, 2020)
        format_change_date = datetime(2020, 10, 15)
        
        # Loop through each hourly timestamp in the date range
        current_time = start_date
        failed_downloads = 0
        
        with tqdm(total=total_hours, desc="Downloading MRMS data") as pbar:
            while current_time <= end_date:
                # Format year, month, day, and hour as strings
                year_str = current_time.strftime('%Y')
                month_str = current_time.strftime('%m')
                day_str = current_time.strftime('%d')
                hour_str = current_time.strftime('%H')
                
                # Determine file format based on date
                if current_time < format_change_date:
                    # Before October 15, 2020: GaugeCorr format
                    product_dir = "GaugeCorr_QPE_01H"
                    file_prefix = "GaugeCorr_QPE_01H"
                else:
                    # October 15, 2020 and after: MultiSensor format
                    product_dir = "MultiSensor_QPE_01H_Pass2"
                    file_prefix = "MultiSensor_QPE_01H_Pass2"
                
                # Construct the file name
                file_name = f"{file_prefix}_00.00_{year_str}{month_str}{day_str}-{hour_str}0000.grib2.gz"
                
                # Construct the full download URL
                file_url = f"{base_url}{year_str}/{month_str}/{day_str}/mrms/ncep/{product_dir}/{file_name}"
                            
                # Define the full local file path
                output_file = os.path.join(download_folder, file_name)
                
                # Download the file
                try:
                    response = requests.get(file_url)
                    response.raise_for_status()  # Raise an exception for HTTP errors
                    
                    # Save the compressed file
                    with open(output_file, 'wb') as f:
                        f.write(response.content)
                    
                    # Extract the file
                    extracted_file = output_file[:-3]  # Remove .gz extension
                    with gzip.open(output_file, 'rb') as f_in:
                        with open(extracted_file, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    # Delete the original .gz file
                    os.remove(output_file)
                except Exception as e:
                    failed_downloads += 1
                    tqdm.write(f"Failed to download or extract: {file_url} - Error: {str(e)[:100]}...")

                # Move to the next hour
                current_time += timedelta(hours=1)
                pbar.update(1)
        
        print(f"MRMS download complete. Files saved in: {os.path.abspath(download_folder)}")
        if failed_downloads > 0:
            print(f"Note: {failed_downloads} files failed to download")

    elif time_step == '1d':
        # daily data
        # Create directories if they do not exist
        if not os.path.exists(download_folder):
            os.makedirs(download_folder)

        # Base URL for data
        base_url = "https://mtarchive.geol.iastate.edu/"

        # Extend end_date by one day to download additional data
        extended_end_date = end_date + timedelta(days=1)

        # Calculate total number of days for progress bar
        total_days = (extended_end_date - start_date).days + 1
        
        print(f"Downloading MRMS daily precipitation data from {start_date} to {extended_end_date}")
        print(f"Files will be saved to: {os.path.abspath(download_folder)}")
        
        # Date threshold for file format change (October 15, 2020)
        format_change_date = datetime(2020, 10, 15)
        
        # Loop through each daily timestamp in the date range
        current_date = start_date
        failed_downloads = 0
        skipped_files = 0
        
        with tqdm(total=total_days, desc="Downloading MRMS daily data") as pbar:
            while current_date <= extended_end_date:
                # Format year, month, and day as strings
                year_str = current_date.strftime('%Y')
                month_str = current_date.strftime('%m')
                day_str = current_date.strftime('%d')
                
                # Determine file format based on date
                if current_date < format_change_date:
                    # Before October 15, 2020: GaugeCorr format
                    product_dir = "GaugeCorr_QPE_24H"
                    file_prefix = "GaugeCorr_QPE_24H"
                else:
                    # October 15, 2020 and after: MultiSensor format
                    product_dir = "MultiSensor_QPE_24H_Pass2"
                    file_prefix = "MultiSensor_QPE_24H_Pass2"
                
                # Construct the file name
                file_name = f"{file_prefix}_00.00_{year_str}{month_str}{day_str}-000000.grib2.gz"
                
                # Construct the full download URL
                file_url = f"{base_url}{year_str}/{month_str}/{day_str}/mrms/ncep/{product_dir}/{file_name}"
                            
                # Define the full local file path
                output_file = os.path.join(download_folder, file_name)
                extracted_file = output_file[:-3]  # Remove .gz extension
                
                # Check if the extracted file already exists
                if os.path.exists(extracted_file):
                    skipped_files += 1
                    # tqdm.write(f"Skipping existing file: {os.path.basename(extracted_file)}")
                    current_date += timedelta(days=1)
                    pbar.update(1)
                    continue
                
                # Download the file
                try:
                    response = requests.get(file_url)
                    response.raise_for_status()  # Raise an exception for HTTP errors
                    
                    # Save the compressed file
                    with open(output_file, 'wb') as f:
                        f.write(response.content)
                    
                    # Extract the file
                    with gzip.open(output_file, 'rb') as f_in:
                        with open(extracted_file, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    # Delete the original .gz file
                    os.remove(output_file)
                except Exception as e:
                    failed_downloads += 1
                    tqdm.write(f"Failed to download or extract: {file_url} - Error: {str(e)[:100]}...")

                # Move to the next day
                current_date += timedelta(days=1)
                pbar.update(1)
        
        print(f"MRMS daily download complete. Files saved in: {os.path.abspath(download_folder)}")
        if skipped_files > 0:
            print(f"Note: {skipped_files} files were skipped (already exist)")
        if failed_downloads > 0:
            print(f"Note: {failed_downloads} files failed to download")
            
def _process_single_file(grib_file, output_folder, basin_clipping, expanded_bounds, start_date=None, end_date=None):
    """
    Helper function to process a single GRIB2 file.
    It converts the file to GeoTIFF format and clips it if basin clipping is enabled.
    
    Args:
        grib_file (str): Path to the GRIB2 file.
        output_folder (str): Folder to save the output GeoTIFF file.
        basin_clipping (bool): Whether to perform clipping to basin bounds.
        expanded_bounds (tuple or None): Expanded bounds for clipping (minx, miny, maxx, maxy).
        
    Returns:
        tuple: (base filename, error message or None if successful)
    """
    base_name = os.path.basename(grib_file)
    output_name = os.path.splitext(base_name)[0] + '.tif'
    output_path = os.path.join(output_folder, output_name)
    try:
        # Open the grib2 file using GDAL
        src_ds = gdal.Open(grib_file)
        if src_ds is None:
            return (base_name, "Could not open file")
        
        if basin_clipping:
            # Process with basin clipping using rasterio
            with rasterio.open(grib_file) as src:
                # Read the data and apply nodata filter
                data = src.read(1)
                data = np.where((data > 1000) | (data < 0), -9999, data)
                data_float32 = data.astype(np.float32)
                
                # Determine window corresponding to the expanded basin bounds
                window = from_bounds(expanded_bounds[0], expanded_bounds[1],
                                     expanded_bounds[2], expanded_bounds[3],
                                     src.transform)
                # Read only the data within the window
                clipped_data = src.read(1, window=window)
                clipped_data = np.where((clipped_data > 1000) | (clipped_data < 0), -9999, clipped_data)
                clipped_data = clipped_data.astype(np.float32)
                
                # Get the transform for the clipped window
                clipped_transform = rasterio.windows.transform(window, src.transform)
                
                # Prepare metadata for output GeoTIFF
                new_meta = {
                    'driver': 'GTiff',
                    'height': clipped_data.shape[0],
                    'width': clipped_data.shape[1],
                    'count': 1,
                    'dtype': 'float32',
                    'crs': src.crs,
                    'transform': clipped_transform,
                    'nodata': -9999,
                    'compress': 'none'
                }
                with rasterio.open(output_path, 'w', **new_meta) as dst:
                    dst.write(clipped_data, 1)
        else:
            # Process without basin clipping
            driver = gdal.GetDriverByName('GTiff')
            dst_ds = driver.CreateCopy(output_path, src_ds, 0)
            band = dst_ds.GetRasterBand(1)
            data = band.ReadAsArray()
            data = data.astype(np.float32)
            data = np.where((data > 1000) | (data < 0), -9999, data)
            band.SetNoDataValue(-9999)
            band.WriteArray(data)
            dst_ds = None
        
        src_ds = None
        return (base_name, None)
    except Exception as e:
        return (base_name, str(e)[:100])

def _process_single_file_daily(grib_file, output_folder, basin_clipping, expanded_bounds, start_date=None, end_date=None):
    """Process a single GRIB2 file for daily data with date adjustment"""
    base_name = os.path.basename(grib_file)
    try:
        # Extract date from filename (assuming format contains YYYYMMDD)
        import re
        date_match = re.search(r'(\d{8})', base_name)
        if date_match:
            original_date_str = date_match.group(1)
            # Convert to datetime and subtract 1 day
            original_date = datetime.strptime(original_date_str, '%Y%m%d')
            adjusted_date = original_date - timedelta(days=1)
            
            # Check if the adjusted date is within the specified time range
            if start_date and end_date:
                end_date_plus_one = end_date + timedelta(days=1)
                if not (start_date <= original_date <= end_date_plus_one):
                    # Skip this file as it's outside the time range
                    return base_name, "skipped_outside_time_range"
            
            adjusted_date_str = adjusted_date.strftime('%Y%m%d')
            
            # Create new filename with adjusted date in format precipitation_MRMS_YYYYMMDD00.tif
            output_name = f"precipitation_MRMS_{adjusted_date_str}00.tif"
        else:
            # Fallback if no date found in filename
            output_name = os.path.splitext(base_name)[0] + '.tif'
        
        output_path = os.path.join(output_folder, output_name)
        
        # Open and process the GRIB2 file
        with rasterio.open(grib_file) as src:
            data = src.read(1)
            
            # Apply basin clipping if enabled
            if basin_clipping and expanded_bounds:
                window = from_bounds(*expanded_bounds, src.transform)
                data = src.read(1, window=window)
                transform = src.window_transform(window)
            else:
                transform = src.transform
            
            # Set invalid values to -9999
            data = np.where((data > 1000) | (data < 0), -9999, data)
            data_float32 = data.astype('float32')
            
            # Create output metadata
            kwargs = src.meta.copy()
            kwargs.update({
                'driver': 'GTiff',
                'dtype': 'float32',
                'nodata': -9999,
                'compress': 'none'
            })
            
            if basin_clipping and expanded_bounds:
                kwargs.update({
                    'height': data.shape[0],
                    'width': data.shape[1],
                    'transform': transform
                })
            
            # Write the output file
            with rasterio.open(output_path, 'w', **kwargs) as dst:
                dst.write(data_float32, 1)
        
        return base_name, None
        
    except Exception as e:
        return base_name, str(e)[:100]

def process_mrms_grib2_to_tif(input_folder='../MRMS_precipitation', 
                              output_folder='../CREST_input/MRMS/', 
                              basin_shp_path='shpFile/Basin_selected_5.shp',
                              time_step='1d',
                              start_date=None,
                              end_date=None,
                              num_processes=1):
    """
    Process MRMS grib2 files to GeoTIFF format and clip to basin boundary.
    The function now supports parallel processing using multiple processes.
    
    Args:
        input_folder (str): Path to the folder containing MRMS grib2 files.
        output_folder (str): Path to save the output GeoTIFF files.
        basin_shp_path (str): Path to the basin shapefile for clipping.
        num_processes (int): Number of processes to use (use 1 for sequential processing).
    """
    if time_step == '1h':
        # Create or clear the output directory
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        else:
            for file_path in glob.glob(os.path.join(output_folder, '*')):
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print(f"Cleared all existing files in {output_folder}")
        
        # Load the basin shapefile and compute expanded bounds for clipping
        try:
            basin_gdf = gpd.read_file(basin_shp_path)
            basin_bounds = basin_gdf.total_bounds  # (minx, miny, maxx, maxy)
            
            # Expand the bounds by 100% of the width/height (adjust factor as needed)
            width = basin_bounds[2] - basin_bounds[0]
            height = basin_bounds[3] - basin_bounds[1]
            buffer_x = width * 1
            buffer_y = height * 1
            
            expanded_bounds = (
                basin_bounds[0] - buffer_x,  # minx
                basin_bounds[1] - buffer_y,  # miny
                basin_bounds[2] + buffer_x,  # maxx
                basin_bounds[3] + buffer_y   # maxy
            )
            
            print(f"Loaded basin shapefile: {basin_shp_path}")
            print(f"Original bounds: ({basin_bounds[0]:.3f}, {basin_bounds[1]:.3f}, "
                f"{basin_bounds[2]:.3f}, {basin_bounds[3]:.3f})")
            print(f"Expanded bounds: ({expanded_bounds[0]:.3f}, {expanded_bounds[1]:.3f}, "
                f"{expanded_bounds[2]:.3f}, {expanded_bounds[3]:.3f})")
            basin_clipping = True
        except Exception as e:
            print(f"Error loading basin shapefile: {str(e)}")
            print("Processing will continue without clipping to basin boundary")
            basin_gdf = None
            expanded_bounds = None
            basin_clipping = False
        
        # Retrieve list of GRIB2 files from the input folder
        grib2_files = glob.glob(os.path.join(input_folder, '*.grib2'))
        if not grib2_files:
            print(f"No grib2 files found in {input_folder}")
            return
        
        print(f"Processing {len(grib2_files)} MRMS grib2 files to GeoTIFF format")
        print(f"Input folder: {os.path.abspath(input_folder)}")
        print(f"Output folder: {os.path.abspath(output_folder)}")
        
        failed_files = 0

        if num_processes == 1:
            # Sequential processing if num_processes is 1
            for grib_file in tqdm(grib2_files, desc="Converting MRMS to GeoTIFF"):
                base_name, error = _process_single_file(grib_file, output_folder, basin_clipping, expanded_bounds, start_date, end_date)
                if error:
                    tqdm.write(f"Error processing {base_name}: {error}")
                    failed_files += 1
        else:
            # Parallel processing using multiple processes
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                futures = {executor.submit(_process_single_file, grib_file, output_folder,
                                            basin_clipping, expanded_bounds, start_date, end_date): grib_file 
                        for grib_file in grib2_files}
                for future in tqdm(as_completed(futures), total=len(futures), desc="Converting MRMS to GeoTIFF"):
                    base_name, error = future.result()
                    if error:
                        tqdm.write(f"Error processing {base_name}: {error}")
                        failed_files += 1
        
        # Post-process the generated TIFF files to ensure proper format (sequentially)
        tif_files = glob.glob(os.path.join(output_folder, '*.tif'))
        print(f"Ensuring proper format for {len(tif_files)} output files")
        
        for file_path in tqdm(tif_files, desc="Checking output formats"):
            try:
                with rasterio.open(file_path) as src:
                    data = src.read(1)
                    data = np.where((data > 1000) | (data < 0), -9999, data)
                    data_float32 = data.astype('float32')
                    meta = src.meta.copy()
                    meta.update({'dtype': 'float32', 'nodata': -9999, 'compress': 'none'})
                with rasterio.open(file_path, 'w', **meta) as dst:
                    dst.write(data_float32, 1)
            except Exception as e:
                print(f"Error formatting {os.path.basename(file_path)}: {str(e)[:100]}...")
        
        print(f"MRMS conversion completed. Output files saved to {os.path.abspath(output_folder)}")
        if failed_files > 0:
            print(f"Note: {failed_files} files failed to process")
    elif time_step == '1d':
        # Daily processing - need to adjust dates by subtracting 1 day
        # Create or clear the output directory
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        else:
            for file_path in glob.glob(os.path.join(output_folder, '*')):
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print(f"Cleared all existing files in {output_folder}")
        
        # Load the basin shapefile and compute expanded bounds for clipping
        try:
            basin_gdf = gpd.read_file(basin_shp_path)
            basin_bounds = basin_gdf.total_bounds  # (minx, miny, maxx, maxy)
            
            # Expand the bounds by 100% of the width/height (adjust factor as needed)
            width = basin_bounds[2] - basin_bounds[0]
            height = basin_bounds[3] - basin_bounds[1]
            buffer_x = width * 1
            buffer_y = height * 1
            
            expanded_bounds = (
                basin_bounds[0] - buffer_x,  # minx
                basin_bounds[1] - buffer_y,  # miny
                basin_bounds[2] + buffer_x,  # maxx
                basin_bounds[3] + buffer_y   # maxy
            )
            
            print(f"Loaded basin shapefile: {basin_shp_path}")
            print(f"Original bounds: ({basin_bounds[0]:.3f}, {basin_bounds[1]:.3f}, "
                f"{basin_bounds[2]:.3f}, {basin_bounds[3]:.3f})")
            print(f"Expanded bounds: ({expanded_bounds[0]:.3f}, {expanded_bounds[1]:.3f}, "
                f"{expanded_bounds[2]:.3f}, {expanded_bounds[3]:.3f})")
            basin_clipping = True
        except Exception as e:
            print(f"Error loading basin shapefile: {str(e)}")
            print("Processing will continue without clipping to basin boundary")
            basin_gdf = None
            expanded_bounds = None
            basin_clipping = False
        
        # Retrieve list of GRIB2 files from the input folder
        grib2_files = glob.glob(os.path.join(input_folder, '*.grib2'))
        if not grib2_files:
            print(f"No grib2 files found in {input_folder}")
            return
        
        # Filter files based on date range if provided
        if start_date is not None and end_date is not None:
            filtered_files = []
            for grib_file in grib2_files:
                filename = os.path.basename(grib_file)
                try:
                    # Extract date from filename (format: *_YYYYMMDD-HHMMSS.grib2)
                    date_part = filename.split('_')[-1].split('-')[0]  # Get YYYYMMDD part
                    file_date = datetime.strptime(date_part, '%Y%m%d').date()
                    
                    # Check if file date is within the specified range
                    if start_date.date() <= file_date <= (end_date + timedelta(days=1)).date():
                        filtered_files.append(grib_file)
                except Exception as e:
                    # If date parsing fails, include the file to be safe
                    filtered_files.append(grib_file)
            
            grib2_files = filtered_files
            print(f"Filtered to {len(grib2_files)} files within date range {start_date.date()} to {end_date.date()}")
        
        if not grib2_files:
            print(f"No grib2 files found within the specified date range")
            return
        
        print(f"Processing {len(grib2_files)} MRMS grib2 files to GeoTIFF format (daily with date adjustment)")
        print(f"Input folder: {os.path.abspath(input_folder)}")
        print(f"Output folder: {os.path.abspath(output_folder)}")
        
        failed_files = 0

        if num_processes == 1:
            # Sequential processing if num_processes is 1
            for grib_file in tqdm(grib2_files, desc="Converting MRMS to GeoTIFF (daily)"):
                base_name, error = _process_single_file_daily(grib_file, output_folder, basin_clipping, expanded_bounds, start_date, end_date)
                if error:
                    tqdm.write(f"Error processing {base_name}: {error}")
                    failed_files += 1
        else:
            # Parallel processing using multiple processes
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                futures = {executor.submit(_process_single_file_daily, grib_file, output_folder,
                                            basin_clipping, expanded_bounds, start_date, end_date): grib_file 
                        for grib_file in grib2_files}
                for future in tqdm(as_completed(futures), total=len(futures), desc="Converting MRMS to GeoTIFF (daily)"):
                    base_name, error = future.result()
                    if error:
                        tqdm.write(f"Error processing {base_name}: {error}")
                        failed_files += 1
        
        print(f"MRMS daily conversion completed. Output files saved to {os.path.abspath(output_folder)}")
        if failed_files > 0:
            print(f"Note: {failed_files} files failed to process")

def precipitation_processor(args):
    download_mrms_precipitation(args.warmup_time_start, args.time_end, args.mrms_data_path, args.time_step)
    process_mrms_grib2_to_tif(args.mrms_data_path, args.crest_input_mrms_path, args.basin_shp_path, args.time_step, args.warmup_time_start, args.time_end, args.num_processes)


