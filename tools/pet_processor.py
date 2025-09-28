import os
import requests
import tarfile
from datetime import datetime, timedelta
from tqdm import tqdm
import glob
import rasterio
import numpy as np


def download_pet_data(start_date, end_date, download_folder='../PET_data', num_threads=4):
    """
    Downloads and extracts USGS PET data for a given date range using multithreading, skipping already downloaded files.
    
    Parameters:
    -----------
    start_date : datetime
        The start date for downloading data
    end_date : datetime
        The end date for downloading data
    download_folder : str, optional
        Directory to save the downloaded and extracted files (default: '../PET_data')
    num_threads : int, optional
        Number of threads to use for downloading (default: 4)
    """
    import concurrent.futures
    import threading
    
    # Create directory if it doesn't exist
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)
    
    # USGS PET data base URL
    base_url = "https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/fews/web/global/daily/pet/downloads/daily/"
    
    # Generate list of dates to download
    dates_to_download = []
    current_date = start_date
    while current_date <= end_date:
        dates_to_download.append(current_date)
        current_date += timedelta(days=1)
    
    total_days = len(dates_to_download)
    
    print(f"Downloading PET data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Files will be saved to: {os.path.abspath(download_folder)}")
    print(f"Using {num_threads} threads for downloading")
    
    # Thread-safe counters
    failed_downloads = 0
    lock = threading.Lock()
    
    def download_single_date(date):
        """Download and extract PET data for a single date"""
        nonlocal failed_downloads
        
        # Generate the filename in "yymmdd" format
        file_name = f"et{date.strftime('%y%m%d')}.tar.gz"
        
        # Construct the full download URL
        file_url = f"{base_url}{file_name}"
        
        # Define the local save path
        save_path = os.path.join(download_folder, file_name)
        
        # Check if the file has already been downloaded
        extracted_file_path = os.path.join(download_folder, f"et{date.strftime('%y%m%d')}.bil")
        if os.path.exists(extracted_file_path):
            # tqdm.write(f"File already exists, skipping download: {extracted_file_path}")
            return True
        
        # Download the file
        try:
            response = requests.get(file_url)
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            # Save the file
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            # Extract the .tar.gz file
            with tarfile.open(save_path, 'r:gz') as tar:
                tar.extractall(path=download_folder)
            
            # Delete the original .tar.gz file to save space
            os.remove(save_path)
            
            return True
            
        except Exception as e:
            with lock:
                failed_downloads += 1
            tqdm.write(f"Failed to download: {file_url} - Error: {str(e)[:100]}...")
            return False
    
    # Use ThreadPoolExecutor for concurrent downloads
    with tqdm(total=total_days, desc="Downloading PET data") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit all download tasks
            future_to_date = {executor.submit(download_single_date, date): date for date in dates_to_download}
            
            # Process completed downloads
            for future in concurrent.futures.as_completed(future_to_date):
                date = future_to_date[future]
                try:
                    result = future.result()
                except Exception as exc:
                    with lock:
                        failed_downloads += 1
                    tqdm.write(f"Date {date.strftime('%Y-%m-%d')} generated an exception: {exc}")
                finally:
                    pbar.update(1)
    
    print(f"PET data download complete. Files saved in: {os.path.abspath(download_folder)}")
    if failed_downloads > 0:
        print(f"Note: {failed_downloads} files failed to download")

def process_pet_bil_to_tif(input_folder='../PET_data', output_folder='../CREST_input/PET/', start_date=None, end_date=None):
    """
    Process PET BIL files to GeoTIFF format.
    
    Args:
        input_folder (str): Path to the folder containing PET BIL files
        output_folder (str): Path to save the output GeoTIFF files
        start_date (datetime): Start date for processing (optional)
        end_date (datetime): End date for processing (optional)
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        # Clear all files in the destination folder before processing
        for file_path in glob.glob(os.path.join(output_folder, '*')):
            if os.path.isfile(file_path):
                os.remove(file_path)
        print(f"Cleared all existing files in {output_folder}")
    
    # Find all BIL files in the input folder
    bil_files = glob.glob(os.path.join(input_folder, '*.bil'))
    
    if not bil_files:
        print(f"No BIL files found in {input_folder}")
        return
    
    # Filter files by date range if provided
    if start_date is not None or end_date is not None:
        filtered_files = []
        for bil_file in bil_files:
            base_name = os.path.basename(bil_file)
            # Extract date from filename (assuming format like etYYMMDD.bil where YYMMDD is year+month+day)
            try:
                # Extract the numeric part from filename
                name_part = os.path.splitext(base_name)[0]
                if name_part.startswith('et') and len(name_part) >= 8:
                    year_month_day = name_part[2:]  # Remove 'et' prefix
                    year = int(year_month_day[:2])
                    month = int(year_month_day[2:4])
                    day = int(year_month_day[4:6])
                    
                    # Convert 2-digit year to 4-digit year (assuming 2000s)
                    if year < 50:
                        full_year = 2000 + year
                    else:
                        full_year = 1900 + year
                    
                    # Create date object
                    file_date = datetime(full_year, month, day)
                    
                    # Check if file date is within the specified range
                    if start_date is not None and file_date.date() < start_date.date():
                        continue
                    if end_date is not None and file_date.date() > end_date.date():
                        continue
                    
                    filtered_files.append(bil_file)
                else:
                    # If filename doesn't match expected pattern, include it
                    filtered_files.append(bil_file)
            except (ValueError, IndexError):
                # If date parsing fails, include the file
                filtered_files.append(bil_file)
        
        bil_files = filtered_files
        
        if not bil_files:
            print(f"No BIL files found in the specified date range")
            return
    
    print(f"Processing {len(bil_files)} PET BIL files to GeoTIFF format")
    print(f"Input folder: {os.path.abspath(input_folder)}")
    print(f"Output folder: {os.path.abspath(output_folder)}")
    if start_date is not None or end_date is not None:
        print(f"Date range: {start_date.strftime('%Y-%m-%d') if start_date else 'N/A'} to {end_date.strftime('%Y-%m-%d') if end_date else 'N/A'}")
    
    failed_files = 0
    
    # Process each BIL file
    with tqdm(total=len(bil_files), desc="Converting PET to GeoTIFF") as pbar:
        for bil_file in bil_files:
            # Get the base filename without extension
            base_name = os.path.basename(bil_file)
            
            # Extract date from filename and create new filename with etYYYYMMDD.tif format
            try:
                name_part = os.path.splitext(base_name)[0]
                if name_part.startswith('et') and len(name_part) >= 8:
                    year_month_day = name_part[2:]  # Remove 'et' prefix
                    year = int(year_month_day[:2])
                    month = int(year_month_day[2:4])
                    day = int(year_month_day[4:6])
                    
                    # Convert 2-digit year to 4-digit year (assuming 2000s)
                    if year < 50:
                        full_year = 2000 + year
                    else:
                        full_year = 1900 + year
                    
                    # Create date object
                    file_date = datetime(full_year, month, day)
                    
                    # Create output filename in etYYYYMMDD.tif format with full 4-digit year
                    output_name = f"et{full_year}{file_date.strftime('%m%d')}.tif"
                else:
                    # If filename doesn't match expected pattern, use original name
                    output_name = os.path.splitext(base_name)[0] + '.tif'
            except (ValueError, IndexError):
                # If date parsing fails, use original name
                output_name = os.path.splitext(base_name)[0] + '.tif'
            
            output_path = os.path.join(output_folder, output_name)
            
            try:
                # Open the BIL file with rasterio
                with rasterio.open(bil_file) as src:
                    # Read the data
                    data = src.read(1)
                    
                    # Set NaN values to -9999
                    data = np.where(np.isnan(data), -9999, data)
                    data_float32 = data.astype(np.float32)
                    
                    # Create a new GeoTIFF with the same metadata but float32 dtype
                    kwargs = src.meta.copy()
                    kwargs.update({
                        'driver': 'GTiff',
                        'dtype': 'float32',
                        'nodata': -9999
                    })
                    
                    # Write the output file
                    with rasterio.open(output_path, 'w', **kwargs) as dst:
                        dst.write(data_float32, 1)
                
            except Exception as e:
                failed_files += 1
                tqdm.write(f"Error processing {base_name}: {str(e)[:100]}...")
            
            pbar.update(1)
    
    print(f"PET conversion completed. Output files saved to {os.path.abspath(output_folder)}")
    if failed_files > 0:
        print(f"Note: {failed_files} files failed to process")

def pet_processor(args):
    download_pet_data(args.warmup_time_start, args.time_end, args.pet_data_path, args.num_processes)
    process_pet_bil_to_tif(args.pet_data_path, args.crest_input_pet_path, args.warmup_time_start, args.time_end)


