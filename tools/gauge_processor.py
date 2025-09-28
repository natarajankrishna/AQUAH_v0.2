import os
import pandas as pd

def get_gauge_coordinates(gauge_meta_path, station_id):
    """
    Get the latitude and longitude of a USGS gauge station by its ID.
    
    Parameters:
    -----------
    gauge_meta_path : str
        Path to the CSV file containing gauge metadata
    station_id : str or int
        The station ID (will be converted to integer to remove leading zeros)
    
    Returns:
    --------
    tuple
        (latitude, longitude) of the gauge station, or None if not found
    """
    import pandas as pd
    
    # Convert station_id to integer to handle leading zeros
    try:
        station_id_int = int(station_id)
    except ValueError:
        print(f"Error: Station ID '{station_id}' is not a valid number")
        return None
    
    # Read the gauge metadata
    try:
        gauge_meta = pd.read_csv(gauge_meta_path)
    except Exception as e:
        print(f"Error reading gauge metadata file: {e}")
        return None
    
    # Find the station in the metadata
    station = gauge_meta[gauge_meta['STAID'] == station_id_int]
    
    if len(station) == 0:
        print(f"Station ID {station_id} not found in metadata")
        return None
    
    # Return the coordinates
    lat = station['LAT_GAGE'].values[0]
    lon = station['LNG_GAGE'].values[0]
    
    return (lat, lon)

def download_usgs_data(site_code, start_date=None, end_date=None, output_dir='USGS_gauge/', time_step='1d'):
    """
    Download discharge data from USGS station and save as CSV file
    
    Parameters:
        site_code (str): USGS station ID
        start_date (datetime): Start date and time
        end_date (datetime): End date and time
        output_dir (str): Output directory path
        
    Returns:
        DataFrame or None: The downloaded discharge data, or None if download failed
    """
    if time_step == '1d':
        try:
            import dataretrieval.nwis as nwis
        except ImportError:
            print("Error: dataretrieval package not found. Please install with 'pip install dataretrieval'")
            return None
        
        if start_date is None or end_date is None:
            print("Error: start_date and end_date must be provided as datetime objects")
            return None
        
        print(f"Downloading daily USGS discharge data for station {site_code} from {start_date} to {end_date}")
        print(f"Files will be saved to: {os.path.abspath(output_dir)}")
        
        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Format dates for NWIS service
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # Call NWIS service to download daily discharge data
        df = nwis.get_record(
            sites=site_code,
            service='dv',             # 'dv' -> daily values
            start=start_date_str,
            end=end_date_str,
            parameterCd='00060',
            statCd='00003'
        )
        if not df.empty:
            # Extract discharge data column and convert units (from cfs to m³/s, conversion factor: 0.0283)
            discharge_cols = [col for col in df.columns if '00060' in col and 'cd' not in col]
            if discharge_cols:
                discharge_col = discharge_cols[0]
                
                # Create result DataFrame - use copy instead of reference
                result_df = pd.DataFrame()
                result_df['datetime'] = df.index.copy()
                
                # Use values directly instead of column reference
                discharge_values = df[discharge_col].values * 0.0283  # Convert cfs to m³/s
                result_df['discharge'] = discharge_values
                
                
                # If NaN values still exist, try direct loop assignment
                if result_df['discharge'].isna().any():
                    print("NaN values detected, trying direct loop assignment...")
                    new_discharge = []
                    for val in df[discharge_col].values:
                        new_discharge.append(val * 0.0283 if pd.notna(val) else val)
                    result_df['discharge'] = new_discharge
                
                # Save as CSV file
                output_file = os.path.join(output_dir, f'USGS_{site_code}_UTC_m3s.csv')
                result_df.to_csv(output_file, index=False, float_format='%.6f')  # Specify float format
                print(f'Successfully downloaded daily data for station {site_code} and saved to {output_file}')
                    
                return result_df
            else:
                print(f"Error: No discharge column found for station {site_code}")
        else:
            print(f'No data available for station {site_code}')
    
    else:
        try:
            import dataretrieval.nwis as nwis
        except ImportError:
            print("Error: dataretrieval package not found. Please install with 'pip install dataretrieval'")
            return None
        
        if start_date is None or end_date is None:
            print("Error: start_date and end_date must be provided as datetime objects")
            return None
        
        print(f"Downloading USGS discharge data for station {site_code} from {start_date} to {end_date}")
        print(f"Files will be saved to: {os.path.abspath(output_dir)}")
        
        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Format dates for NWIS service
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # Call NWIS service to download discharge data
        df = nwis.get_record(
            sites=site_code,
            service='iv',             # 'iv' -> instantaneous values
            start=start_date_str,
            end=end_date_str,
            parameterCd='00060'
        )
        # Convert time to UTC
        df = df.tz_convert('UTC')

        if not df.empty:
            # Extract discharge data column and convert units (from cfs to m³/s, conversion factor: 0.0283)
            discharge_cols = [col for col in df.columns if '00060' in col and 'cd' not in col]
            if discharge_cols:
                discharge_col = discharge_cols[0]
                
                # Create result DataFrame - use copy instead of reference
                result_df = pd.DataFrame()
                result_df['datetime'] = df.index.copy()
                
                # Use values directly instead of column reference
                discharge_values = df[discharge_col].values * 0.0283  # Convert cfs to m³/s
                result_df['discharge'] = discharge_values
                
                # If NaN values still exist, try direct loop assignment
                if result_df['discharge'].isna().any():
                    print("NaN values detected, trying direct loop assignment...")
                    new_discharge = []
                    for val in df[discharge_col].values:
                        new_discharge.append(val * 0.0283 if pd.notna(val) else val)
                    result_df['discharge'] = new_discharge
                
                # Save as CSV file
                output_file = os.path.join(output_dir, f'USGS_{site_code}_UTC_m3s.csv')
                result_df.to_csv(output_file, index=False, float_format='%.6f')  # Specify float format
                print(f'Successfully downloaded data for station {site_code} and saved to {output_file}')
                    
                return result_df
            else:
                print(f"Error: No discharge column found for station {site_code}")
        else:
            print(f'No data available for station {site_code}')
    return None
            
def gauge_processor(args):
    
    for idx, gauge in args.gauges_list.iterrows():
        gauge_id = f"{gauge.STAID:0>8}"
        latitude_gauge, longitude_gauge = get_gauge_coordinates(args.gauge_meta_path, gauge_id)
        df_usgs = download_usgs_data(gauge_id, args.time_start, args.time_end, args.usgs_data_path, args.time_step)

    latitude_gauge, longitude_gauge = get_gauge_coordinates(args.gauge_meta_path, args.gauge_id)
    df_usgs = download_usgs_data(args.gauge_id, args.time_start, args.time_end, args.usgs_data_path, args.time_step)
    return latitude_gauge, longitude_gauge



