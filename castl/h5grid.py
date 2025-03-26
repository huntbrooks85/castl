#-----------------------------------------------------------------------#
# castl.h5grid v0.6.2
# By Hunter Brooks, at NAU, Flagstaff: Mar. 26, 2025
#
# Purpose: Build the h5 spectral model grid
#-----------------------------------------------------------------------#

# ------ Import File Processing Packages ------ #
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import glob
import h5py
import os
import re
# --------------------------------------------- # 

# ------ Import File Loading Packages ------ #
from astropy.table import Table
from astropy.io import fits
import xarray as xr
import pandas as pd
# ------------------------------------------ #

# ------ Import Math Packages ------ #
import numpy as np
# ---------------------------------- #

# ------ Ignore all warnings ------ #
import warnings
warnings.filterwarnings('ignore')
# --------------------------------- #
     
# --------------------------------- #     
def h5grid(model_directory, model_parameters, output_file, wavelength_region = [0, np.inf]): 
    '''
    Loads model grid and saves it to a loadable h5 file

        Parameters:
            model_directory (str): Model directory path
            model_parameters (list): List of model parameters, in order of the model file name
            output_file (str): Output file name
            wavelength_region (list): Wavelength region saved to h5 file

        Returns:
            Model Grid (h5): An h5 file with model parameters, wavelength, and flux
    '''
    
    # Obtain all model file path directories
    model_files = glob.glob(f'{model_directory}/*')
    num_files = len(model_files)

    # Build a temporary dictionary to load in model spectra
    model_type = model_files[0].split('.')[-1].lower()
    total_data = {'file_path': [], 'file_data': []}

    # Function to load in csv, ascii, and fits model tables
    def process_file(file_path):
        if model_type == "csv":  # Loads CSV files
            return file_path, pd.read_csv(file_path)

        elif model_type in ["txt", "dat", "tbl"]:  # Loads ASCII tables
            return file_path, pd.read_csv(file_path, delim_whitespace=True, comment="#")

        elif model_type == "fits":  # Loads FITS files
            with fits.open(file_path) as hdul:
                data = Table(hdul[1].data).to_pandas()
            return file_path, data

        elif model_type == "vot":  # Loads VOTable files
            votable = votable.parse(file_path)
            data = votable.get_table()
            return file_path, data
        
        elif model_type == "nc":  # Loads NetCDF files using xarray
            with xr.open_dataset(file_path) as ds:
                var_names = list(ds.data_vars.keys())[:2]
                data_arrays = [ds[var].values.squeeze() for var in var_names]
                min_size = min(arr.shape[0] for arr in data_arrays if arr.ndim == 1)
                trimmed_data = [arr[:min_size] for arr in data_arrays]    
                df = pd.DataFrame(np.column_stack(trimmed_data), columns=var_names)
            
            del ds, data_arrays, trimmed_data, var_names
            return file_path, df

        elif model_type in ["tsv"]:  # Loads Tab-Separated files
            return file_path, pd.read_csv(file_path, sep="\t")

        else:
            raise ValueError(f"Unsupported file type: {model_type}")

    # Multi-threads loading the model spectra
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_file, model_files), total=num_files, desc='Loading Model Spectra: '))

    # Saves the data to the temporary model dictionary
    for file_path, file_data in results:
        total_data['file_path'].append(file_path)
        total_data['file_data'].append(file_data)
        
    # Build new, final, dictionary containing every parameter in the grid    
    total_grid = {'wavelength': [], 'flux': []}
    for parm in model_parameters:
        total_grid[parm] = []
    
    # Makes observed wavelengths an array for interpolation
    model_parameters_len = len(model_parameters)
    for j in tqdm(range(num_files), total=num_files, desc='Building Model Grid: '): 
        # Convert to numpy arrays once
        model_wave = np.array(total_data['file_data'][j].iloc[:, 0])  # Wavelength conversion
        model_flux = np.array(total_data['file_data'][j].iloc[:, 1])
        
        # Apply the wavelength filter
        mask = (model_wave >= wavelength_region[0]) & (model_wave <= wavelength_region[1])
        filtered_wave = np.array(model_wave[mask])
        filtered_flux = np.array(model_flux[mask])
        
        # Append results efficiently
        total_grid['wavelength'].append(filtered_wave)
        total_grid['flux'].append(filtered_flux)

        # Extract model parameters efficiently (using regex once)
        numbers = re.findall(r'-?\d+\.?\d*', total_data['file_path'][j].split('/', 100000)[-1])
        for p in range(min(len(numbers), model_parameters_len)): 
            total_grid[model_parameters[p]].append(float(numbers[p]))
            
        # Drop the first column in the DataFrame (assuming it's the one you want to remove)
        total_data['file_data'][j] = total_data['file_data'][j].drop(total_data['file_data'][j].columns[0], axis=1)
    
    total_data = None

    # Save to HDF5
    with h5py.File(f"{output_file}.h5", "w") as h5f:
        for key, value in total_grid.items():
            safe_key = key.replace("/", "|")

            if isinstance(value, list):
                try:
                    value = np.array(value, dtype=np.float32)
                except: 
                    max_len = max(len(item) for item in value)
                    padded_value = [item.tolist() + [np.nan] * int(max_len - len(item)) for item in value]
                    value = np.array(padded_value, dtype=np.float32)

            elif isinstance(value, (int, float)):
                value = np.array([value], dtype=np.float32) 
                
            h5f.create_dataset(safe_key, data=value, compression="gzip", compression_opts=9, shuffle=True, dtype=value.dtype, chunks=True)
            h5f[safe_key].attrs["original_name"] = key
# --------------------------------- #            