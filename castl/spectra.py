from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import NearestNDInterpolator
from concurrent.futures import ThreadPoolExecutor
from scipy.interpolate import interp1d
from astropy.table import Table
from astropy.io import fits
from tqdm import tqdm
import pandas as pd
import numpy as np
import glob
import os
import re
import os

def obspec(input_file): 
    observed = pd.read_csv(input_file)
    observed_wave = observed.iloc[:, 0].tolist()
    observed_flux = observed.iloc[:, 1].tolist()
    
    observed_flux = [x / np.nanpercentile(observed_flux, 99.9) for x in observed_flux]
    observed_flux = [x if x >= 0 else 2.2250738585072014e-30 for x in observed_flux]
    observed_flux = (np.nan_to_num(observed_flux, nan=2.2250738585072014e-30))
    
    return observed_wave, observed_flux
     
def gridspec(model_directory, model_parameters, observed_wave, unit_convert): 
    model_files = glob.glob(f'{model_directory}/*')
    num_files = len(model_files)

    model_type = os.path.splitext(model_files[0])[1]
    total_data = {'file_path': [], 'file_data': []}

    def process_file(file_path):
        if model_type == ".csv":
            return file_path, pd.read_csv(file_path)
        elif model_type == ".txt":
            return file_path, pd.read_csv(file_path, delim_whitespace=True)
        elif model_type == ".fits":
            with fits.open(file_path) as hdul:
                data = Table(hdul[1].data).to_pandas()
            return file_path, data
        else:
            raise ValueError("Invalid Model File Type. Valid Model File Types Include (csv, ascii, fits)")

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_file, model_files), total=num_files))

    for file_path, file_data in results:
        total_data['file_path'].append(file_path)
        total_data['file_data'].append(file_data)
        
        
    total_grid = {'wavelength': [], 'flux': []}
    for parm in model_parameters:
        total_grid[parm] = []
    
    observed_wave = np.array(observed_wave)

    for j in tqdm(range(num_files)): 
        model_wave = np.array(total_data['file_data'][j].iloc[:, 0]) * unit_convert
        model_flux = np.array(total_data['file_data'][j].iloc[:, 1])

        f_interp = interp1d(model_wave, model_flux, kind='linear', fill_value="extrapolate")
        resampled_flux = f_interp(observed_wave)
        resampled_flux /= np.nanmax(resampled_flux)  # Normalize by the max value

        total_grid['wavelength'].append(observed_wave)
        total_grid['flux'].append(resampled_flux)

        numbers = re.findall(r'-?\d+\.?\d*', total_data['file_path'][j])
        for p, number in enumerate(numbers): 
            if p < len(model_parameters):
                total_grid[model_parameters[p]].append(number)

    return total_grid

def respec(input_wave, input_flux, resample_wave): 
    f_interp = interp1d(input_wave, input_flux, kind='linear', fill_value="extrapolate")
    resampled_flux = f_interp(resample_wave)
    return resampled_flux
        
def gridinter(temp_grid, type): 
    if temp_grid is None: 
        ValueError('Please Input the Required Grid')
    if type is None: 
        type = 'nearest'
        
    value_grid = np.array(temp_grid['flux'])
        
    items = list(temp_grid.items())
    remaining_items = items[2:]
    grid_array = (np.array(list((dict(remaining_items)).values()))).T
    
    if type == 'nearest':
        interpolator = NearestNDInterpolator(grid_array, value_grid)
    elif type == 'total':
        interpolator = LinearNDInterpolator(grid_array, value_grid)
    
    return interpolator
