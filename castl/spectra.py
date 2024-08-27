from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import interp1d
from astropy.table import Table
from astropy.io import fits
import pandas as pd
import numpy as np
import glob
import os
import re

def obspec(input_file): 
    if input_file is None: 
        raise ValueError('Please Input A Input File Directory')
    
    observed = pd.read_csv(input_file)
    observed_wave = observed.iloc[:, 0].tolist()
    observed_flux = observed.iloc[:, 1].tolist()
    
    return observed_wave, observed_flux

def gridinter(temp_grid): 
    if temp_grid is None: 
        ValueError('Please Input the Required Grid')
        
    value_grid = np.array(temp_grid['flux'])
        
    items = list(temp_grid.items())
    remaining_items = items[2:]
    grid_array = (np.array(list((dict(remaining_items)).values()))).T
    
    interpolator = LinearNDInterpolator(grid_array, value_grid)
    
    return interpolator

def respec(input_wave, input_flux, resample_wave): 
    try: 
        f_interp = interp1d(input_wave, input_flux, kind='linear', fill_value="extrapolate")
        resampled_flux = f_interp(resample_wave)
        return resampled_flux
    except: 
        ValueError('Please Input the Required Inputs')
        
def gridspec(model_directory, model_parameters, observed_wave): 
    
    if model_directory is None: 
        raise ValueError('Please Input a Model Directory')
    if model_parameters is None: 
        raise ValueError('Please Input the Model Parameters')
    if observed_wave is None: 
        raise ValueError('Please Input the Observed Wavelength')
    
    model_files = glob.glob(f'{model_directory}/*')
    
    if len(model_files) != 0: 
        model_type = os.path.splitext(model_files[0])[1]
        total_data = {'file_path': [], 'file_data': []}
        if model_type == ".csv":
            for i in range(len(model_files)): 
                temp_file_data = pd.read_csv(model_files[i])
                total_data['file_path'].append(model_files[i])
                total_data['file_data'].append(temp_file_data)
        elif model_type == ".txt":
            for i in range(len(model_files)): 
                temp_file_data = pd.read_csv(model_files[i], delim_whitespace=True)
                total_data['file_path'].append(model_files[i])
                total_data['file_data'].append(temp_file_data)
        elif model_type == ".fits":
            for i in range(len(model_files)): 
                temp_file_hdul = fits.open(model_files[i])
                temp_file_data = (Table(temp_file_hdul[1].data)).to_pandas()
                total_data['file_path'].append(model_files[i])
                total_data['file_data'].append(temp_file_data)
        else: 
            raise ValueError("Invalid Model File Type. Valid Model File Types Include (csv, ascii, fits)")
    else: 
        raise ValueError("File Directory is Empty, Please Input Correct File Directory")
    
    total_grid = {'wavelength': [], 'flux': []}
    for parm in model_parameters:
        total_grid[parm] = []
        
    for j in range(len(total_data['file_path'])): 
        model_wave = total_data['file_data'][j].iloc[:, 0].tolist()
        model_flux = total_data['file_data'][j].iloc[:, 1].tolist()
        
        f_interp = interp1d(model_wave, model_flux, kind='linear', fill_value="extrapolate")
        resampled_flux = f_interp(observed_wave)
        percentile_999 = np.nanmax(resampled_flux)
        resampled_flux /= percentile_999
        
        total_grid['wavelength'].append(observed_wave)
        total_grid['flux'].append(resampled_flux)
        
        numbers = re.findall(r'-?\d+\.?\d*', total_data['file_path'][j])
        for p in range(len(numbers)): 
            total_grid[model_parameters[p]].append(numbers[p])
            
    return total_grid

def normspec(flux): 
    if flux is None: 
        raise ValueError('Please Input an Array of Fluxes')
    
    normalized_flux = [x / np.nanpercentile(flux, 99.9) for x in flux]
    normalized_flux = [x if x >= 0 else 2.2250738585072014e-30 for x in flux]
    normalized_flux = np.nan_to_num(flux, nan=2.2250738585072014e-30)
    
    return normalized_flux
