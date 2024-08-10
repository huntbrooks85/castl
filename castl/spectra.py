from astropy.table import Table
from astropy.io import fits
import pandas as pd
import numpy as np
import glob
import os

def observed_spectra(input_file): 
    if input_file is None: 
        raise ValueError('Please Input A Input File Directory')
    
    try: 
        observed = pd.read_csv(input_file)

        observed_wave = observed.iloc[:, 0].tolist()
        observed_flux = observed.iloc[:, 1].tolist()
    except ValueError: 
        raise ValueError
    
    return observed_wave, observed_flux

def model_spectra(model_directory, model_parameters): 
    
    if model_directory is None: 
        raise ValueError('Please Input a Model Directory')
    if model_parameters is None: 
        raise ValueError('Please Input the Model Parameters')
    
    model_files = glob.glob(f'{model_directory}*')
    
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
                temp_file_data = pd.read_csv(model_files[i], delimiter='\t')
                total_data['file_path'].append(model_files[i])
                total_data['file_data'].append(temp_file_data)
        elif model_type == ".fits":
            for i in range(len(model_files)): 
                temp_file_hdul = fits.open(model_files[i])
                temp_file_data = Table(temp_file_hdul[1].data)
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
        total_grid['wavelength'].append(model_wave)
        total_grid['flux'].append(model_flux)
        
        file_name = (total_data['file_path'][j].split(model_directory)[1]).split(model_type)[0]
        for parm in model_parameters: 
            total_grid[parm].append((file_name.split(parm)[1]).split('_')[0])
            
    return total_grid

def normalize_flux(flux): 
    if flux is None: 
        raise ValueError('Please Input an Array of Fluxes')
    
    try:
        normalized_flux = [x / np.nanpercentile(flux, 99.9) for x in flux]
        normalized_flux = [x if x >= 0 else 2.2250738585072014e-30 for x in flux]
        normalized_flux = np.nan_to_num(flux, nan=2.2250738585072014e-30)
    except ValueError: 
        raise ValueError
    
    return normalized_flux
