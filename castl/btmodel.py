#-----------------------------------------------------------------------#
# castl.btmodel v0.6.2
# By Hunter Brooks, at NAU, Flagstaff: Mar. 26, 2025
#
# Purpose: Find best fit grid point using chi square
#-----------------------------------------------------------------------#

# ------ Import File Processing Packages ------ #
from tqdm import tqdm
import h5py
# --------------------------------------------- # 

# ------ Import File Loading Packages ------ #
from astropy.table import Table
from astropy.io import fits
import pandas as pd
# ------------------------------------------ #

# ------ Import Math Packages ------ #
from scipy.interpolate import interp1d
import astropy.units as u
import numpy as np
# ---------------------------------- #

# ------ Ignore all warnings ------ #
import warnings
warnings.filterwarnings('ignore')
# --------------------------------- #

# --------------------------------- #
def btmodel(input_file, model_directory, model_parm, unit_wave=[u.um, u.um]): 
    '''
    Runs a simple chi square fit on the model.

        Parameters:
            input_file (str): Input .csv file with observed spectrum
            model_directory (str): Model directory path to h5 file
            model_parm (list): List of model parameters, in order of the model file name
            unit_wave (list): List of astropy units for the model and observed wavelength untis

        Returns:
            Chi Square (float): The best fit model spectrum chi square value
            Best Fit Parameters (array): Exact Model Best Fit Grid Point

    '''
    
    if type(input_file) != str: 
        ValueError(f'Input a string as your input_file. Current variable type: {type(input_file)}')
    
    if (type(model_directory) != str) and (model_directory[-3:] != '.h5'): 
        ValueError(f'Inputted Wrong File Type. Current file type: {model_directory[-3:]}')
        
    if type(model_parm) != list: 
        ValueError(f'Input a list as your model_parm. Current variable type: {type(model_parm)}')
    
    # ------ Load in the observed models and spectra ------ #
    wave, flux, unc = obspec(input_file)
    # ----------------------------------------------------- #
    
    # ------ Build interpolation grid and start mcmc calculation ------ #
    return chisquare(model_parm, model_directory, wave, flux, unc, unit_wave)   
    
# --------------------------------- #

# --------------------------------- #
def obspec(input_file): 
    '''
    Loads in Observed Spectrum and Converts Observed Flux and Uncertainty Units to Model Units

        Parameters:
            input_file (str): Input .csv file with observed spectrum
            unit_wave (list): List of astropy units for the model and observed wavelength untis
            unit_flux (list): List of astropy units for the model and observed flux untis

        Returns:
            observed_wave (array): Observed spectrum wavelength
            observed_flux (array): Unit converted observed spectrum flux
            observed_unc (array): Unit converted observed spectrum uncertainties 
    '''
    
    file_ext = input_file.split('.')[-1].lower()  # Get file extension

    if file_ext == "csv":  # CSV files
        observed = pd.read_csv(input_file)
        
    elif file_ext == "vot":  # VOTable files
        votable = votable.parse(input_file) 
        observed = votable.get_table() 

    elif file_ext in ["txt", "dat", "tbl"]:  # ASCII tables with whitespace or tab delimiters
        observed = pd.read_csv(input_file, delim_whitespace=True)
        
    elif file_ext in ["tsv"]:  # Loads Tab-Separated files
        observed = pd.read_csv(input_file, sep="\t")

    elif file_ext == "fits":  # FITS files
        with fits.open(input_file) as hdul:
            data = Table(hdul[1].data).to_pandas()
        observed = data
        
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    # Load in the wavelength, flux, and uncertainties
    try: 
        observed_wave = observed.iloc[:, 0].tolist()
        observed_flux = observed.iloc[:, 1].tolist()
        observed_unc = observed.iloc[:, 2].tolist()
    except: 
        ValueError('Observed spectrum is missing uncertainties')
    
    # Filter both lists using the indices to keep
    indices_to_keep = [i for i, flux in enumerate(observed_flux) if not np.isnan(flux) and flux > 0]
    observed_wave = ([observed_wave[i] for i in indices_to_keep])
    observed_flux = ([observed_flux[i] for i in indices_to_keep])
    observed_unc = ([observed_unc[i] for i in indices_to_keep])
    
    # Convert flux and uncertainty to model flux units only once (remove repeated unit operations)
    observed_flux = np.array(observed_flux)
    observed_unc = np.array(observed_unc)
    
    # Converts observed spectrum to model units
    observed_unc = np.divide(observed_unc, np.nanmax(observed_flux))
    observed_flux = np.divide(observed_flux, np.nanmax(observed_flux))

    return observed_wave, observed_flux, observed_unc

# --------------------------------- #
def chisquare(parm_list, model_directory, observed_wave, observed_flux, observed_unc, unit_wave):
    '''
    Runs chi square fit to find best fit grid point

        Parameters:
            model_parm (list): List of model parameters, in order of the model file name
            model_directory (str): Model directory path to h5 file
            
            observed_wave (array): Observed spectrum wavelength
            observed_flux (array): Normalized observed spectrum flux
            observed_unc (array): Normalized observed spectrum uncertainties 

        Returns:
            Chi Square (float): The best fit model spectrum chi square value
            Best Fit Parameters (array): Exact Model Best Fit Grid Point

    '''
    
    # Pre-load the file and store all data in memory
    with h5py.File(f"{model_directory}", "r") as h5f:
        original_names = {key: h5f[key].attrs.get("original_name", key) for key in h5f.keys()}
        loaded_grid = {original_names[key]: h5f[key][()] for key in h5f.keys()}

    # Load in model wavelength, flux, and grid data
    wavelength_data_list = loaded_grid['wavelength']
    flux_data_list = loaded_grid['flux'] 
    grid_params = np.column_stack([loaded_grid[key] for key in parm_list]) 

    # Convert the observed data to float32
    observed_wave = np.array(observed_wave, dtype=np.float32)
    observed_flux = np.array(observed_flux, dtype=np.float32)
    observed_unc = np.array(observed_unc, dtype=np.float32)

    # Pre-allocate the list for total chi-squares
    total_chi_list = []

    # Processes the flux and calculates chi-squares
    for wavelength_data, flux_data in tqdm(zip(wavelength_data_list, flux_data_list), total=len(wavelength_data_list), desc="Finding Best Model", ncols=100, unit="step"):
        # Convert wavelength to the appropriate unit
        wave_values = (np.array(wavelength_data) * unit_wave[1]).to(unit_wave[0]).value.astype(np.float32)
        flux_data = np.array(flux_data, dtype=np.float32)
        flux_data = np.divide(flux_data, np.nanmax(flux_data))
        
        # Perform interpolation
        f_interp = interp1d(wave_values, flux_data, kind='linear', fill_value="extrapolate", assume_sorted=True)
        new_flux = np.array(f_interp(observed_wave), dtype=np.float32)  

        # Calculate chi-square for each model
        diff = observed_flux - new_flux
        chi_square = np.nansum(np.square(diff / observed_unc)) / (len(observed_flux) - len(parm_list))

        total_chi_list.append(chi_square)

        # Clean up memory
        del f_interp, wave_values, flux_data

    # Find the model with the minimum chi-square
    min_value = float('inf')
    for i, chi in enumerate(total_chi_list):
        if chi < min_value:
            min_value = chi
            min_sublist_idx = i

    # Return the best model parameters
    best_point = list(grid_params[min_sublist_idx])
    return min_value, best_point
# --------------------------------- #