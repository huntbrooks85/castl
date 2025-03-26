#-----------------------------------------------------------------------#
# castl.mcfit v0.6.2
# By Hunter Brooks, at NAU, Flagstaff: Mar. 26, 2025
#
# Purpose: Perform MCMC calculation on model spectra
#
# There were 2 developers, 
# myself and the Devil, 
# and I have forgotten how it works.
#-----------------------------------------------------------------------#

# ------ Import Markov-Chain Monte-Carlo Packages ------ # 
import emcee
# ------------------------------------------------------- # 

# ------ Import File Processing Packages ------ #
from IPython.display import clear_output
from tqdm import tqdm
import h5py
# --------------------------------------------- # 

# ------ Import File Loading Packages ------ #
from astropy.table import Table
from astropy.io import fits
import pandas as pd
# ------------------------------------------ #

# ------ Import Plotting Packages ------ #
from IPython.display import display, Math
import matplotlib.pyplot as plt
import corner
# -------------------------------------- #

# ------ Import Math Packages ------ #
from scipy.interpolate import LinearNDInterpolator
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
import astropy.units as u
import numpy as np
# ---------------------------------- #

# ------ Ignore all warnings ------ #
import warnings
warnings.filterwarnings('ignore')
# --------------------------------- #

# --------------------------------- #
def mcfit(input_file, output_file, model_directory, model_parm, 
          grid_scale = 10, unit_wave=[u.um, u.um], unit_flux=[(u.erg / (u.cm**2 * u.s * u.um)), (u.erg / (u.cm**2 * u.s * u.um))], 
          walkers=15, steps=1000, 
          rv_fit=False, monitor=False, save_output=True): 
    '''
    Runs Spectral Fitting Markov-Chain Monte-Carlo Simulations Based on Inputted Model Spectra

        Parameters:
            input_file (str): Input .csv file with observed spectrum
            output_file (str): Output file name that is used when saving tables and figures
            model_directory (str): Model directory path to h5 file
            model_parm (list): List of model parameters, in order of the model file name
            grid_scale (int): Number of points used in each dimension of the grid
            unit_wave (list): List of astropy units for the model and observed wavelength untis
            unit_flux (list): List of astropy units for the model and observed flux untis
            walkers (int): Number of MCMC walkers
            steps (int): Number of Steps
            rv_fit (boolean): When 'True' fits a radial velocity in km/s
            monitor (boolean): When 'True' step monitors are activated
            save_output (boolean): When 'True' saves best fit line and corner plot

        Returns:
            Chi Square (float): The best fit model spectrum chi square value
            Best Fit Parameters (array): Best fit value for each parameter and the associated uncertainties
            Table (csv): Saves best fit model spectrum
            Table (h5): Saves all walker information
            Figures (pdf): Save value over step and combined figure with best fit model compared to observed and corner plot
    '''
    
    if type(input_file) != str: 
        ValueError(f'Input a string as your input_file. Current variable type: {type(input_file)}')
        
    if type(output_file) != str: 
        ValueError(f'Input a string as your output_file. Current variable type: {type(output_file)}')
    
    if (type(model_directory) != str) and (model_directory[-3:] != '.h5'): 
        ValueError(f'Inputted Wrong File Type. Current file type: {model_directory[-3:]}')
        
    if type(model_parm) != list: 
        ValueError(f'Input a list as your model_parm. Current variable type: {type(model_parm)}')
    
    # Ensures there are enough walkers for emcee
    if walkers < 2*(len(model_parm) + 3): 
        walkers = 2*(len(model_parm) + 3)
    
    # ------ Load in the observed models and spectra ------ #
    wave, flux, unc = obspec(input_file, unit_wave, unit_flux)
    # ----------------------------------------------------- #
    
    # ------ Build interpolation grid and start mcmc calculation ------ #
    inter, scaler, grid, best_point = gridinter(model_parm, model_directory, grid_scale, wave, flux, unc, unit_wave, rv_fit)
    sampler, discard = specmc(model_parm, inter, scaler, best_point, wave, flux, unc, grid, walkers, steps, rv_fit, monitor)
    # ----------------------------------------------------------------- #
    
    # ------ Provides user with mcmc best fit model ------ #
    if save_output == True:
        chi_square, best_fit_params = mcplot(sampler, discard, output_file, inter, scaler, wave, flux, unc, model_parm, unit_wave, unit_flux, rv_fit)
        return chi_square, best_fit_params
    # ---------------------------------------------------- #
# --------------------------------- #

# --------------------------------- #
def obspec(input_file, unit_wave, unit_flux): 
    '''
    Loads in Observed Spectrum and Converts Observed Flux and Uncertainty Units to Model Units

        Parameters:
            input_file (str): Input .csv file with observed spectrum
            unit_wave (list): List of astropy units for the model and observed wavelength untis
            unit_flux (list): List of astropy units for the model and observed flux untis

        Returns:
            observed_wave (array): Observed spectrum wavelength
            observed_flux (array): Unit converted observed spectrum flux
            observed_unc_erg (array): Unit converted observed spectrum uncertainties 
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
    observed_flux_erg = observed_flux * unit_flux[0] 
    observed_unc_erg = observed_unc * unit_flux[0]
    
    # Converts observed spectrum to model units
    observed_flux_erg = (observed_flux_erg.to_value(unit_flux[1], equivalencies=u.spectral_density(observed_wave * unit_wave[0])))
    observed_unc_erg =  (observed_unc_erg.to_value(unit_flux[1], equivalencies=u.spectral_density(observed_wave * unit_wave[0])))

    return observed_wave, observed_flux_erg, observed_unc_erg

# --------------------------------- #
def gridinter(parm_list, model_directory, grid_scale, observed_wave, observed_flux, observed_unc, unit_wave, rv_fit):
    '''
    1) Load in model spectral data
    2) Resamples model to observed wavelength
    3) Finds best grid point along the model
    4) Builds interpolator for spectral model data

        Parameters:
            parm_list (list): List of model parameters, in order of the model file name
            model_directory (str): Model directory path
            grid_scale (int): Number of points used in each dimension of the grid
            observed_wave (array): Observed wave data from "obspec"
            observed_flux (array): Observed flux data from "obspec"
            observed_unc (array): Observed uncertainty data from "obspec"    
            unit_wave (list): List of astropy units for the model and observed wavelength untis
            rv_fit (boolean): When 'True' fits a radial velocity in km/s

        Returns:
            interpolator (array): Linear interpolator for the spectral model grid
            scaler (array): Normalizing array for the model grid
            grid_params (array): Grid point array for the model grid
            best_point (list): The best point found along the model grid
    '''
    
    # Pre-load the file and store all data in memory
    with h5py.File(f"{model_directory}", "r") as h5f:
        original_names = {key: h5f[key].attrs.get("original_name", key) for key in h5f.keys()}
        loaded_grid = {original_names[key]: h5f[key][()] for key in h5f.keys()}
        del original_names

    # Load in model wavelength, flux, and grid data
    wavelength_data_list = loaded_grid['wavelength']
    flux_data_list = loaded_grid['flux'] 
    grid_params = np.column_stack([loaded_grid[key] for key in parm_list]) 
    del loaded_grid

    # Interpolation function and replacement using a generator to avoid storing all flux data
    observed_wave = np.array(observed_wave, dtype=np.float32)
    observed_flux = np.array(observed_flux, dtype=np.float32)
    observed_unc = np.array(observed_unc, dtype=np.float32)
    dist_list = np.linspace(-27.2, -15.3, 50)
    
    # Processes the flux and appends chi square list to find best grid point
    def process_flux():
        for wavelength_data, flux_data in tqdm(zip(wavelength_data_list, flux_data_list), total=len(wavelength_data_list), desc="Resampling Flux: ", ncols=100, unit="step"):
            # Convert data to float32 to save memory
            wave_values = (np.array(wavelength_data) * unit_wave[1]).to(unit_wave[0]).value.astype(np.float32)
            flux_data = np.array(flux_data, dtype=np.float32)

            # Perform interpolation
            f_interp = interp1d(wave_values, flux_data, kind='linear', fill_value="extrapolate", assume_sorted=True)
            new_flux = np.array(f_interp(observed_wave), dtype=np.float32)  
            
            # Calculates chi square for each loaded model grid point
            chi_list = []
            for dist in dist_list:
                observed_to_model_flux = observed_flux / 10**(dist)
                observed_to_model_unc = observed_unc / 10**(dist)
                diff = np.subtract(observed_to_model_flux, new_flux)
                chi_square = np.nansum(np.square(diff / observed_to_model_unc)) / ((len(observed_flux) - len(parm_list)))
                chi_list.append(chi_square)
            total_chi_list.append(chi_list)

            del f_interp, wave_values, flux_data, chi_list  
            yield new_flux

    # Process flux data on the fly instead of storing in a list
    total_chi_list = []
    new_flux_data_list = list(process_flux()) 
    
    # Finds sublist index and index of the minimum point
    min_value = float('inf')
    for i, sublist in enumerate(total_chi_list):
        j = sublist.index(min(sublist))
        if sublist[j] < min_value: 
            min_value = sublist[j]
            min_sublist_idx = i
            min_element_idx = j
    
    # Generates best grid point along the model
    if rv_fit: 
        best_point = list(grid_params[min_sublist_idx]) + [0, dist_list[min_element_idx], 1]
    else: 
        best_point = list(grid_params[min_sublist_idx]) + [dist_list[min_element_idx], 1]
    
    # Normalize parameters
    scaler = MinMaxScaler()
    normalized_grid_params = scaler.fit_transform(grid_params)
    N, D = grid_params.shape
    
    new_flux_data_list = np.array(new_flux_data_list, dtype=np.float32)

    # Normalize best_point_array using the same scaler used for grid_params
    if rv_fit:
        best_point_array = np.array(best_point[:-3])
    else: 
        best_point_array = np.array(best_point[:-2])
    best_point_array = best_point_array.reshape(1, -1)
    best_point_normalized = scaler.transform(best_point_array)

    # Select the nearest "grid_scale" points for each dimension
    nearest_points_indices = []
    for dim in range(normalized_grid_params.shape[1]):
        distances = np.abs(normalized_grid_params[:, dim] - best_point_normalized[0, dim]) 
        sorted_indices = np.argsort(distances)
        nearest_points_indices.append(sorted_indices[:grid_scale])

    # Flatten the indices for the closest 2 points across all dimensions
    nearest_points_indices = np.unique(np.concatenate(nearest_points_indices))
    subset_grid_params = normalized_grid_params[nearest_points_indices] 
    subset_flux_data = new_flux_data_list[nearest_points_indices]

    # Builds interpolator based on grid size
    if D < 5:
        interpolator = LinearNDInterpolator(subset_grid_params, subset_flux_data)
    else:
        interpolator = RBFInterpolator(subset_grid_params, subset_flux_data)

    return interpolator, scaler, grid_params, best_point
# --------------------------------- #

# --------------------------------- #
def statmc(observed_wave, observed_flux, observed_unc, interpolator, scaler, parm, rv_fit):
    """
    Compute the log-likelihood including radial velocity shift.
    Optimized for speed.
    """
    # Ensure inputs are NumPy arrays
    observed_wave = np.asarray(observed_wave, dtype=np.float64)
    observed_flux = np.asarray(observed_flux, dtype=np.float64)
    observed_unc = np.asarray(observed_unc, dtype=np.float64)

    # Model interpolation
    if rv_fit:
        point = np.array(parm[:-3]).reshape(1, -1)
    else: 
        point = np.array(parm[:-2]).reshape(1, -1)
    normalized_new_point = scaler.transform(point)
    model_flux = interpolator(normalized_new_point)

    # Apply Gaussian smoothing
    resampled_flux = gaussian_filter1d(model_flux, parm[-1], mode='nearest')[0]
    
    # Apply Doppler shift
    if rv_fit: 
        v_r = parm[-3]
        c_inv = 1 / 299792.458
        scale_factor = np.sqrt((1 + v_r * c_inv) / (1 - v_r * c_inv))
        shifted_wave = observed_wave * scale_factor
        resampled_flux = np.interp(observed_wave, shifted_wave, resampled_flux, left=np.nan, right=np.nan)

    # Flux scaling
    dilution_factor = 10**(-parm[-2])
    observed_to_model_flux = observed_flux * dilution_factor
    observed_to_model_unc = observed_unc * dilution_factor

    # Compute chi-square efficiently
    diff = observed_to_model_flux - resampled_flux
    inv_unc_sq = np.reciprocal(observed_to_model_unc**2, where=observed_to_model_unc > 0, out=np.zeros_like(observed_to_model_unc))
    chi_square = np.nansum(diff**2 * inv_unc_sq) / (len(observed_flux) - len(parm))

    return -0.5 * chi_square
# --------------------------------- #

# --------------------------------- #
def prior(parm, parm_bound):
    if np.any(parm < np.array([low for low, high in parm_bound])) or np.any(parm > np.array([high for low, high in parm_bound])):
        return -np.inf
    return 0
# --------------------------------- #

# --------------------------------- #
def log_posterior(parm, observed_wave, observed_flux, unc, interpolator, scaler, parm_bound, rv_fit):
    lp = prior(parm, parm_bound)
    if not np.isfinite(lp):
        return -np.inf
    
    stat = statmc(observed_wave, observed_flux, unc, interpolator, scaler, parm, rv_fit)  
    return lp + stat if stat != 0 else -np.inf
# --------------------------------- #

# --------------------------------- #
def specmc(model_parm, interpolator, scaler, best_point, observed_wave, observed_flux, unc, grid, walkers, max_step, rv_fit, monitor):
    '''
    Runs the MCMC calculation using the interpolator from "gridinter"

        Parameters:
            model_parm (list): List of model parameters, in order of the model file name
            inperpolator (array): Linear interpolator for the spectral model grid
            scalar (array): Normalizing array for the model grid
            best_point (list): The best point found along the model grid
            observed_flux (array): Observed flux data from "obspec"
            unc (array): Observed uncertainty data from "obspec"    
            grid (array): Grid point array for the model grid
            walkers (int): Number of MCMC walkers
            max_step (int): Number of Steps
            rv_fit (boolean): When 'True' fits a radial velocity in km/s
            monitor (boolean): When 'True' step monitors are activated

        Returns:
            sampler (array): The entire emcee sampler array
            best_discard (int): Number of discards (25% of total steps)
    '''
    
    # Finds the min and max for each dimension in the grid
    def get_min_max_ranges(grid):
        return [(min(grid[:, i]), max(grid[:, i])) for i in range(grid.shape[1])]

    # Creates list of the parameter bounds
    if rv_fit: 
        parm_bound = get_min_max_ranges(grid) + [(-2500, 2500), (np.log10(5.1332207142e-28), np.log10(5.1332207142e-16)), (0.5, 5)]
    else: 
        parm_bound = get_min_max_ranges(grid) + [(np.log10(5.1332207142e-28), np.log10(5.1332207142e-16)), (0.5, 5)]
    n_params = len(parm_bound)

    # Creates the intial positions around a random 10% ball of the best point
    initial_positions = np.zeros((walkers, n_params))
    for i, (low, high) in enumerate(parm_bound):
        best_value = best_point[i]
        param_range = high - low
        random_offsets = np.random.uniform(-0.1 * param_range, 0.1 * param_range, size=walkers)
        initial_positions[:, i] = np.clip(best_value + random_offsets, low, high)

    # GAMMA STEPA
    moves = [
        (emcee.moves.DESnookerMove(), 0.05), 
        (emcee.moves.DEMove(gamma0=1.2), 0.4), 
        (emcee.moves.StretchMove(a=5), 0.55)
    ]

    # Use multiprocessing Pool
    sampler = emcee.EnsembleSampler(walkers, n_params, log_posterior, moves=moves, 
                                    args=(observed_wave, observed_flux, unc, interpolator, scaler, parm_bound, rv_fit))
    
    # Starts running the emcee sampler checking every 1000 steps
    if rv_fit: 
        labels = model_parm + ['Radial Velocity', 'Dilution Factor', 'Smoothing']
    else: 
        labels = model_parm + ['Dilution Factor', 'Smoothing']
    best_discard = 0
    for sample in tqdm(sampler.sample(initial_positions, iterations=max_step), total=max_step, desc="Starting MCMC: ", ncols=100, unit="step"):
        if sampler.iteration == max_step:
            best_discard = max_step * 0.25
            break
        if sampler.iteration % 1000:
            continue

        # Monitoring step
        if monitor: 
            plt.clf()
            plt.close('all')
            clear_output()
            print('<------ Monitor MCMC Calculation ------>')
            fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 4))
            for i, ax in enumerate(axes):
                for walker in sampler.get_chain()[:, :, i].T:
                    ax.plot(walker, alpha=0.5)
                ax.set_title(labels[i])
                ax.set_xlabel("Step")
            plt.tight_layout()
            plt.show()   
    
    clear_output()
    return sampler, best_discard
# --------------------------------- #

# --------------------------------- #
def mcbest(sampler, interpolator, scaler, observed_flux, unc, discard, labels, rv_fit): 
    '''
    Finds the best fit parameters and prints them out

        Parameters:
            sampler (array): The entire emcee sampler array
            inperpolator (array): Linear interpolator for the spectral model grid
            scalar (array): Normalizing array for the model grid
            observed_wave (array): Observed wave data from "obspec"
            observed_flux (array): Observed flux data from "obspec"
            discard (int): Number of discards (25% of total steps)
            labels (list): List of model parameters, in order of the model file name
            rv_fit (boolean): When 'True' fits a radial velocity in km/s

        Returns:
            Best Fit Parameters (printed): Best fit value for each parameter and the associated uncertainties
    '''
    
    # Obtains the shape of the Markov-Chain Monte-Carlo sample
    chain_shape = sampler.chain.shape
    
    # Flattens the Markov-Chain Monte-Carlo sample
    flat_samples = sampler.get_chain(discard=int(discard), flat=True)
    best_fit_params = []
    
    if rv_fit: 
        labels = labels + ['Radial Velocity', 'Dilution Factor']
    else: 
        labels = labels + ['Dilution Factor']
    
    # Finds the best fit parameters with their associated uncertainties
    print('<------ Best Fit Parameters ------>')
    print(f'Number of Discards: {int(discard)}')
    for i in range(chain_shape[2]):
        mcmc = np.percentile(flat_samples[:, i], [5, 50, 95])
        best_fit_params.append([mcmc[0], mcmc[1], mcmc[2]])
        
        if i != (chain_shape[2] - 1):
            print(f'{labels[i]}: {mcmc[1]:.5f} \n (Upper: {np.abs(mcmc[1] - mcmc[2]):.5f}, Lower: {np.abs(mcmc[0] - mcmc[1]):.5f})')
    
    best_values = [sublist[len(sublist) // 2] for sublist in best_fit_params][:-1]
    
    if rv_fit: 
        point = np.array(np.array(best_values[:-2]).reshape(1, -1))
    else: 
        point = np.array(np.array(best_values[:-1]).reshape(1, -1))
    normalized_new_point = scaler.transform(point)
    interpolated_flux = interpolator(normalized_new_point)
    model_flux = gaussian_filter1d(interpolated_flux[0], best_fit_params[-1][1])
    
    # Scales interpolated spectrum
    observed_flux = observed_flux / (10**best_fit_params[-2][1])
    unc = unc / (10**best_fit_params[-2][1])

    chi_square = (np.nansum(np.square(np.divide(np.subtract(observed_flux, model_flux), unc))))/len(observed_flux)
    
    return chi_square, best_fit_params
# --------------------------------- #

# --------------------------------- #
def mcplot(sampler, discard, output_file, interpolator, scaler, observed_wave, observed_flux, unc, labels, unit_wave, unit_flux, rv_fit): 
    '''
    Saves best fit spectra, the emcee sampler, step figure, spectra comparison figure, and corner plot

        Parameters:
            sampler (array): The entire emcee sampler array
            discard (int): Number of discards (25% of total steps)
            output_file (str): Out file name that is used when saving tables and figures
            inperpolator (array): Linear interpolator for the spectral model grid
            scalar (array): Normalizing array for the model grid
            observed_wave (array): Observed wave data from "obspec"
            observed_flux (array): Observed flux data from "obspec"
            unc (array): Observed uncertainty data from "obspec"
            labels (list): List of model parameters, in order of the model file name
            rv_fit (boolean): When 'True' fits a radial velocity in km/s

        Returns:
            Best Fit Spectra (csv): CSV file of the best fit interpolated model spectrum
            Sampler (h5): A h5 file of the entire emcee sampler
            Spectral Comparison Plot (pdf): Comparison of the observed and best fit model spectrum with delta flux
            Stepping Plot (pdf): The stepping plot for each walker and model parameter
            Corner Plot (pdf): The corner plot for each model parameter
    '''
    
    # SAVES DATA TO H5PY FILE
    # ------------------------------------------------------------- #
    chain = sampler.get_chain()
    
    with h5py.File(f'{output_file}_sampler.h5', 'w') as f:
        f.create_dataset('chain', data=chain)
        f.create_dataset('log_prob', data=sampler.get_log_prob())
    # ------------------------------------------------------------- #

    # SETS UP PLOTTING DATA
    # ------------------------------------------------------------- #
    plt.rcParams['font.family'] = 'serif'  # Use a generic serif font

    chain_shape = sampler.chain.shape# Obtains shape of simulation sample
    chi_square, best_fit_params = mcbest(sampler, interpolator, scaler, observed_flux, unc, discard, labels, rv_fit) # Obtains best fit simulation parameters
    # ------------------------------------------------------------- #
    
    # PLOTS WALKERS
    # ------------------------------------------------------------- #
    samples = sampler.get_chain() # Obtains the readable version of simulation sample

    # Create subplots
    fig, axes = plt.subplots(chain_shape[2], figsize=(12, 8), sharex=True)

    # Plot each parameter
    if rv_fit:
        temp_labels = labels + ['Radial Velocity', 'Dilution Factor', 'Smoothing']
    else: 
        temp_labels = labels + ['Dilution Factor', 'Smoothing']
    for i in range(chain_shape[2]):
        ax = axes[i]
        for j in range(chain_shape[0]):
            ax.plot(samples[:, :, i].T[j], lw=1, color='k', alpha=0.5)
        ax.set_ylabel(f"{temp_labels[i]}")
        ax.set_xlim(0, samples.shape[0])

    # Label x-axis only for the last subplot
    axes[-1].set_xlabel("Step number")

    # Saves figure as a final step figure
    plt.tight_layout()
    plt.savefig(f'{output_file}_steps.pdf')
    plt.close('all')
    # ------------------------------------------------------------- #

    # PLOTS SPECTRA
    # ------------------------------------------------------------- #
    fig, (ax_chain, ax_diff) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]}, sharex=True) # Sets of figure dimesion and height ratios
    best_values = [sublist[len(sublist) // 2] for sublist in best_fit_params][:-1]
    
    # Interpolates best fit point
    if rv_fit: 
        point = np.array(np.array(best_values[:-2]).reshape(1, -1))
    else: 
        point = np.array(np.array(best_values[:-1]).reshape(1, -1))
    normalized_new_point = scaler.transform(point)
    interpolated_flux = interpolator(normalized_new_point)
    model_flux = gaussian_filter1d(interpolated_flux[0], best_fit_params[-1][1])
    
    # Scales interpolated spectrum
    observed_flux = observed_flux / (10**best_fit_params[-2][1])
    unc = unc / (10**best_fit_params[-2][1])

    chi_square = (np.nansum(np.square(np.divide(np.subtract(observed_flux, model_flux), unc))))/len(observed_flux)
    
    # Plots the observed and model spectrum
    ax_chain.plot(observed_wave, observed_flux, lw=2, c='k', label='Observed')
    ax_chain.plot(observed_wave, unc, color='silver', alpha=0.8, label='Uncertainties')
    ax_chain.plot(observed_wave, model_flux, lw=5, c='red', label='Best Fit Model', alpha=0.75) 

    # Calculates the difference between the model and observed and plots it
    flux_diff = np.subtract(observed_flux, model_flux)
    ax_diff.axhline(0, c='k', lw=2)
    ax_diff.plot(observed_wave, flux_diff, c='red', lw=3)
    
    # Defines the axis labels for both subplots
    ax_diff.set_xlabel(f"Wavelength ({unit_wave[0].to_string('latex')})", fontsize=25)
    ax_diff.set_ylabel('ΔFlux', fontsize=25)
    for ax in [ax_chain, ax_diff]:
        ax.minorticks_on(), ax.grid(True, alpha=0.3), ax.tick_params(which='minor', width=1, length=3, labelsize=10), ax.tick_params(which='major', width=2, length=6, labelsize=10)

    # Makes the grid and makes tick marks cleaner
    ax_chain.set_ylabel(f"Flux ({unit_flux[1].to_string('latex')})", fontsize=25)
    ax_chain.tick_params(axis='x', labelsize=12), ax_diff.tick_params(axis='x', labelsize=12)
    ax_chain.tick_params(axis='y', labelsize=12, labelrotation=45), ax_diff.tick_params(axis='y', labelsize=12, labelrotation=45)
    ax_chain.legend(prop={'size': 20})

    # Saves spectral comparison plot as a temporary figure
    ax_chain.set_title(f'χ² = {round(chi_square, 2)}', fontsize = 40, pad=10)
    plt.subplots_adjust(hspace=0)
    plt.savefig(f'{output_file}_fit.pdf')
    plt.close('all')
    # ------------------------------------------------------------- #
    
    # SAVE Best Fit Model
    # ------------------------------------------------------------- #
    # Save data to CSV
    csv_filename = f"{output_file}_spec.csv"
    data = np.column_stack((observed_wave, model_flux))
    np.savetxt(csv_filename, data, delimiter=",", header="observed_wave,model_flux", comments="")

    # Alternatively, using pandas:
    df = pd.DataFrame({"wavelength": observed_wave, "model_flux": model_flux})
    df.to_csv(csv_filename, index=False)
    # ------------------------------------------------------------- #
    
    # PLOTS CORNER PLOT
    # ------------------------------------------------------------- #
    plt.figure(figsize=(10, 10)) # Sets figure size for the corner plot
    flat_samples = sampler.get_chain(discard=int(discard), flat=True) # Obtains readable simulation sample for corner plot

    # Create title with LaTeX formatting
    def calculate_errors_and_limits(best_fit_params):
        titles = []
        for i, params in enumerate(best_fit_params):
            titles.append(r'${:.2f}^{{+{:.2f}}}_{{-{:.2f}}}$'.format(params[1], round(params[2] - params[1], 2), round(params[1] - params[0], 2)))
        return titles

    # Flatten the array of best fit parameters
    if rv_fit:
        labels = labels + [r'$v_{r}$', r"$\log_{10} \left( \frac{R^2}{D^2} \right)$"]
    else: 
        labels = labels + [r"$\log_{10} \left( \frac{R^2}{D^2} \right)$"]
    filtered_truths = [best_fit_params[i][1] for i in range(len(labels))]
    flat_samples = flat_samples[:, :-1]
    
    # Corner plot with needed values
    corner_fig = corner.corner(flat_samples, labels=labels, truths=filtered_truths, truth_color='k', title_fmt='.2f', title_kwargs={'fontsize': 10}, 
                               plot_contours=True, label_kwargs={'fontsize': 20}, quantiles=[0.05, 0.5, 0.95], use_math_text=True, color = 'red')
    
    # Calls the title function and finds the location each title is needed
    titles = calculate_errors_and_limits(best_fit_params)
    index_list = [(p*(len(labels)) + p) for p in range(len(labels))]
    index_list[0] = 0 
    
    # Puts 
    for i in range(len(index_list)):
        ax = corner_fig.axes[index_list[i]]
        ax.set_title(titles[i], fontsize=18)
        
    # Saves the corner plot as a temporary figure    
    plt.subplots_adjust(left=0.125, right=0.875, bottom=0.125, top=0.875)
    plt.savefig(f'{output_file}_corner.pdf')
    plt.close('all')
    
    return chi_square, best_fit_params
    # ------------------------------------------------------------- #
# --------------------------------- #    