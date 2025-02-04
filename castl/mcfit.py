# ------ Import Monte-Carlo Marchov-Chain Packages ------ # 
import emcee
# ------------------------------------------------------- # 

# ------ Import File Processing Packages ------ #
from pathos.multiprocessing import ProcessingPool as Pool
from concurrent.futures import ThreadPoolExecutor
from IPython.display import clear_output
from tqdm import tqdm
import glob
import h5py
import os
import re
# --------------------------------------------- # 

# ------ Import File Loading Packages ------ #
from astropy.table import Table
from netCDF4 import Dataset
from astropy.io import fits
import pandas as pd
# ------------------------------------------ #

# ------ Import Plotting Packages ------ #
import matplotlib.pyplot as plt
import corner
# -------------------------------------- #

# ------ Import Math Packages ------ #
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import RBFInterpolator
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from emcee.moves import StretchMove
import numpy as np
import math
# ---------------------------------- #

# ------ Ignore all warnings ------ #
import warnings
warnings.filterwarnings('ignore')
# --------------------------------- #

def mcmcfit(input_file, output_file, model_directory, model_parm, unit=1, walkers=15, max_step=1000, safety_coeff=10, stretch_factor=2, monitor = False): 
    '''
    Runs Spectral Fitting Markov-Chain Monte-Carlo Simulations Based on Inputted Model Spectra

        Parameters:
            input_file (str): Input .csv file with observed spectrum
            output_file (str): Out file name that is used when saving tables and figures
            model_directory (str): Model directory path
            model_parm (list): List of model parameters, in order of the model file name
            unit (float): Unit converter, observed spectrum wavelength unit used as standard
            walkers (int): Number of MCMC walkers
            max_steps (int): Max Number of Steps
            safety_coeff (float): The Multiplication Factor after tau is Reached
            stretch_factor (float): The emcee Stretch Factor
            monitor (boolean): When 'True' step monitors are activated

        Returns:
            Tables (csv): Saves best fit model spectrum and every walker value for each parameter
            Figures (pdf): Save value over step and combined figure with best fit model compared to observed and corner plot
            Best Fit Parameters (list): Best fit value for each parameter and the associated uncertainties
    '''
    
    # ------ Load in the observed models and spectra ------ #
    wave, flux, unc = obspec(input_file)
    # ----------------------------------------------------- #
    
    # ------ Build interpolation grid and start mcmc calculation ------ #
    inter, scaler, grid = gridinter(model_parm, model_directory, wave, unit)
    sampler, discard = specmc(wave, inter, scaler, flux, unc, grid, walkers, max_step, stretch_factor, monitor)
    # ----------------------------------------------------------------- #
    
    # ------ Provides user with mcmc best fit model ------ #
    mcplot(sampler, discard, output_file, inter, scaler, wave, flux, unc, model_parm)
    # ---------------------------------------------------- #
    return

def obspec(input_file): 
    # Read in observed spectrum
    file_ext = input_file.split('.')[-1].lower()  # Get file extension

    if file_ext == "csv":  # CSV files
        observed = pd.read_csv(input_file)
        
    elif file_ext == "vot":  # VOTable files
        votable = votable.parse(input_file) 
        observed = votable.get_table() 
    
    elif file_ext == "nc":  # NetCDF files
        with Dataset(input_file, 'r') as nc_file:
            variables = list(nc_file.variables.keys())
            observed = nc_file.variables[variables[0]][:]

    elif file_ext in ["txt", "dat", "tbl"]:  # ASCII tables with whitespace or tab delimiters
        observed = pd.read_csv(input_file, delim_whitespace=True)
        
    elif file_ext in ["tsv"]:  # Loads Tab-Separated files
        observed = pd.read_csv(file_path, sep="\t")

    elif file_ext == "fits":  # FITS files
        with fits.open(input_file) as hdul:
            data = Table(hdul[1].data).to_pandas()
        observed = data
        
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    # Load in the wavelength and flux
    observed_wave = observed.iloc[:, 0].tolist()
    observed_flux = observed.iloc[:, 1].tolist()
    observed_unc = observed.iloc[:, 2].tolist()
    
    # Normalize the flux and remove nan values
    observed_unc = [x / np.nanpercentile(observed_flux, 99) for x in observed_unc]
    observed_flux = [x / np.nanpercentile(observed_flux, 99) for x in observed_flux]
    
    # Filter both lists using the indices to keep
    indices_to_keep = [i for i, flux in enumerate(observed_flux) if not np.isnan(flux) and flux > 0]
    observed_wave = ([observed_wave[i] for i in indices_to_keep])
    observed_flux = ([observed_flux[i] for i in indices_to_keep])
    observed_unc = ([observed_unc[i] for i in indices_to_keep])

    return observed_wave, observed_flux, observed_unc

def gridinter(parm_list, model_directory, observed_wave, unit):
    # Pre-load the file and store all data in memory
    with h5py.File(f"{model_directory}", "r") as h5f:
        original_names = {key: h5f[key].attrs.get("original_name", key) for key in h5f.keys()}
        loaded_grid = {original_names[key]: h5f[key][()] for key in h5f.keys()}
         
    wavelength_data_list = loaded_grid['wavelength']  # List of wavelength lists
    flux_data_list = loaded_grid['flux']  # List of flux lists

    # Interpolation function and replacement
    new_flux_data_list = []
    for wavelength_data, flux_data in tqdm(zip(wavelength_data_list, flux_data_list), total=len(wavelength_data_list), desc="Resampling Flux: ", ncols=100, unit="step"):
        f_interp = interp1d(wavelength_data*unit, flux_data, kind='linear', fill_value="extrapolate")
        new_flux = f_interp(observed_wave)
        new_flux /= np.nanmax(new_flux)
        new_flux_data_list.append(new_flux)

    # Efficiently create grid parameters using np.column_stack
    grid_params = np.column_stack([loaded_grid[key] for key in parm_list])
    
    # Normalize parameters
    scaler = MinMaxScaler()
    normalized_grid_params = scaler.fit_transform(grid_params)

    # Build RBF interpolator with nearest neighbors
    N, D = grid_params.shape
    if D < 5:
        interpolator = LinearNDInterpolator(normalized_grid_params, new_flux_data_list)
    else:
        interpolator = RBFInterpolator(normalized_grid_params, new_flux_data_list)

    return interpolator, scaler, grid_params

def statmc(observed_wave, observed_flux, unc, interpolator, scaler, parm):
    filter_width = parm[-1]
    unc = np.array(unc)

    # Denormalize the input parameters using the provided scaler
    point = np.array(parm[:-1].reshape(1, -1))
    normalized_new_point = scaler.transform(point)
    
    # Reshape the denormalized parameters to a 2D array for the interpolator
    model_flux = interpolator(normalized_new_point)  # Reshape to (1, n_dimensions)

    # Apply Gaussian filter to the model flux
    resampled_flux = gaussian_filter1d(model_flux, filter_width)  # Access the first element after interpolation

    # Efficiently calculate chi-square between observed and model flux
    temp_stat = (np.nansum(np.square(np.divide(np.subtract(observed_flux, resampled_flux[0]), unc))))/len(observed_flux)
    stat = -0.5 * temp_stat
    
    return stat

def specmc(observed_wave, interpolator, scaler, observed_flux, unc, grid, walkers=15, max_step=10000, stretch_factor=2, monitor = False):
    # Function to check if the walkers' parameters are within the set bounds
    def prior(parm):
        if np.any(parm < np.array([low for low, high in parm_bound])) or np.any(parm > np.array([high for low, high in parm_bound])):
            return -np.inf
        return 0

    # Log-posterior function combining the prior and the likelihood
    def log_posterior(parm):
        lp = prior(parm)
        if not np.isfinite(lp):
            return -np.inf
        
        # Calls the statistic function
        stat = statmc(observed_wave, observed_flux, unc, interpolator, scaler, parm)
        
        # Checks if the statistic is valid
        if stat == 0:
            return -np.inf
        else:
            return lp + stat

    def get_min_max_ranges(grid):
        min_max_list = []
        
        for i in range(len(grid[0, :])):
            values = grid[:, i]
            tlow = min(values)
            thigh = max(values)
            min_max_list.append((tlow, thigh))
        
        return min_max_list

    parm_bound = get_min_max_ranges(grid) + [(1, 1.5)]
    n_params = len(parm_bound)  # Number of parameters
    initial_positions = np.zeros((walkers, n_params))  # Initialize positions array
    
    for i, (low, high) in enumerate(parm_bound):
        initial_positions[:, i] = np.random.uniform(low=low + 0.1*low, high=high - 0.1*high, size=walkers)
    
    move = StretchMove(a=stretch_factor)

    # Makes the sampler ready to go
    sampler = emcee.EnsembleSampler(walkers, len(parm_bound), log_posterior, moves=move)

    best_discard = 0
    for sample in tqdm(sampler.sample(initial_positions, iterations=max_step),  total=max_step, desc="Starting MCMC: ", ncols=100, unit="step"):
        if sampler.iteration == max_step:
            best_discard = max_step*0.25
            break
        if sampler.iteration % 1000:
            continue

        # Plots stepping plot for monitoring
        if monitor == True: 
            plt.clf()
            plt.close('all')
            clear_output(wait=True)
            print('<------ Monitor MCMC Calculation ------>')
            n_params = sampler.chain.shape[-1]
            fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 4))
            for i, ax in enumerate(axes):
                for walker in sampler.chain[..., i]:
                    ax.plot(walker, alpha=0.5)
                ax.set_title(f"Param: {i}")
                ax.set_xlabel("Step")
                if i == 0: 
                    ax.set_ylabel("Value")

            plt.axvline(best_discard, c = 'k')
            plt.tight_layout()
            plt.show()   
    
    clear_output(wait=True)
    return sampler, best_discard

def mcbest(sampler, discard, labels): 
    # Obtains the shape of the Markov-Chain Monte-Carlo sample
    chain_shape = sampler.chain.shape
    
    # Flattens the Markov-Chain Monte-Carlo sample
    flat_samples = sampler.get_chain(discard=int(discard), flat=True)
    best_fit_params = []
    
    # Finds the best fit parameters with their associated uncertainties
    print('<------ Best Fit Parameters ------>')
    print(f'Number of Discards: {int(discard)}')
    for i in range(chain_shape[2]):
        mcmc = np.percentile(flat_samples[:, i], [5, 50, 95])
        best_fit_params.append([mcmc[0], mcmc[1], mcmc[2]])
        
        if i != (chain_shape[2] - 1):
            print(f'{labels[i]}: {mcmc[1]:.2f} \n (Upper: {np.abs(mcmc[1] - mcmc[2]):.2f}, Lower: {np.abs(mcmc[0] - mcmc[1]):.2f})')
    
    return best_fit_params

def mcplot(sampler, discard, output_file, interpolator, scaler, observed_wave, observed_flux, unc, labels): 
    # SAVES DATA TO H5PY FILE
    # -------------------------------------------------------------
    chain = sampler.get_chain()
    
    with h5py.File(f'{output_file}_sampler.h5', 'w') as f:
        f.create_dataset('chain', data=chain)
        f.create_dataset('log_prob', data=sampler.get_log_prob())
    # -------------------------------------------------------------

    # SETS UP PLOTTING DATA
    # -------------------------------------------------------------
    plt.rcParams['font.family'] = 'serif'  # Use a generic serif font

    chain_shape = sampler.chain.shape# Obtains shape of simulation sample
    best_fit_params = mcbest(sampler, discard, labels) # Obtains best fit simulation parameters
    # -------------------------------------------------------------
    
    # PLOTS WALKERS
    # --------------------------------------------------------------------------------------------------------------------------------------------------- #
    samples = sampler.get_chain() # Obtains the readable version of simulation sample

    # Create subplots
    fig, axes = plt.subplots(chain_shape[2], figsize=(12, 8), sharex=True)

    # Plot each parameter
    for i in range(chain_shape[2]):
        ax = axes[i]
        for j in range(chain_shape[0]):
            ax.plot(samples[:, :, i].T[j], lw=1, color='k', alpha=0.5)
        ax.set_ylabel(f"Parameter {i+1}")
        ax.set_xlim(0, samples.shape[0])

    # Label x-axis only for the last subplot
    axes[-1].set_xlabel("Step number")

    # Saves figure as a final step figure
    plt.tight_layout()
    plt.savefig(f'{output_file}_steps.pdf')
    plt.close('all')
    # --------------------------------------------------------------------------------------------------------------------------------------------------- #

    # PLOTS SPECTRA
    # -------------------------------------------------------------
    fig, (ax_chain, ax_diff) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]}) # Sets of figure dimesion and height ratios
    best_values = [sublist[len(sublist) // 2] for sublist in best_fit_params][:-1]
    
    point = np.array(np.array(best_values).reshape(1, -1))
    normalized_new_point = scaler.transform(point)
    interpolated_flux = interpolator(normalized_new_point)
    model_flux = gaussian_filter1d(interpolated_flux[0], best_fit_params[-1][1])

    chi_square = round(np.nansum(np.square(np.divide(np.subtract(np.array(observed_flux), np.array(model_flux)), np.array(unc)))), 2)
    
    # Plots the observed and model spectrum
    ax_chain.plot(observed_wave, unc, color='silver', alpha=0.8)
    ax_chain.plot(observed_wave, observed_flux, lw=2, c='k', label='Observed')
    ax_chain.plot(observed_wave, model_flux, lw=5, c='fuchsia', label='Best Fit Model', alpha=0.6)

    # Calculates the difference between the model and observed and plots it
    flux_diff = np.subtract(observed_flux, model_flux)
    ax_diff.axhline(0, c='k', lw=2)
    ax_diff.plot(observed_wave, flux_diff, c='fuchsia')
    
    # Defines the axis labels for both subplots
    ax_diff.set_xlabel('Wavelength (µm)', fontsize=25)
    ax_diff.set_ylabel('ΔFlux', fontsize=25)
    for ax in [ax_chain, ax_diff]:
        ax.minorticks_on(), ax.grid(True, alpha=0.3), ax.tick_params(which='minor', width=1, length=3, labelsize=10), ax.tick_params(which='major', width=2, length=6, labelsize=10)

    # Makes the grid and makes tick marks cleaner
    ax_chain.set_ylabel('Normalized Flux \n log$_{10}$(erg/s/cm$^{2}$/Å)', fontsize=25)
    ax_chain.tick_params(axis='x', labelsize=12), ax_diff.tick_params(axis='x', labelsize=12)
    ax_chain.tick_params(axis='y', labelsize=12, labelrotation=45), ax_diff.tick_params(axis='y', labelsize=12, labelrotation=45)
    ax_chain.legend(prop={'size': 20})
    ax_chain.set_ylim(-0.1, 1.1)

    # Saves spectral comparison plot as a temporary figure
    ax_chain.set_title(f'χ² = {round(chi_square/len(observed_wave), 4)}', fontsize = 40, pad=10)
    plt.subplots_adjust(hspace=0)
    plt.savefig(f'{output_file}_figure1.png', dpi=200)
    plt.close('all')
    # -------------------------------------------------------------
    
    # PLOTS CORNER PLOT
    # --------------------------------------------------------------------------------------------------------------------------------------------------- #
    plt.figure(figsize=(5, 5)) # Sets figure size for the corner plot
    flat_samples = sampler.get_chain(discard=int(discard), flat=True) # Obtains readable simulation sample for corner plot
    
    # Makes empty lists for best fit uncertainties
    upper_errors = []
    lower_errors = []
    limits = []
    titles = []

    # Function to append the empty lists to fill with needed titles
    def calculate_errors_and_limits(best_fit_params):
        # Makes empty lists for best fit uncertainties
        upper_errors = []
        lower_errors = []
        limits = []
        titles = []

        for i, params in enumerate(best_fit_params):
            # Calculate upper and lower errors
            upper_error = round(params[2] - params[1], 2)
            lower_error = round(params[1] - params[0], 2)
            
            # Append errors
            upper_errors.append(upper_error)
            lower_errors.append(lower_error)
            
            # Calculate limits
            limits.append((params[0] - (lower_error / 2), params[2] + (upper_error / 2)))
            
            # Create title with LaTeX formatting
            titles.append(r'${:.3f}^{{+{:.3f}}}_{{-{:.3f}}}$'.format(params[1], upper_error, lower_error))
        
        return upper_errors, lower_errors, limits, titles

    # Flatten the array of best fit parameters
    filtered_truths = [best_fit_params[i][1] for i in range(len(labels))]
    flat_samples = flat_samples[:, :-1]
    
    # Corner plot with needed values
    corner_fig = corner.corner(flat_samples, labels=labels, truths=filtered_truths, truth_color='fuchsia', title_fmt='.2f', title_kwargs={'fontsize': 25}, plot_contours=True, label_kwargs={'fontsize': 25}, quantiles=[0.05, 0.5, 0.95], use_math_text=True)
    
    # Calls the title function and finds the location each title is needed
    upper_errors, lower_errors, limits, titles = calculate_errors_and_limits(best_fit_params)
    index_list = [(p*(len(labels)) + p) for p in range(len(labels))]
    index_list[0] = 0 
    
    # Puts 
    for i in range(len(index_list)):
        ax = corner_fig.axes[index_list[i]]
        ax.set_title(titles[i], fontsize=18)
        
    # Saves the corner plot as a temporary figure    
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.1, hspace=0.1)    
    plt.savefig(f'{output_file}_figure2.png', dpi = 200)
    plt.close('all')
    # --------------------------------------------------------------------------------------------------------------------------------------------------- #
    
    # --------------------------------------------------------------------------------------------------------------------------------------------------- #
    # Combine figures
    fig, axs = plt.subplots(1, 2, figsize=(15, 8), gridspec_kw={'width_ratios': [1.25, 1]})

    # Display scatter and corner plot images
    scatter_img = plt.imread(f"{output_file}_figure1.png")
    axs[0].imshow(scatter_img)
    axs[0].axis('off')
    os.remove(f"{output_file}_figure1.png")
    corner_img = plt.imread(f"{output_file}_figure2.png")
    axs[1].imshow(corner_img)
    axs[1].axis('off')
    os.remove(f"{output_file}_figure2.png")

    # Saves final combines spectral and corner plot
    plt.subplots_adjust(wspace=-0.5)
    plt.tight_layout()
    plt.savefig(f'{output_file}.pdf', dpi=500)
    plt.close('all')
    # --------------------------------------------------------------------------------------------------------------------------------------------------- #
    