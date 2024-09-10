# ------ Import Monte-Carlo Marchov-Chain Packages ------ # 
import emcee
# ------------------------------------------------------- # 

# ------ Import File Processing Packages ------ #
from pathos.multiprocessing import ProcessingPool as Pool
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import glob
import os
import re
# --------------------------------------------- # 

# ------ Import File Loading Packages ------ #
from astropy.table import Table
from astropy.io import fits
import pandas as pd
# ------------------------------------------ #

# ------ Import Plotting Packages ------ #
import matplotlib.pyplot as plt
import corner
# -------------------------------------- #

# ------ Import Math Packages ------ #
from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from scipy.spatial import Delaunay
import numpy as np
import math
# ---------------------------------- #

def mcmcfit(input_file, output_file, model_directory, model_parm, model_bounds, inter_type, unit, steps, walkers, stretch, discard): 
    '''
    Runs Spectral Fitting Markov-Chain Monte-Carlo Simulations Based on Inputted Model Spectra

        Parameters:
            input_file (str): Input .csv file with observed spectrum
            output_file (str): Out file name that is used when saving tables and figures
            model_directory (str): Model directory path
            model_parm (list): List of model parameters, in order of the model file name
            model_bounds (tuple): Bounds for each model parameter
            unit (float): Unit converter, observed spectrum wavelength unit used as standard
            inter_type (str): Type of interpolation grid, nearest or total
            steps (int): Number of steps the MCMC walkers take
            walkers (int): Number of MCMC walkers
            stretch (int): EMCEE stretch factor
            discard (int): Number of MCMC discard after running

        Returns:
            Tables (csv): Saves best fit model spectrum and every walker value for each parameter
            Figures (pdf): Save value over step and combined figure with best fit model compared to observed and corner plot
            Best Fit Parameters (list): Best fit value for each parameter and the associated uncertainties
    '''
    
    # ------ Checks each input parameter for correct format type ------ #
    if not (isinstance(input_file, str) and input_file.endswith('.csv')): # Checks Input Variable Type
        return ValueError('Input a Correct Input File Path (Ensure That it is a .csv File)')
    
    if not (isinstance(output_file, str)): # Checks Output Variable Type
        return ValueError('Input a Correct Output File Name')
    
    if not (isinstance(model_directory, str)): # Checks Model Directory Variable Type
        return ValueError('Input a Correct Model File Directory Path')
    
    if not (isinstance(model_parm, list)): # Checks Model Parameter Variable Type
        return ValueError('Input Correct Model Parameters')
    
    if not (isinstance(model_bounds, list)): # Checks Model Parameters Bounds Variable Type
        return ValueError('Input Correct Model Parameter Bounds')
    
    if len(model_parm) != len(model_bounds): # Checks that Model Parameters and Bounds are the Same Length
        return ValueError('Ensure the Number of Input Parameters and Input Parameter Bounds Are the Same')
    
    if not (isinstance(unit, int)): 
        if not (isinstance(unit, float)): # Checks Unit Variable Type
            return ValueError('Input a Correct Unit Conversion (Units of Observed Spectrum Wavelength Units)')
    
    if not (isinstance(steps, int)): # Checks MCMC Steps Variable Type
        return ValueError('Input Correct Number of Steps (Must be a Positive Integer)')
    
    if not (isinstance(walkers, int)): # Checks MCMC Walker Variable Type
        return ValueError('Input Correct Number of Walkers (Must be a Positive Integer)')
    
    if not (isinstance(stretch, int)): # Checks EMCEE Stretch Variable Type
        return ValueError('Input Correct Emcee Stetch Parameter (Must be a Postive Integer)')
    
    if not (isinstance(discard, int)): # Checks MCMC Discard Variable Type
        return ValueError('Input Correct Number of Discard Steps (Must be a Postive Integer)')
    # ----------------------------------------------------------------- #
    
    # ------ Load in the observed models and spectra ------ #
    wave,flux = obspec(input_file)
    grid = gridspec(model_directory, model_parm, wave, unit)
    # ----------------------------------------------------- #
    
    # ------ Build interpolation grid and start mcmc calculation ------ #
    inter = gridinter(grid, model_parm, inter_type)
    sampler = specmc(steps, walkers, inter, flux, stretch, model_bounds)
    # ----------------------------------------------------------------- #
    
    # ------ Provides user with mcmc best fit model ------ #
    best = mcbest(sampler, discard)
    mctable(sampler, inter, output_file, wave, model_parm, discard)
    mcplot(sampler, output_file, discard, inter, wave, flux, model_parm)
    # ---------------------------------------------------- #
    
    return best

def obspec(input_file): 
    # Read in observed spectrum
    observed = pd.read_csv(input_file)
    
    # Load in the wavelength and flux
    observed_wave = observed.iloc[:, 0].tolist()
    observed_flux = observed.iloc[:, 1].tolist()
    
    # Normalize the flux and remove nan values
    observed_flux = [x / np.nanpercentile(observed_flux, 99) for x in observed_flux]
    
    # Filter both lists using the indices to keep
    indices_to_keep = [i for i, flux in enumerate(observed_flux) if not np.isnan(flux) and flux > 0]
    observed_wave = [observed_wave[i] for i in indices_to_keep]
    observed_flux = [observed_flux[i] for i in indices_to_keep]

    return observed_wave, observed_flux
     
def gridspec(model_directory, model_parameters, observed_wave, unit_convert): 
    # Obtain all model file path directories
    model_files = glob.glob(f'{model_directory}/*')
    num_files = len(model_files)

    # Build a temporary dictionary to load in model spectra
    model_type = os.path.splitext(model_files[0])[1]
    total_data = {'file_path': [], 'file_data': []}

    # Function to load in csv, ascii, and fits model tables
    def process_file(file_path):
        if model_type == ".csv": # Loads csv files
            return file_path, pd.read_csv(file_path)
        elif model_type == ".txt": # Loads ascii files
            return file_path, pd.read_csv(file_path, delim_whitespace=True)
        elif model_type == ".fits": # Loads fits files
            with fits.open(file_path) as hdul:
                data = Table(hdul[1].data).to_pandas()
            return file_path, data

    # Multi-threads loading the model spectra
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_file, model_files), total=num_files))

    # Saves the data to the temporary model dictionary
    for file_path, file_data in results:
        total_data['file_path'].append(file_path)
        total_data['file_data'].append(file_data)
        
    # Build new, final, dictionary containing every parameter in the grid    
    total_grid = {'wavelength': [], 'flux': []}
    for parm in model_parameters:
        total_grid[parm] = []
    
    # Makes observed wavelengths an array for interpolation
    observed_wave = np.array(observed_wave)

    for j in tqdm(range(num_files)): 
        # Loads in each model wavelength and flux for each index and converts to array
        model_wave = np.array(total_data['file_data'][j].iloc[:, 0]) * unit_convert # Applies wavelength unit conversion here
        model_flux = np.array(total_data['file_data'][j].iloc[:, 1])

        # Resamples model spectrum to the resolution of the observed spectrum
        f_interp = interp1d(model_wave, model_flux, kind='linear', fill_value="extrapolate")
        resampled_flux = f_interp(observed_wave)
        resampled_flux /= np.nanmax(resampled_flux)  # Normalize by the max value

        resampled_flux = gaussian_filter1d(resampled_flux, 2)
        
        # Saves resampeld model spectrum to total grid
        total_grid['wavelength'].append(observed_wave)
        total_grid['flux'].append(resampled_flux)

        # Adds model parameter number to the final grid
        numbers = (re.findall(r'-?\d+\.?\d*', total_data['file_path'][j]))[1:]
        for p, number in enumerate(numbers): 
            if p < len(model_parameters):
                total_grid[model_parameters[p]].append(float(number))

    return total_grid
        
def gridinter(temp_grid, parm_list, inter_type):
    # Directly access the relevant keys in temp_grid to reduce overhead
    value_grid = np.array(temp_grid['flux'])

    # Access grid parameters directly without extra dict conversions
    grid_params = (np.vstack([np.array(temp_grid[key]) for key in parm_list])).T

    # Build the interpolator
    if inter_type == 'nearest': 
        interpolator = NearestNDInterpolator(grid_params, value_grid)
    elif inter_type == 'total': 
        interpolator = LinearNDInterpolator(grid_params, value_grid)
    
    return interpolator

def statmc(observed_flux, interpolator, parm):
    # Uses interpolation grid to interpolate model spectrum
    model_flux = interpolator(*parm)
    
    # Calculates chi square between observed flux and interpolated model flux
    chi_square = np.nansum(((observed_flux - model_flux)**2 / model_flux))
    stat = -0.5 * chi_square
    
    return stat

def specmc(steps, walkers, interpolator, observed_flux, stretch, parm_bound): 
    # Function seeing if the walkers parameters are withing the set bounds
    def prior(parm):
        bound_good_bad = []
        for i in range(len(parm_bound)):
            # Determines if the set parameter is within the bounds
            if (parm_bound[i][0] < parm[i] < parm_bound[i][1]):
                bound_good_bad.append(0) # 0 indicates within bounds
            else: 
                bound_good_bad.append(1) # 1 indicates outside of bounds
        if 1 in bound_good_bad:
            return -np.inf # If any value is outside the bounds then return bad value
        else: 
            return 0 # If all values are within the bounds then return good value

    # Calls statistics function and calls prior function
    def log_posterior(parm):
        lp = prior(parm)
        # If parameters are outside of bounds return bad statistics
        if not np.isfinite(lp):
            return -np.inf
        # Calls statistics function to compare model interpolated spectrum to observed spectrum
        return lp + statmc(observed_flux, interpolator, parm)

    # Sets lists of lower and upper bound set by the user
    low =  [bound[0] for bound in parm_bound]
    high = [bound[1] for bound in parm_bound]   

    # Sets the intial position of the walkers on a random position    
    initial_positions = np.random.uniform(low=low, high=high, size=(walkers, len(parm_bound)))   

    # Starts running the EMCEE Markov-Chain Monte-Carlo simulations
    sampler = emcee.EnsembleSampler(walkers, len(parm_bound), log_posterior, a=stretch)
    sampler.run_mcmc(initial_positions, steps, progress=True)
    # ------------------------------------------------------------ #

    
    return sampler

def mcbest(sampler, discard): 
    # Obtains the shape of the Markov-Chain Monte-Carlo sample
    chain_shape = sampler.chain.shape
    
    # Flattens the Markov-Chain Monte-Carlo sample
    flat_samples = sampler.get_chain(discard=discard, thin=15, flat=True)
    best_fit_params = []
    # Finds the best fit parameters with their associated uncertainties
    for i in range(chain_shape[2]):
        mcmc = np.percentile(flat_samples[:, i], [5, 50, 95])
        best_fit_params.append([mcmc[0], mcmc[1], mcmc[2]])
    
    return best_fit_params

def mcplot(sampler, output_file, discard, interpolator, observed_wave, observed_flux, labels): 
    # SETS UP PLOTTING DATA
    # -------------------------------------------------------------
    # plt.rcParams['font.family'] = 'Times New Roman' # Makes all figure text in Times New Roman
    plt.rcParams['font.family'] = 'serif'  # Use a generic serif font

    chain_shape = sampler.chain.shape# Obtains shape of simulation sample
    best_fit_params = mcbest(sampler, discard) # Obtains best fit simulation parameters
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
    best_values = [sublist[len(sublist) // 2] for sublist in best_fit_params]
    model_flux = gaussian_filter1d((interpolator(best_values)[0]), 1.4) # Finds the best fit interpolated spectral model and smooths it by a factor of 1.4
    
    # Plots the observed and model spectrum
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
    ax_chain.set_ylabel('Flux log$_{10}$(erg/s/cm$^{2}$/Å)', fontsize=25)
    ax_chain.tick_params(axis='x', labelsize=12), ax_diff.tick_params(axis='x', labelsize=12)
    ax_chain.tick_params(axis='y', labelsize=12, labelrotation=45), ax_diff.tick_params(axis='y', labelsize=12, labelrotation=45)
    ax_chain.legend(prop={'size': 20})
    ax_chain.set_ylim(-0.1, 1.1)

    # Saves spectral comparison plot as a temporary figure
    plt.subplots_adjust(hspace=0)
    plt.savefig(f'{output_file}_figure1.png', dpi=200)
    plt.close('all')
    # -------------------------------------------------------------
    
    # PLOTS CORNER PLOT
    # --------------------------------------------------------------------------------------------------------------------------------------------------- #
    plt.figure(figsize=(5, 5)) # Sets figure size for the corner plot
    flat_samples = sampler.get_chain(discard=discard, thin=15, flat=True) # Obtains readable simulation sample for corner plot
    
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
            titles.append(r'${:.2f}^{{+{:.2f}}}_{{-{:.2f}}}$'.format(params[1], upper_error, lower_error))
        
        return upper_errors, lower_errors, limits, titles

    # Flatten the array of best fit parameters
    filtered_truths = [best_fit_params[i][1] for i in range(len(labels))]
    
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

def mctable(sampler, interpolator, output_file, observed_wave, labels, discard): 
    # Gets the shape of the Markov-Chain Monte-Carlo sample
    chain_shape = sampler.chain.shape
    
    # Flattens the Markov-Chain Monte-Carlo sample to a machine readable format
    samples = sampler.get_chain()
    
    # Saves each parameter value from each walkers step into a csv table
    for i in range(chain_shape[2]):
        flattened_array = samples[:, :, i].reshape((chain_shape[1], chain_shape[0]))
        df = pd.DataFrame(flattened_array, columns=[f'WALKER_{j+1}' for j in range(chain_shape[0])])
        df.to_csv(f'{output_file}_parm_{labels[i]}.csv', index=False)

    # Calculates the best fit parameters from the simulation and interpolates the model spectrum
    best_fit_params = mcbest(sampler, discard)
    best_values = [sublist[len(sublist) // 2] for sublist in best_fit_params]
    model_flux = (interpolator(best_values))[0]
    
    # Saves best fit model spectrum to a csv table
    df = pd.DataFrame({'WAVELENGTH': observed_wave, 'FLUX': model_flux})
    df.to_csv(f'{output_file}_best_fit.csv', index=False)
    