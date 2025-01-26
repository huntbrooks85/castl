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
    
    print('<------ castl Calculation Started ------>')
    
    # ------ Load in the observed models and spectra ------ #
    wave,flux,unc = obspec(input_file)
    grid = gridspec(model_directory, model_parm, wave, unit)
    # ----------------------------------------------------- #
    
    # ------ Build interpolation grid and start mcmc calculation ------ #
    inter, scaler = gridinter(grid, model_parm)
    sampler, discard = specmc(inter, scaler, wave, flux, unc, grid, model_parm, walkers, max_step, safety_coeff, stretch_factor, monitor)
    # ----------------------------------------------------------------- #
    
    # ------ Provides user with mcmc best fit model ------ #
    mcplot(sampler, discard, output_file, inter, scaler, wave, flux, unc, model_parm)
    # ---------------------------------------------------- #
    return

def obspec(input_file): 
    # Read in observed spectrum
    observed = pd.read_csv(input_file)
    
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
     
def gridspec(model_directory, model_parameters, observed_wave, unit_convert=1): 
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
    print('<------ Loading Model Spectra ------>')
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

    print('\n<------ Building Model Grid ------>')
    for j in tqdm(range(num_files)): 
        # Loads in each model wavelength and flux for each index and converts to array
        model_wave = (np.array(total_data['file_data'][j].iloc[:, 0]) * unit_convert) # Applies wavelength unit conversion here
        model_flux = (np.array(total_data['file_data'][j].iloc[:, 1]))

        # Resamples model spectrum to the resolution of the observed spectrum
        f_interp = interp1d(model_wave, model_flux, kind='linear', fill_value="extrapolate")
        resampled_flux = f_interp(observed_wave)
        resampled_flux /= np.nanmax(resampled_flux)  # Normalize by the max value
        
        # Saves resampeld model spectrum to total grid
        total_grid['wavelength'].append(observed_wave)
        total_grid['flux'].append(resampled_flux)

        # Adds model parameter number to the final grid
        numbers = (re.findall(r'-?\d+\.?\d*', total_data['file_path'][j].split('/', 100000)[-1]))[0:]
        for p, number in enumerate(numbers): 
            if p < len(model_parameters):
                total_grid[model_parameters[p]].append((float(number)))
    return total_grid

def gridinter(temp_grid, parm_list):
    # Ensure flux and parameters are NumPy arrays
    grid_params = np.array([temp_grid[key] for key in parm_list]).T
    value_grid = np.asarray(temp_grid['flux'])
    
    scaler_params = MinMaxScaler()
    normalized_grid_params = scaler_params.fit_transform(grid_params)
    
    # Create the RBF interpolator
    print('\n<------ Building Interpolation Grid ------>')
    interpolator = RBFInterpolator(normalized_grid_params, value_grid)
    
    return interpolator, scaler_params

def statmc(wave, observed_flux, unc, interpolator, scaler, parm):
    filter_width = parm[-1]
    unc = np.array(unc)
    
    # Denormalize the input parameters using the provided scaler
    point = np.array(parm[:-1].reshape(1, -1))
    normalized_new_point = scaler.transform(point)
    
    # Reshape the denormalized parameters to a 2D array for the interpolator
    model_flux = interpolator(normalized_new_point)  # Reshape to (1, n_dimensions)

    # Apply Gaussian filter to the model flux
    resampled_flux = gaussian_filter1d(model_flux[0], filter_width)  # Access the first element after interpolation

    # Efficiently calculate chi-square between observed and model flux
    temp_stat = (np.nansum(np.square(np.divide(np.subtract(observed_flux, resampled_flux), unc))))/len(observed_flux)
    stat = -0.5 * temp_stat

    return stat

def specmc(interpolator, scaler, wave, observed_flux, unc, grid, model_parm, walkers=15, max_step=10000, safety_coeff=10, stretch_factor=2, monitor = False):
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
        stat = statmc(wave, observed_flux, unc, interpolator, scaler, parm)
        
        # Checks if the statistic is valid
        if stat == 0:
            return -np.inf
        else:
            return lp + stat

    def get_min_max_ranges(grid):
        min_max_list = []
        keys = list(grid.keys())  # Get a list of keys
        
        # Start from the third key (index 2)
        for key in keys[2:]:
            values = grid[key]
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
    print('\n<------ Started MCMC Calculation ------>')
    sampler = emcee.EnsembleSampler(walkers, len(parm_bound), log_posterior, moves=move)

    index = 0
    autocorr = np.empty(max_step)
    old_tau = np.inf
    best_discard = 0
    for sample in sampler.sample(initial_positions, iterations=max_step, progress=True):
        # Only check convergence every 100 steps
        if sampler.iteration == max_step:
            if best_discard == 0: 
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
            
        # Compute the autocorrelation time so far
        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        index += 1
    
        # Check convergence
        if best_discard == 0: 
            if np.all(np.abs(old_tau - tau) / tau < 0.025):
                best_discard = sampler.iteration
            
        converged = np.all(tau * safety_coeff < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.025)
        if converged:
            break
        old_tau = tau
    
    clear_output(wait=True)
    return sampler, best_discard

def mcbest(sampler, discard): 
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
        
        print(f'Parameter {i}: {mcmc[1]:.2f} \n (Upper: {np.abs(mcmc[1] - mcmc[2]):.2f}, Lower: {np.abs(mcmc[0] - mcmc[1]):.2f})')
    
    return best_fit_params

def mcplot(sampler, discard, output_file, interpolator, scaler, observed_wave, observed_flux, unc, labels): 
    # SAVES DATA TO H5PY FILE
    # -------------------------------------------------------------
    chain = sampler.get_chain()
    
    with h5py.File(f'{output_file}.h5', 'w') as f:
        f.create_dataset('chain', data=chain)
        f.create_dataset('log_prob', data=sampler.get_log_prob())
    # -------------------------------------------------------------

    # SETS UP PLOTTING DATA
    # -------------------------------------------------------------
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
    