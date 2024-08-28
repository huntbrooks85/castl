from pathos.multiprocessing import ProcessingPool as Pool
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import corner
import emcee
import os

def statmc(observed_flux, interpolator, parm, scaling):
    model_flux = interpolator(*parm)
    
    squared_differences = np.square(model_flux - observed_flux)
    mean_squared_difference = np.nanmean(squared_differences) 
    rmse = np.sqrt(mean_squared_difference)
    
    return -scaling*rmse

def specmc(steps, walkers, interpolator, observed_flux, scaling, stretch, parm_bound): 
    def prior(parm):
        bound_good_bad = []
        for i in range(len(parm_bound)):
            if (parm_bound[i][0] < parm[i] < parm_bound[i][1]):
                bound_good_bad.append(0)
            else: 
                bound_good_bad.append(1)
        if 1 in bound_good_bad:
            return -np.inf
        else: 
            return 0

    def log_posterior(parm):
        lp = prior(parm)
        if not np.isfinite(lp):
            return -np.inf
        return lp + statmc(observed_flux, interpolator, parm, scaling)

    low =  [bound[0] for bound in parm_bound]
    high = [bound[1] for bound in parm_bound]
        
    initial_positions = np.random.uniform(low=low, high=high, size=(walkers, len(parm_bound)))

    sampler = emcee.EnsembleSampler(walkers, len(parm_bound), log_posterior, a=stretch)
    sampler.run_mcmc(initial_positions, steps, progress=True)
    return sampler

def mcbest(sampler, discard): 
    chain_shape = sampler.chain.shape
    
    flat_samples = sampler.get_chain(discard=discard, thin=15, flat=True)
    best_fit_params = []
    for i in range(chain_shape[2]):
        mcmc = np.percentile(flat_samples[:, i], [5, 50, 95])
        best_fit_params.append([mcmc[0], mcmc[1], mcmc[2]])
    
    return best_fit_params

def mcplot(sampler, output_file, discard, interpolator, observed_wave, observed_flux, labels): 
    # SETS UP PLOTTING DATA
    # -------------------------------------------------------------
    plt.rcParams['font.family'] = 'Times New Roman'

    samples = sampler.get_chain()
    chain_shape = sampler.chain.shape
    best_fit_params = mcbest(sampler, discard)
    flat_samples = sampler.get_chain(discard=discard, thin=15, flat=True)
    # -------------------------------------------------------------
    
    # PLOTS WALKERS
    # --------------------------------------------------------------------------------------------------------------------------------------------------- #
    samples = sampler.get_chain()

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

    plt.tight_layout()
    plt.savefig(f'{output_file}_steps.pdf')
    plt.close('all')
    # --------------------------------------------------------------------------------------------------------------------------------------------------- #

    # PLOTS SPECTRA
    # -------------------------------------------------------------
    fig, (ax_chain, ax_diff) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
    best_values = [sublist[len(sublist) // 2] for sublist in best_fit_params]
    model_flux = gaussian_filter1d((interpolator(best_values)[0]), 1.4)
    
    ax_chain.plot(observed_wave, observed_flux, lw=2, c='k', label='Observed')
    ax_chain.plot(observed_wave, model_flux, lw=5, c='fuchsia', label='Best Fit Model', alpha=0.6)

    flux_diff = np.subtract(observed_flux, model_flux)
    ax_diff.axhline(0, c='k', lw=2)
    ax_diff.plot(observed_wave, flux_diff, c='fuchsia')
    
    ax_diff.set_xlabel('Wavelength (µm)', fontsize=25)
    ax_diff.set_ylabel('ΔFlux', fontsize=25)
    for ax in [ax_chain, ax_diff]:
        ax.minorticks_on(), ax.grid(True, alpha=0.3), ax.tick_params(which='minor', width=1, length=3, labelsize=10), ax.tick_params(which='major', width=2, length=6, labelsize=10)

    ax_chain.set_ylabel('Flux log$_{10}$(erg/s/cm$^{2}$/Å)', fontsize=25)
    ax_chain.tick_params(axis='x', labelsize=12), ax_diff.tick_params(axis='x', labelsize=12)
    ax_chain.tick_params(axis='y', labelsize=12, labelrotation=45), ax_diff.tick_params(axis='y', labelsize=12, labelrotation=45)
    ax_chain.legend(prop={'size': 20})
    ax_chain.set_ylim(-0.1, 1.1)

    plt.subplots_adjust(hspace=0)
    plt.savefig('figure1.png', dpi=200)
    plt.close('all')
    # -------------------------------------------------------------
    
    # --------------------------------------------------------------------------------------------------------------------------------------------------- #
    plt.figure(figsize=(5, 5))
    upper_errors = []
    lower_errors = []
    limits = []
    titles = []

    def calculate_errors_and_limits(best_fit_params):
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
            titles.append(
                r'${:.2f}^{{+{:.2f}}}_{{-{:.2f}}}$'.format(params[1], upper_error, lower_error)
            )
        
        return upper_errors, lower_errors, limits, titles

    filtered_truths = [best_fit_params[i][1] for i in range(len(labels))]
    
    corner_fig = corner.corner(flat_samples, labels=labels, truths=filtered_truths, truth_color='fuchsia', title_fmt='.2f', title_kwargs={'fontsize': 25}, plot_contours=True, label_kwargs={'fontsize': 25}, quantiles=[0.05, 0.5, 0.95], use_math_text=True)
    
    upper_errors, lower_errors, limits, titles = calculate_errors_and_limits(best_fit_params)
    index_list = [(p*(len(labels)) + p) for p in range(len(labels))]
    index_list[0] = 0 
    for i in range(len(index_list)):
        ax = corner_fig.axes[index_list[i]]
        ax.set_title(titles[i], fontsize=18)
        
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.1, hspace=0.1)    
    plt.savefig('figure2.png', dpi = 200)
    plt.close('all')
    # --------------------------------------------------------------------------------------------------------------------------------------------------- #
    
    # --------------------------------------------------------------------------------------------------------------------------------------------------- #
    # Combine figures
    fig, axs = plt.subplots(1, 2, figsize=(15, 8), gridspec_kw={'width_ratios': [1.25, 1]})

    # Display scatter and corner plot images
    scatter_img = plt.imread("figure1.png")
    axs[0].imshow(scatter_img)
    axs[0].axis('off')
    os.remove("figure1.png")

    corner_img = plt.imread("figure2.png")
    axs[1].imshow(corner_img)
    axs[1].axis('off')
    os.remove("figure2.png")

    plt.subplots_adjust(wspace=-0.5)
    plt.tight_layout()
    plt.savefig(f'{output_file}.pdf', dpi=500)
    plt.close('all')
    # --------------------------------------------------------------------------------------------------------------------------------------------------- #

def mctable(sampler, interpolator, output_file, observed_wave, labels, discard): 
    chain_shape = sampler.chain.shape
    
    samples = sampler.get_chain()
    best_fit_params = mcbest(sampler, discard)

    for i in range(chain_shape[2]):
        flattened_array = samples[:, :, i].reshape((chain_shape[1], chain_shape[0]))
        df = pd.DataFrame(flattened_array, columns=[f'WALKER_{j+1}' for j in range(chain_shape[0])])
        df.to_csv(f'{output_file}_parm_{labels[i]}.csv', index=False)

    # Model flux calculation
    best_values = [sublist[len(sublist) // 2] for sublist in best_fit_params]
    model_flux = (interpolator(best_values))[0]
    df = pd.DataFrame({'WAVELENGTH': observed_wave, 'FLUX': model_flux})
    df.to_csv(f'{output_file}_best_fit.csv', index=False)