<h1 align="center" id="title"> 🏰 castl 🏰 </h1>
<div align="center">
  <p id="description"> <b> Computional Anaylsis of Spectral TempLates (castl) </b> is a package designed to efficiently run <a href="https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo">Markov-Chain Monte-Carlo</a> simulations using the <a href="https://emcee.readthedocs.io/en/stable/">emcee</a> package. It uses a <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html">Radial Basis Function Interpolator</a> to perform efficient interpolation. The algorithm is streamlined for ease of use, requiring only four input variables to run simulations on any spectral model calculated on a grid. However, additional input variables are available for more customization and control. This software supports both regular and irregular grids, making it versatile and adaptable to various modeling scenarios. An example jupyter notebook for using castl is provided. Where castl outputs: a step figure for each parameter, the best fit spectra alongside your input spectra with a corner plot to the right, and a h5 file with the walker data. </p>
</div>

<div align="center">
  <h2>🛠️ Installation 🛠️</h2>
</div>

<div align="center">
<pp><b> pip Installation </b><pp>
</div>
<div align="center">
</div>

1. **Download Python:** Visit [here](https://www.python.org/downloads/) to install Python 
2. **Download pip:** Visit [here](https://pip.pypa.io/en/stable/installation/) to install pip
3. **Run Install Command:** Run the command in terminal:
   ```bash
   pip install castl

<div align="center">
  <p><b> Manual Installation </b></p>
</div>
<div align="center">
</div>

1. **Download Python:** Visit [here](https://www.python.org/downloads/) to install Python
2. **Download pip:** Visit [here](https://pip.pypa.io/en/stable/installation/) to install pip
3. **Downloading castl:** Download the latest version of castl in the "Releases" tab of the [Github](https://github.com/huntbrooks85/castl) page
4. **pip Packages:** Go into the directory of castl and run the command:
   ```bash
   pip install -r requirements.txt

<div align="center">
  <h2>⚙️ Using castl ⚙️</h2>
</div>

<div align="center">
  <p><b> How to Use h5grid </b></p>
</div>
<div align="center">
</div>

1. After castl is installed, verify the installation by running the following command: ```from castl.h5grid import *```. If you encounter any issues during installation, please reach out to Hunter Brooks for assistance. 
2. Ensure that model spectra are assigned as: column 1 (wavelength) and column 2 (flux) 
3. Assign the relavent variables as described below. 
4. To compile code, execute the command: ```h5grid(model_directory, model_parm, output_h5)```. These are the minimum required parameters for castl to run. You can include optional variables if needed.


<div align="center">
  <pp><b> Relavent Variables For h5grid </b></pp> 
</div>

- **Required Variables:**
  - **model_directory:** File path/name to your model spectra: *string*:
     - *example:* ```/Desktop/spectra/test.csv```

  - **model_parameters:** Model parameter names: *list*: *(must be in order that the paramters are in the file name)*
     - *example:* ```['Teff', 'log(g)', '[M/H]', 'C/O', 'log(Kzz)']```

  - **output_file:** File path/name of output file: *string*:
     - *example:* ```/models/test```

- **Optional Variables:**
  - **wavelength_region:** The wavelength region saved for each spectrum: *list*
    - *example:* ```[0, 10]```, default=```[0, np.inf]```




<div align="center">
  <p><b> How to Use mcfit </b></p>
</div>
<div align="center">
</div>

1. Ensure that your model is compiled into a h5 file using ```h5grid```
2. Ensure that the observed spectra are assigned as: column 1 (wavelength), column 2 (flux), and column 3 (uncertainty)
3. Once castl is successfully imported, define the relevant variables as shown below. Ensure that all required variables are formatted correctly.
4. To run the Markov-Chain Monte-Carlo simulation, execute the command: ```mcmcfit(input_file, output_file, h5_directory, model_parm)```. These are the minimum required parameters for castl to run. You can include optional variables if needed.

<div align="center">
  <pp><b> Relavent Variables For mcfit </b></pp> 
</div>

- **Required Variables:**
  - **input_file:** File path/name to your input spectrum: *string*:
     - *example:* ```/Desktop/spectra/test.csv```

  - **output_file:** File path/name of output file: *string*: *(do not include file type)*
    - *example:* ```/Output/test```

  - **h5_directory:** Directory name to model spectra h5 file: *string*: *(ensure no numbers are included in path name outside of numbers in model file name)*
    - *example:* ```/Desktop/model/LOWZ.h5```

  - **model_parm:** Model parameter names: *list*: *(must be the exact name used in h5 file)*
   - *example code:* ```['Teff', 'log(g)', '[M/H]', 'C/O', 'log(Kzz)']```

- **Optional Variables:**
  - **grid_scale:** Number of grid points around best grid point in each dimension: *int*
    - *example:* ```50```, default=10

  - **unit_wave:** Astropy units of the observed and model wavelength units: *list*: *(first index is observed, second index is model)*
    - *example:* ```[u.um, u.um]```, default=```[u.um, u.um]```

  - **unit_flux:** Astropy units of the observed and model flux units: *list*: *(first index is observed, second index is model)*
    - *example:* ```[u.Jy, (u.erg / (u.cm**2 * u.s * u.um))]```, default=```[(u.erg / (u.cm**2 * u.s * u.um)), (u.erg / (u.cm**2 * u.s * u.um))]```

  - **walkers:** Number of walkers for emcee calculation: *int*
    - *example:* ```25```, default=15

  - **steps:** Number of max steps for emcee calculation: *int*: *May be cutoff before this as a result of auto-correlation*
    - *example:* ```15000```, default=1000

  - **monitor:** Whether a monitoring step figure is displayed every 1000 steps: *boolean*
    - *example:* ```True```, default=False

  - **save_output:** Whether the output tables and figures are saved: *boolean*
    - *example:* ```True```, default=True

<div align="center">
  <h2>📞 Support & Development Team 📞</h2>
</div>

- **Mr. Hunter Brooks**
  - Email: hcb98@nau.edu

- **Mr. Efrain Efrain Alvarado III**

- **Dr. Adam Burgasser**

- **Dr. Chris Theissen**

- **Dr. Roman Gerasimov** 

<div align="center">
  <h2>📖 Acknowledgments 📖</h2>
</div>

1. If you intend to publish any calculations done by castl, please reference Brooks et al. (in prep.).

2. Please reference the relavent model citation.

