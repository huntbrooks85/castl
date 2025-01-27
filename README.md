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
<pp><b>-----------------------------------------</b><pp>
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
<pp><b>-----------------------------------------</b><pp>
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
  <p><b> How to Use castl </b></p>
</div>
<div align="center">
<pp><b>-----------------------------------------</b><pp>
</div>

1. After castl is installed, verify the installation by running the following command: ```from castl.mcfit import *```. If you encounter any issues during installation, please reach out to Hunter Brooks for assistance.
2. Once castl is successfully imported, define the relevant variables as shown below. Ensure that all required variables are formatted correctly.
3. To run the Markov-Chain Monte-Carlo simulation, execute the command: ```mcmcfit(input_file, output_file, model_directory, model_parm)```. These are the minimum required parameters for castl to run. You can include optional variables if needed.

<div align="center">
  <p><b> Relavent Variables </b></p>
  <p>-----------------------------------------</p>
</div>

- **Required Variables:**
  - **input_file:** File path/name to your input spectrum: *string*: *(only csv supported)*
     - *example:* ```/Desktop/spectra/test.csv```

  - **output_file:** File path/name of output file: *string*: *(do not include file type)*
    - *example:* ```/Output/test```

  - **model_directory:** Directory name to model spectra: *string*: *(ensure no numbers are included in path name outside of numbers in model file name)*
    - *example:* ```/Desktop/model/LOWZ/```

  - **model_parm:** Model parameter names: *list*: *(must be in order that the paramters are in the file name)*
   - *example order:*
      - <b>Filename:</b> LOW_Z_BD_GRID_CLEAR_Teff_500.0_logg_3.5_logZ_-0.5_CtoO_0.1_logKzz_10.0_spec.txt
      - <b>Order:</b> Teff, logg, logZ, CtoO, logKzz
   - *example code:* ```['Teff', 'log(g)', '[M/H]', 'C/O', 'log(Kzz)']```

- **Optional Variables:**
  - **unit:** The number to convert model spectra wavelength unit to observed spectrum wavelength: *float*
    - *example:* 0.0001, default=1 (model: Å -> observed: µm)

  - **walkers:** Number of walkers for emcee calculation: *int*
    - *example:* 25, default=15

  - **max_step:** Number of max steps for emcee calculation: *int*: *May be cutoff before this as a result of auto-correlation*
    - *example:* 15000, default=1000

  - **safety_coeff:** The number multiplying the steps found after ideal tau was found: *int*
    - *example:* 8, default=10

  - **stretch_factor:** The emcee StretchMove factor: *float*
    - *example:* 10, default=2

  - **monitor:** Whether a monitoring step figure is displayed every 1000 steps: *boolean*
    - *example:* True, default=False

<div align="center">
  <h2> Example castl Output </h2>
</div>

<p align="center">
  <a href="https://ibb.co/wpby3Tz"><img src="/castl/castl_test-1.png" width="100%"></a> <br>
  Example Output Using castl on the LOWZ Model With Default Setting Activated 
</p>

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

