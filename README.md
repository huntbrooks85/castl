<h1 align="center" id="title"> 🏰 castl 🏰 </h1>
<div align="center">
  <p id="description"> <b> Computional Anaylsis of Spectral TempLates (castl) </b> is a package designed to efficiently run <a href="https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo">Markov-Chain Monte-Carlo</a> simulations using the <a href="https://emcee.readthedocs.io/en/stable/">emcee</a> package. It uses a <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html">Radial Basis Function Interpolator</a> to perform efficient interpolation. The algorithm is streamlined for ease of use, requiring only four input variables to run simulations on any spectral model calculated on a grid. However, additional input variables are available for more customization and control. This software supports both regular and irregular grids, making it versatile and adaptable to various modeling scenarios. An example jupyter notebook for using castl is provided. Where castl outputs: a step figure for each parameter, the best fit spectra alongside your input spectra with a corner plot to the right, and a h5 file with the walker data. </p>
</div>

<div align="center">
  <h2>🛠️ Installation 🛠️</h2>
</div>

<div align="center">
<pp><b>⬇️ pip Installation ⬇️</b><pp>
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
  <p><b>💪 Manual Installation 💪</b></p>
</div>
<div align="center">
<pp><b>-----------------------------------------</b><pp>
</div>

1. **Download Python Version 3.8:** Visit [here](https://www.python.org/downloads/) to install Python
2. **Download pip:** Visit [here](https://pip.pypa.io/en/stable/installation/) to install pip
3. **Downloading castl:** Download the latest version of castl in the "Releases" tab on the [Github](https://github.com/huntbrooks85/castl) page
4. **pip Packages:** Go into the directory of castl and run the command:
   ```bash
   pip install -r requirements.txt

<div align="center">
  <h2>⚙️ Using castl ⚙️</h2>
</div>

<div align="center">
  <p><b>🦾 How to Use castl 🦾</b></p>
</div>
<div align="center">
<pp><b>-----------------------------------------</b><pp>
</div>

1. After castl is installed, verify the installation by running the following command: ```from castl.mcfit import *```. If you encounter any issues during installation, please reach out to Hunter Brooks for assistance.
2. Once castl is successfully imported, define the relevant variables as shown below. Ensure that all required variables are formatted correctly.
3. To run the Markov-Chain Monte-Carlo simulation, execute the command: ```mcmcfit(input_file, output_file, model_directory, model_parm)```. These are the minimum required parameters for castl to run. You can include optional variables if needed.

<div align="center">
  <p><b>🔎 Relavent Variables 🔍</b></p>
  <p>-----------------------------------------</p>
</div>

- **Required Variables:**
  - **input_file**: blah blah blah

  - **output_file**: blah blah blah

  - **model_directory**: blah blah blah

  - **model_parm**: blah blah blah

- **Optional Variables:**
  - **unit**: blah blah blah

  - **walkers**: blah blah blah

  - **max_step**: blah blah blah

  - **safety_coeff**: blah blah blah

  - **stretch_factor**: blah blah blah

  - **monitor**: blah blah blah

<div align="center">
  <p><b>☢️ Significant Details ☢️</b></p>
  <p>-----------------------------------------</p>
</div>

- **Note 1**: castl is current in a beta version, thus bugs may occur 

<div align="center">
  <h2>🧐 Example castl Output 🧐</h2>
</div>

<p align="center">
  <a href="https://ibb.co/wpby3Tz"><img src="/castl/castl_test-1.png" width="100%"></a> <br>
  Example Output run using the LOWZ Model Using castl Default Settings
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

