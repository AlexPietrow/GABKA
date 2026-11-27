# GABKA

GABKA (Generic Atmospheric Band Korrector Algorithm, Polish for sponge) removes telluric absorption lines from ground-based solar spectra. It fits the observed spectrum with the telluric-free solar disk-center atlas from the [HelioSpectrotron 5000](https://HS5000.vo.aip.de) (HS5000) multiplied by a precomputed [TAPAS](https://tapas.aeris-data.fr/) transmission spectrum, where the water vapor column, the instrumental profile, the solar Doppler shift, the wavelength solution and a continuum per fit range are free parameters. The fitted transmission is then divided out of the data.

The approach follows that of molecfit, but replaces the line-by-line radiative transfer with TAPAS spectra for the US Standard 1976 atmosphere, and uses the solar atlas rather than a smooth continuum as the reference spectrum. Because the telluric lines are at rest with respect to the observer, they also set the absolute wavelength scale of the data. GABKA was developed to clean the He I D3 and He I 10830 Å lines, which are both blended with telluric water vapor lines, but it works on any spectral window between 3290 and 12000 Å.

## Installation

    git clone https://github.com/AlexPietrow/GABKA.git
    cd GABKA
    python -m pip install .

The TAPAS models (US Standard 1976, zenith, 0 m and 2500 m altitude) and the telluric-free atlas are bundled with the package. GABKA needs numpy, scipy, matplotlib and astropy.

## Usage

    from gabka import remove_tellurics

    clean, T, info = remove_tellurics(wave, spec)

`wave` is the wavelength axis in air Å (an offset of up to about 0.5 Å is found by the fit) and `spec` the spectrum on that axis, in any units. The call returns the corrected spectrum, the fitted telluric transmission on the same pixels, and a dictionary with the fitted values and statistics (water column along the line of sight in mm, column scale and error, PSF FWHM, resolving power, solar shift, axis correction, corrected wavelength axis, residual rms and the model spectrum).

Options:

- `psf='fit'` fits a Gaussian instrumental profile (default). A number holds a Gaussian with that FWHM in Å fixed, and a function `psf(dlam)` or a tuple `(dlam, profile)` supplies your own profile.
- `altitude=2500` site altitude in m, used to pick the closest TAPAS model (0 or 2500 m).
- `exclude=[(lo, hi), ...]` wavelength ranges left out of the fit, for line cores that differ from the disk-center atlas (e.g. He I, Na D, active-region profiles).
- `verbose=True` prints each step and the fitted values.
- `plot=True` shows a diagnostic plot of the fit.
- `save='folder'` writes `<name>_tellurics.fits` (wavelength, corrected wavelength, observed, clean, transmission and model, with the fitted values in the header) and `<name>_tellurics.png`.
- `name='spectrum'` name used for the output files.

To correct a slit image or data cube, fit a representative average spectrum and divide the whole array by the transmission:

    from gabka import remove_tellurics, correct

    clean, T, info = remove_tellurics(wave, np.nanmean(cube, axis=(0, 1)))
    cube_clean = correct(cube, T)

Since the water vapor column and the airmass change during the day, we recommend refitting each observation.

## Examples and tests

    python examples/example_txt.py        # ZIMPOL He I D3 and GRIS He I 10830 example spectra
    python tests/test_synthetic.py        # recovers a known water column, resolving power and axis offset

The example spectra in `examples/data` are a slit-averaged ZIMPOL-3 (IRSOL) He I D3 frame and the median of a GRIS (GREGOR) He I 10830 Å scan from 2014 May 12.

## Code layout

- `gabka/functions.py`: the building blocks, one documented function per step (loading TAPAS and the atlas, fit masks, transmission, solar spectrum, instrumental profile, wavelength solution, continuum, forward model, grid search, least-squares fit, summary, output)
- `gabka/pipeline.py`: `remove_tellurics`, which chains these steps into the full correction

## Citing GABKA

If you use GABKA in a scientific publication, please cite the GABKA paper, as well as the underlying reference datasets and methods listed below, as appropriate for your application.

### References

- Pietrow, A.G.M. (2026), *Astronomy & Astrophysics*, in prep.  
  GABKA reference paper

- Pietrow, A.G.M. (2026), *Open Journal of Astrophysics*, **9**  
  HS5000 and the telluric-free solar atlas  
  https://ui.adsabs.harvard.edu/abs/2026arXiv260220101P/abstract

- Neckel, H., & Labs, D. (1984), *Solar Physics*, **90**, 205  
  Solar disk-center reference spectrum  
  https://ui.adsabs.harvard.edu/abs/1984SoPh...90..205N/abstract

- Bertaux, J.-L., et al. (2014), *Astronomy & Astrophysics*, **564**, A46  
  TAPAS atmospheric transmission service  
  https://ui.adsabs.harvard.edu/abs/2014A%26A...564A..46B/abstract

- Smette, A., et al. (2015), *Astronomy & Astrophysics*, **576**, A77  
  molecfit, method for fitting telluric absorption  
  https://ui.adsabs.harvard.edu/abs/2015A%26A...576A..77S/abstract

- Kausch, W., et al. (2015), *Astronomy & Astrophysics*, **576**, A78  
  molecfit, telluric correction  
  https://ui.adsabs.harvard.edu/abs/2015A%26A...576A..78K/abstract

## License

This project is licensed under the MIT License.

See the LICENSE file or:
https://opensource.org/license/mit
