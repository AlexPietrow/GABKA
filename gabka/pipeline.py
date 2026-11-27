"""
GABKA pipeline: chains the steps in gabka.functions into one telluric correction.

    from gabka import remove_tellurics
    clean, T, info = remove_tellurics(wave, spec)
"""
import time
import numpy as np
from . import functions as f

WL_DEGREE = 1        # wavelength solution: offset and dispersion
CONT_DEGREE = 2      # continuum polynomial per fit range
EDGE = 0.5           # A left out at both ends of the window (detector edges, prefilter roll-off)


def remove_tellurics(wave, spec, psf='fit', altitude=2500, exclude=None, verbose=False, plot=False,
                     save=None, name='spectrum'):
    """
    Fit and remove the telluric lines from a solar spectrum.

    wave      wavelength axis [air A], roughly calibrated (offsets up to ~0.5 A are found)
    spec      spectrum on that axis, any normalization
    psf       'fit'      fit a Gaussian instrumental profile (default)
              number     fixed Gaussian with this FWHM [A]
              callable   own profile psf(dlam), dlam the offset from line center [A]
              (dlam, p)  own profile as two arrays
    altitude  site altitude [m], picks the closest TAPAS model (0 or 2500 m)
    exclude   list of (lo, hi) ranges left out of the fit, e.g. line cores that differ from the
              disk-center atlas (He I 10830, He I D3, Na D, Si I 10827)
    verbose   print each step and the fitted values
    plot      show the diagnostic plot (observed and model in red and black, residuals)
    save      folder to write <name>_tellurics.fits (results table, fitted values in the header) and
              <name>_tellurics.png (diagnostic plot)
    name      name of the input, used for the output files (spaces and dots become underscores)

    Returns (clean, T, info): the corrected spectrum, the fitted telluric transmission on the same pixels
    (convolved with the PSF), and a dict with the fitted values and statistics (see functions.summarize).
    """
    t0 = time.time()
    wave, y = f.check_input(wave, spec)

    # 1. a descending axis is fitted flipped and flipped back
    if wave[-1] < wave[0]:
        clean, T, info = remove_tellurics(wave[::-1], y[::-1], psf, altitude, exclude, verbose)
        clean, T = clean[::-1], T[::-1]
        info['wave_true'], info['model'] = info['wave_true'][::-1], info['model'][::-1]
        output(wave, y, clean, T, info, altitude, plot, save, name, verbose)
        return clean, T, info

    # 2. normalize and set up the fit ranges and masks
    f.check_window(wave)
    norm = np.nanmedian(y)
    y = y / norm
    pix = np.median(np.diff(wave))
    exclude = list(exclude or [])
    fit_ranges = f.default_fit_ranges(wave, EDGE)
    ranges, use = f.fit_masks(wave, y, fit_ranges, exclude)
    f.log(verbose, t0, f'spectrum: {wave.size} px, {wave[0]:.2f}-{wave[-1]:.2f} A, {pix * 1e3:.2f} mA/px; '
                       f'{len(fit_ranges)} fit ranges, {len(exclude)} excluded, {use.sum()} px used')

    # 3. model grid, TAPAS opacities and the solar atlas
    grid, dg = f.model_grid(wave)
    tg, taus, meta, tapas_path = f.load_tapas(altitude, wave[0], wave[-1])
    tau = {s: np.interp(grid, tg, t) for s, t in taus.items()}
    species = f.select_species(tau)
    atlas_wave, atlas_int = f.load_atlas(wave[0], wave[-1])
    f.log(verbose, t0, f'TAPAS {meta["atmosphere"]}, {meta["altitude_m"]} m; fitting '
                       + ', '.join(f'{s} (tau max {tau[s].max():.3f})' for s in species))

    # 4. instrumental profile
    if isinstance(psf, str) and psf == 'fit':
        psf_mode, kernel = 'fit', None
    elif np.isscalar(psf):
        psf_mode, kernel = 'fixed Gaussian', f.gaussian_kernel(float(psf), dg)
    else:
        psf_mode, kernel = 'user PSF', f.psf_kernel(psf, dg)
    f.log(verbose, t0, f'PSF: {psf_mode}')

    # 5. everything the model needs, in one dict
    lam_c, half = 0.5 * (wave[0] + wave[-1]), 0.5 * (wave[-1] - wave[0])
    setup = dict(wave=wave, pix=pix, grid=grid, dg=dg, tau=tau, species=species,
                 atlas_wave=atlas_wave, atlas_int=atlas_int, fit_psf=psf_mode == 'fit', psf_mode=psf_mode,
                 kernel=kernel, lam_c=lam_c, half=half, x=(wave - lam_c) / half, ranges=ranges, use=use,
                 fit_ranges=fit_ranges, exclude=exclude, tapas_path=tapas_path,
                 ns=len(species), nw=WL_DEGREE + 1, nr=len(fit_ranges), nc=CONT_DEGREE + 1)

    # 6. start values, grid search for the axis offset and solar shift, least-squares fit
    p0 = f.initial_guess(setup, y)
    p0, axis = f.grid_search(p0, setup, y, use)
    f.log(verbose, t0, f'grid search: axis correction {axis * 1e3:+.0f} mA, '
                       f'solar shift {p0[setup["ns"]] * 1e3:+.0f} mA')
    lb, ub = f.parameter_bounds(setup)
    r, err, rms = f.fit_least_squares(p0, lb, ub, setup, y, use)
    f.log(verbose, t0, f'least squares: {p0.size} parameters, {r.nfev} evaluations, {r.message}')

    # 7. corrected spectrum, transmission and fitted values
    model, T = f.forward_model(r.x, setup)
    clean = f.correct(spec, T)
    info = f.summarize(r, err, rms, setup, meta)
    info['model'] = model * norm
    if verbose:
        f.print_summary(info)

    # 8. plot and save
    output(wave, np.asarray(spec, float), clean, T, info, altitude, plot, save, name, verbose)
    f.log(verbose, t0, 'done')
    return clean, T, info


def output(wave, spec, clean, T, info, altitude, plot, save, name, verbose):
    """Show the diagnostic plot if plot is True, and write the FITS table and PNG if a save folder is given."""
    if not (plot or save):
        return
    import matplotlib.pyplot as plt
    fig = f.plot_fit(wave, spec, clean, info['model'], info)
    if save:
        stem = f.output_stem(save, name)
        f.save_fits(stem + '.fits', wave, spec, clean, T, info, altitude)
        fig.savefig(stem + '.png', dpi=150)
        if verbose:
            print(f'wrote {stem}.fits and {stem}.png')
    if plot:
        plt.show()
    else:
        plt.close(fig)
