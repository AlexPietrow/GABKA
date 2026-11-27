"""
GABKA building blocks. Each function does one step of the telluric fit; gabka.pipeline chains them.

Model (molecfit-style, with TAPAS instead of LBLRTM and a telluric-free solar atlas as reference):

    T(lam)  = exp[ - sum_j s_j tau_j(lam) ]              per-species TAPAS line opacity, one free scale each
    S(lam)  = S_atlas(lam - dlam_sun)                    telluric-free atlas, Doppler shifted
    M(w_i)  = P_r(w_i) * [ (S T) * G_fwhm ](lam_i)       Gaussian PSF, continuum polynomial per fit range
    lam_i   = lam_c + (D/2) sum_k a_k T_k(x_i)           Chebyshev wavelength solution of the data axis
    I_corr  = I_obs / [ T * G_fwhm ](lam_i)

All wavelengths are in air Angstrom. The parameter vector p used by the fit is ordered as
[species scales s_j, solar shift, PSF FWHM (only if fitted), Chebyshev coefficients, continuum coefficients].
"""
import os
import time
import numpy as np
from scipy.optimize import least_squares

C_KMS = 299792.458
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
TAPAS_FILES = {0: os.path.join(DATA_DIR, 'tapas_us76_0m.npz'), 2500: os.path.join(DATA_DIR, 'tapas_us76_2500m.npz')}
ATLAS_FILE = os.path.join(DATA_DIR, 'telfree_atlas.npy')
TAPAS_RANGE = (3000.0, 12000.0)     # bundled TAPAS files
ATLAS_RANGE = (3290.0, 12510.0)     # telluric-free Hamburg FTS atlas
SPECIES = ('h2o', 'o2', 'co2')

def log(verbose, t0, msg):
    """Print msg with the elapsed time since t0 when verbose is True."""
    if verbose:
        print(f'[{time.time() - t0:6.1f} s] {msg}', flush=True)


# ---------------------------------------------------------------------------------------------- input data

def check_input(wave, spec):
    """Return wave and spec as float arrays; raise if they are not 1D arrays of equal length."""
    wave = np.asarray(wave, float)
    spec = np.asarray(spec, float)
    if wave.ndim != 1 or spec.shape != wave.shape:
        raise ValueError(f'wave and spec must be 1D arrays of the same length, got {wave.shape} and {spec.shape}')
    return wave, spec


def check_window(wave):
    """Raise if the wavelength window falls outside the bundled TAPAS or atlas coverage (5 A margin)."""
    lo, hi = wave.min(), wave.max()
    if lo < TAPAS_RANGE[0] + 5 or hi > TAPAS_RANGE[1] - 5:
        raise ValueError(f'window {lo:.1f}-{hi:.1f} A is outside the bundled TAPAS range '
                         f'{TAPAS_RANGE[0]:.0f}-{TAPAS_RANGE[1]:.0f} A')
    if lo < ATLAS_RANGE[0] + 5 or hi > ATLAS_RANGE[1] - 5:
        raise ValueError(f'window {lo:.1f}-{hi:.1f} A is outside the atlas range '
                         f'{ATLAS_RANGE[0]:.0f}-{ATLAS_RANGE[1]:.0f} A')


def load_tapas(altitude, lo, hi, path=None):
    """
    Per-species TAPAS line optical depth around [lo, hi] (5 A margin).

    altitude  site altitude [m]; the bundled file closest in altitude is used unless path is given
    path      own .npz file with lam0, dv [km/s], n, one tau array per species and a meta dict
    Returns (wavelength grid, {species: tau}, meta, path of the file used).
    """
    if path is None:
        path = TAPAS_FILES[min(TAPAS_FILES, key=lambda a: abs(a - altitude))]
    d = np.load(path, allow_pickle=True)
    grid = float(d['lam0']) * np.exp(np.arange(int(d['n'])) * float(d['dv']) / C_KMS)
    m = (grid > lo - 5) & (grid < hi + 5)
    return grid[m], {s: d[s][m].astype(float) for s in SPECIES}, d['meta'].item(), path


def load_atlas(lo, hi, path=None):
    """Telluric-free solar atlas (wavelength, I/Ic) around [lo, hi] (5 A margin)."""
    a = np.load(path or ATLAS_FILE).astype(float)
    m = (a[0] > lo - 5) & (a[0] < hi + 5)
    return a[0][m], a[1][m]


# ---------------------------------------------------------------------------------------------- fit setup

def default_fit_ranges(wave, edge=0.5, n_ranges=None, length=7.0):
    """Split the window, minus `edge` A at both ends, into n_ranges pieces (default: about `length` A each)."""
    a, b = wave[0] + edge, wave[-1] - edge
    n = n_ranges or max(1, int(round((b - a) / length)))
    e = np.linspace(a, b, n + 1)
    return list(zip(e[:-1], e[1:]))


def fit_masks(wave, spec, fit_ranges, exclude):
    """
    Pixel masks for the fit.
    Returns (one boolean mask per fit range, mask of all pixels used: inside a range, finite, not excluded).
    """
    ranges = [(wave >= a) & (wave < b) for a, b in fit_ranges]
    use = np.any(ranges, axis=0) & np.isfinite(spec)
    for a, b in exclude:
        use &= ~((wave > a) & (wave < b))
    return ranges, use


def model_grid(wave, oversample=8, pad=1.5):
    """Fine uniform model grid covering the data with `pad` A margin, `oversample` points per data pixel."""
    dg = np.median(np.diff(wave)) / oversample
    return np.arange(wave[0] - pad, wave[-1] + pad, dg), dg


def select_species(tau, species='auto', threshold=0.005):
    """Species to fit: those with peak tau above threshold for 'auto', otherwise the given list (checked)."""
    if species == 'auto':
        return [s for s in SPECIES if tau[s].max() > threshold]
    bad = [s for s in species if s not in SPECIES]
    if bad:
        raise ValueError(f'unknown species {bad}, choose from {SPECIES}')
    return list(species)


# ---------------------------------------------------------------------------------------------- model pieces

def transmission(tau, species, scales):
    """Telluric transmission exp(-sum_j s_j tau_j) on the model grid."""
    total = sum(s * tau[sp] for s, sp in zip(scales, species)) if len(species) else 0.0
    return np.exp(-total) * np.ones_like(next(iter(tau.values())))


def solar_spectrum(grid, atlas_wave, atlas_int, shift):
    """Atlas intensity on the model grid, shifted by `shift` A."""
    return np.interp(grid - shift, atlas_wave, atlas_int)


def gaussian_kernel(fwhm, dg):
    """Normalized Gaussian of the given FWHM [A] sampled at step dg, out to 5 sigma."""
    sig = max(fwhm, 1e-4) / 2.3548
    x = np.arange(-int(np.ceil(5 * sig / dg)), int(np.ceil(5 * sig / dg)) + 1) * dg
    k = np.exp(-0.5 * (x / sig) ** 2)
    return k / k.sum()


def psf_kernel(psf, dg, half_width=2.0):
    """
    User PSF sampled on the model grid step dg and normalized to unit area.
    psf   callable profile(dlam) with dlam the offset from line center [A], evaluated within +-half_width A,
          or a tuple (dlam, profile) of arrays, interpolated onto the grid (zero outside the given range)
    """
    x = np.arange(-int(half_width / dg), int(half_width / dg) + 1) * dg
    if callable(psf):
        k = np.asarray(psf(x), float)
    else:
        dl, prof = (np.asarray(a, float) for a in psf)
        k = np.interp(x, dl, prof, left=0.0, right=0.0)
    if k.sum() <= 0:
        raise ValueError('the PSF has no positive area on the sampled range')
    return k / k.sum()


def kernel_fwhm(kernel, dg):
    """Full width at half maximum of a sampled, single-peaked kernel [A], from the interpolated half-maximum crossings."""
    k = kernel / kernel.max()
    above = np.where(k >= 0.5)[0]
    i0, i1 = above[0], above[-1]
    left = i0 - (k[i0] - 0.5) / (k[i0] - k[i0 - 1]) if i0 > 0 else 0.0
    right = i1 + (k[i1] - 0.5) / (k[i1] - k[i1 + 1]) if i1 < k.size - 1 else k.size - 1.0
    return float((right - left) * dg)


def wavelength_solution(x, cheb, lam_c, half):
    """True wavelength of each pixel, lam_c + half * sum_k a_k T_k(x), x the pixel axis scaled to [-1, 1]."""
    return lam_c + half * np.polynomial.chebyshev.chebval(x, cheb)


def convolve_to_pixels(f, kernel, grid, lam):
    """Convolve f (on the model grid) with the kernel and interpolate onto the pixel wavelengths lam."""
    return np.interp(lam, grid, np.convolve(f, kernel, mode='same'))


def continuum(wave, ranges, coefs):
    """Continuum polynomial of each fit range (centered on the range), NaN outside the ranges."""
    out = np.full_like(wave, np.nan)
    for m, c in zip(ranges, coefs):
        out[m] = np.polyval(c[::-1], wave[m] - wave[m].mean())
    return out


# ---------------------------------------------------------------------------------------------- parameters

def unpack(p, setup):
    """Split the parameter vector into (scales, solar shift, PSF FWHM, Chebyshev coefficients, continuum)."""
    ns, nw, nr, nc = setup['ns'], setup['nw'], setup['nr'], setup['nc']
    i = 0
    scales = p[i:i + ns]; i += ns
    shift = p[i]; i += 1
    if setup['fit_psf']:
        fwhm = p[i]; i += 1
    else:
        fwhm = None
    cheb = p[i:i + nw]; i += nw
    cont = p[i:i + nr * nc].reshape(nr, nc)
    return scales, shift, fwhm, cheb, cont


def axis_index(setup):
    """Index of the zeroth Chebyshev coefficient (the axis offset) in the parameter vector."""
    return setup['ns'] + 1 + (1 if setup['fit_psf'] else 0)


def initial_guess(setup, y):
    """Start values: unit scales, no shifts, R = 60000 if the PSF is fitted, unit dispersion, flat continua."""
    nw, nr, nc = setup['nw'], setup['nr'], setup['nc']
    cheb = np.zeros(nw)
    if nw > 1:
        cheb[1] = 1.0
    cont = np.zeros((nr, nc))
    cont[:, 0] = [np.nanmedian(y[m]) if m.any() else 1.0 for m in setup['ranges']]
    psf = [np.median(setup['wave']) / 60000.0] if setup['fit_psf'] else []
    return np.r_[np.ones(setup['ns']), 0.0, psf, cheb, cont.ravel()]


def parameter_bounds(setup):
    """
    Lower and upper bounds: scales 0-10, solar shift and axis offset within 0.5 A, PSF from half a pixel to
    1 A, dispersion within 2 %, higher Chebyshev terms within 1 %, continua free.
    """
    ns, nw, nr, nc, half = setup['ns'], setup['nw'], setup['nr'], setup['nc'], setup['half']
    psf_lo, psf_hi = ([0.5 * setup['pix']], [1.0]) if setup['fit_psf'] else ([], [])
    cheb_lo = [-0.5 / half, 0.98][:nw] + [-0.01] * (nw - 2)
    cheb_hi = [0.5 / half, 1.02][:nw] + [0.01] * (nw - 2)
    lb = np.r_[np.zeros(ns), -0.5, psf_lo, cheb_lo, -np.inf * np.ones(nr * nc)]
    ub = np.r_[10 * np.ones(ns), 0.5, psf_hi, cheb_hi, np.inf * np.ones(nr * nc)]
    return lb, ub


# ---------------------------------------------------------------------------------------------- model and fit

def forward_model(p, setup):
    """
    Model spectrum on the data pixels for parameters p.
    Returns (model with continuum, NaN outside the fit ranges; PSF-convolved transmission on the pixels).
    """
    scales, shift, fwhm, cheb, cont = unpack(p, setup)
    grid = setup['grid']
    T = transmission(setup['tau'], setup['species'], scales)
    S = solar_spectrum(grid, setup['atlas_wave'], setup['atlas_int'], shift)
    k = gaussian_kernel(fwhm, setup['dg']) if setup['fit_psf'] else setup['kernel']
    lam = wavelength_solution(setup['x'], cheb, setup['lam_c'], setup['half'])
    ST = convolve_to_pixels(S * T, k, grid, lam)
    Tc = convolve_to_pixels(T, k, grid, lam)
    return ST * continuum(setup['wave'], setup['ranges'], cont), Tc


def chi2(p, setup, y, use):
    """Sum of squared residuals over the used pixels."""
    r = (y - forward_model(p, setup)[0])[use]
    return np.sum(r * r)


def grid_search(p0, setup, y, use, max_offset=0.5, max_shift=0.3):
    """
    Coarse start for the two shifts: first the axis offset (moves tellurics and solar lines together, step half
    a pixel within max_offset A), then the solar shift (step a quarter pixel within max_shift A).
    Returns the updated start vector and the axis correction found [A].
    """
    p0 = p0.copy()
    ns, ia, half, pix = setup['ns'], axis_index(setup), setup['half'], setup['pix']
    offsets = np.arange(-max_offset, max_offset + 1e-9, max(pix / 2, 0.005))
    best = min(offsets, key=lambda o: chi2(np.r_[p0[:ia], -o / half, p0[ia + 1:]], setup, y, use))
    p0[ia] = -best / half
    shifts = np.arange(-max_shift, max_shift + 1e-9, max(pix / 4, 0.003))
    p0[ns] = min(shifts, key=lambda d: chi2(np.r_[p0[:ns], d, p0[ns + 1:]], setup, y, use))
    return p0, -best


def fit_least_squares(p0, lb, ub, setup, y, use):
    """
    Bounded least-squares fit (scipy trust-region reflective).
    Returns (scipy result, 1-sigma errors from the Jacobian scaled by the reduced chi2, residual rms).
    """
    p0 = np.clip(p0, lb + 1e-12, ub - 1e-12)
    r = least_squares(lambda p: (y - forward_model(p, setup)[0])[use], p0, bounds=(lb, ub),
                      x_scale='jac', method='trf')
    s2 = np.sum(r.fun ** 2) / max(r.fun.size - r.x.size, 1)
    err = np.sqrt(np.abs(np.diag(np.linalg.pinv(r.jac.T @ r.jac)) * s2))
    return r, err, float(np.sqrt(s2))


# ---------------------------------------------------------------------------------------------- output

def summarize(r, err, rms, setup, meta):
    """
    Fitted values and statistics as a plain dict: column scale per species (+ error), line-of-sight water vapor column [mm],
    PSF FWHM [A] and resolving power, solar shift [A and km/s], axis correction [A], corrected wavelength
    axis, residual rms (fraction of the median), pixels used, optimizer message.
    """
    scales, shift, fwhm, cheb, _ = unpack(r.x, setup)
    escales, eshift, efwhm, _, _ = unpack(err, setup)
    wave, species = setup['wave'], setup['species']
    lam = wavelength_solution(setup['x'], cheb, setup['lam_c'], setup['half'])
    if fwhm is None:
        fwhm, efwhm = kernel_fwhm(setup['kernel'], setup['dg']), 0.0
    wc = np.median(wave)
    return dict(
        scale=dict(zip(species, map(float, scales))), scale_err=dict(zip(species, map(float, escales))),
        h2o_mm=float(scales[species.index('h2o')] * meta['h2o_column_mm']) if 'h2o' in species else np.nan,
        psf_fwhm=float(fwhm), psf_fwhm_err=float(efwhm), psf=setup['psf_mode'], R=float(wc / fwhm),
        solar_shift=float(shift), solar_shift_err=float(eshift), solar_shift_kms=float(shift / wc * C_KMS),
        axis_offset=float(np.median(lam - wave)), wave_true=lam, rms=rms,
        n_used=int(setup['use'].sum()), fit_ranges=setup['fit_ranges'], exclude=setup['exclude'],
        tapas=os.path.basename(setup['tapas_path']), message=r.message)


def print_summary(info):
    """Print the fitted values of an info dict from summarize()."""
    print('-' * 72)
    for sp, s in info['scale'].items():
        extra = f'  -> {info["h2o_mm"]:.2f} mm water vapor along the line of sight' if sp == 'h2o' else ''
        print(f'  {sp:4s} column scale   {s:8.3f} +- {info["scale_err"][sp]:.3f}{extra}')
    err = f' +- {info["psf_fwhm_err"] * 1e3:.1f}' if info['psf'] == 'fit' else f' ({info["psf"]})'
    print(f'  PSF FWHM          {info["psf_fwhm"] * 1e3:8.1f} mA{err}  -> R = {info["R"]:,.0f}')
    print(f'  solar shift       {info["solar_shift"] * 1e3:+8.1f} +- {info["solar_shift_err"] * 1e3:.1f} mA  '
          f'({info["solar_shift_kms"]:+.2f} km/s)')
    print(f'  axis correction   {info["axis_offset"] * 1e3:+8.1f} mA (median)')
    print(f'  residual rms      {info["rms"] * 100:8.3f} %   on {info["n_used"]} px')
    print('-' * 72)


def correct(spec, T):
    """Divide a spectrum (or any array with wavelength last) by the transmission T."""
    return np.asarray(spec, float) / T


def plot_fit(wave, spec, clean, model, info, path=None):
    """Observed vs model (a), observed vs corrected (b) and residuals (c). Saves to `path` if given."""
    import matplotlib.pyplot as plt
    y = np.asarray(spec, float)
    fig, ax = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True, gridspec_kw=dict(height_ratios=[1, 1, 0.5]))
    ax[0].plot(wave, y, 'r', lw=1, label='observed')
    ax[0].plot(wave, model, 'k--', lw=1, label='model')
    ax[1].plot(wave, y, 'r', lw=1, label='observed')
    ax[1].plot(wave, clean, 'k', lw=1, label='corrected')
    ax[2].plot(wave, (y - model) / np.nanmedian(y), 'k', lw=0.8)
    w0, w1 = min(wave[0], wave[-1]), max(wave[0], wave[-1])
    shade = list(info['exclude']) + [(w0, info['fit_ranges'][0][0]), (info['fit_ranges'][-1][1], w1)]
    for a in ax:
        for lo, hi in shade:
            if hi > lo:
                a.axvspan(lo, hi, color='0.9', zorder=0)
    ax[0].set_ylabel('I')
    ax[1].set_ylabel('I')
    ax[2].set_ylabel('obs $-$ model')
    ax[2].set_xlabel('Wavelength [Å]')
    ax[2].ticklabel_format(axis='x', useOffset=False)
    ax[2].tick_params(axis='x', rotation=45)
    for a, l in zip(ax, 'abc'):
        a.text(0.01, 0.92, l + ')', transform=a.transAxes, va='top', bbox=dict(fc='w', ec='none', pad=1.5))
    ax[0].legend(loc='lower right', ncol=2, frameon=False)
    ax[1].legend(loc='lower right', ncol=2, frameon=False)
    fig.tight_layout()
    fig.align_ylabels()
    if path:
        fig.savefig(path, dpi=150)
        print(f'wrote {path}')
    return fig


def output_stem(folder, name):
    """Path stem for the output files: name with spaces, dots and slashes replaced by underscores."""
    clean = ''.join(c if c.isalnum() or c in '-_' else '_' for c in str(name)).strip('_') or 'spectrum'
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, clean + '_tellurics')


def save_fits(path, wave, spec, clean, T, info, altitude):
    """
    Write the results to a FITS binary table: WAVE, WAVE_TRUE, OBSERVED, CLEAN, TRANSMISSION, MODEL, with the
    fitted values (water column, PSF, R, shifts, rms) in the header.
    """
    from astropy.io import fits
    cols = [fits.Column(name=n, format='D', array=np.asarray(a, float)) for n, a in
            (('WAVE', wave), ('WAVE_TRUE', info['wave_true']), ('OBSERVED', spec), ('CLEAN', clean),
             ('TRANSMISSION', T), ('MODEL', info['model']))]
    hdu = fits.BinTableHDU.from_columns(cols, name='GABKA')
    h = hdu.header
    h['H2O_MM'] = (info['h2o_mm'], 'line-of-sight water vapor column [mm]')
    for sp, sc in info['scale'].items():
        h[f'SCL_{sp.upper()}'] = (sc, f'{sp} column scale relative to TAPAS')
        h[f'ESCL_{sp.upper()}'] = (info['scale_err'][sp], f'{sp} column scale error')
    h['PSF'] = (info['psf'], 'PSF mode')
    h['PSF_FWHM'] = (info['psf_fwhm'], 'PSF FWHM [A]')
    h['R'] = (info['R'], 'resolving power')
    h['SOLSHIFT'] = (info['solar_shift'], 'solar shift [A]')
    h['AXISOFF'] = (info['axis_offset'], 'median axis correction [A]')
    h['RMS'] = (info['rms'], 'residual rms / median')
    h['NUSED'] = (info['n_used'], 'pixels used in the fit')
    h['ALTITUDE'] = (altitude, 'site altitude [m]')
    h['TAPAS'] = (info['tapas'], 'TAPAS model')
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path, overwrite=True)
