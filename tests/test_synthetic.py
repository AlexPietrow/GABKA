"""
Synthetic recovery test: atlas x TAPAS water (known column and PSF) plus noise, fitted with each PSF option.
The fitted-PSF case is run verbose and shows its diagnostic plot; the other PSF options print one line each.
Run with:  ipython3 tests/test_synthetic.py
"""
import numpy as np
from gabka import remove_tellurics
from gabka import functions as f


def make(lo, hi, pix, scale, R, offset, altitude=2500, noise=2e-3, seed=1):
    """Fake observation: telluric-free atlas times TAPAS water, Gaussian PSF, shifted axis, tilt and noise."""
    grid, dg = f.model_grid(np.arange(lo, hi, pix))
    tg, taus, meta, _ = f.load_tapas(altitude, lo, hi)
    aw, ai = f.load_atlas(lo, hi)
    T = np.exp(-scale * np.interp(grid, tg, taus['h2o']))
    fw = 0.5 * (lo + hi) / R
    conv = np.convolve(np.interp(grid, aw, ai) * T, f.gaussian_kernel(fw, dg), mode='same')
    wave = np.arange(lo, hi, pix)
    spec = np.interp(wave + offset, grid, conv) * (1 + 0.02 * (wave - wave.mean()) / (hi - lo))
    return wave, spec + np.random.default_rng(seed).normal(0, noise, wave.size), fw


def check(name, lo, hi, pix, scale, R, offset, exclude):
    wave, spec, fw = make(lo, hi, pix, scale, R, offset)
    sig = fw / 2.3548
    psfs = {'fit': 'fit', 'fixed': fw, 'callable': lambda d: np.exp(-0.5 * (d / sig) ** 2),
            'arrays': (np.linspace(-1, 1, 801), np.exp(-0.5 * (np.linspace(-1, 1, 801) / sig) ** 2))}
    ok = True
    for label, psf in psfs.items():
        if label == 'fit':
            print(f'\n--- {name}: fake spectrum with water scale {scale}, R = {R:,}, axis offset {offset * 1e3:+.0f} mA')
        clean, T, info = remove_tellurics(wave, spec, psf=psf, exclude=exclude, verbose=label == 'fit',
                                          plot=label == 'fit')
        good = (abs(info['scale']['h2o'] / scale - 1) < 0.05 and abs(info['R'] / R - 1) < 0.05
                and abs(info['axis_offset'] - offset) < pix / 4)
        ok &= good
        print(f'{name:8s} psf={label:8s} scale {info["scale"]["h2o"]:.3f} (true {scale}), R {info["R"]:,.0f} '
              f'(true {R:,}), offset {info["axis_offset"] * 1e3:+.1f} mA (true {offset * 1e3:+.1f}), '
              f'rms {info["rms"] * 100:.2f} %  ' + ('OK' if good else 'FAILED'))
    return ok


def main():
    ok = check('HeD3', 5872.0, 5901.0, 0.0236, 1.6, 40000, 0.12,
               [(5875.2, 5876.8), (5889.4, 5890.5), (5895.4, 5896.5)])
    ok &= check('He10830', 10823.5, 10841.8, 0.0181, 0.95, 110000, -0.10,
                [(10826.6, 10827.4), (10828.9, 10831.2)])
    print('all tests passed' if ok else 'some tests FAILED')


if __name__ == '__main__':
    main()
