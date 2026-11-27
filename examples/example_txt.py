"""
Minimal example: remove the tellurics from the two example spectra in examples/data.
Run from the examples folder with:  ipython3 example_txt.py
Shows the diagnostic plot of each fit (close the window to continue) and writes the results and plots to examples/gabka_out.
"""
import os
import numpy as np
from gabka import remove_tellurics

HERE = os.path.dirname(os.path.abspath(__file__))

# line cores that differ from the disk-center atlas, left out of the fit
EXCLUDE_D3 = [(5875.2, 5876.8), (5889.4, 5890.5), (5895.4, 5896.5)]      # He I D3, Na D2, Na D1
EXCLUDE_10830 = [(10826.6, 10827.4), (10828.9, 10831.2)]                  # Si I 10827, He I 10830

for fname, altitude, exclude in [('zimpol_HeD3_mean.txt', 0, EXCLUDE_D3),          # IRSOL, Locarno
                                 ('gris_He10830_median.txt', 2500, EXCLUDE_10830)]:  # GREGOR, Tenerife
    print(f'\n=== {fname}')
    wave, spec = np.loadtxt(os.path.join(HERE, 'data', fname), unpack=True)
    clean, T, info = remove_tellurics(wave, spec, altitude=altitude, exclude=exclude, verbose=True, plot=True,
                                      save=os.path.join(HERE, 'gabka_out'), name=fname[:-4])
