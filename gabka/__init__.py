"""GABKA: Generic Atmospheric Band Korrector Algorithm. Telluric correction for ground-based solar spectra."""
from .pipeline import remove_tellurics
from .functions import correct, plot_fit

__version__ = '1.0.0'
__all__ = ['remove_tellurics', 'correct', 'plot_fit']
