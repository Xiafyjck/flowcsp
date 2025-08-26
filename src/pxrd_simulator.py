from pymatgen.core.structure import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator
import numpy as np

class PXRDSimulator:
    def __init__(self, wavelength='CuKa', fwhm=0.1, two_theta_min=5.0, two_theta_max=120.0, num_points=11501):
        self.wavelength = wavelength
        self.fwhm = fwhm
        self.two_theta_min = two_theta_min
        self.two_theta_max = two_theta_max
        self.num_points = num_points

        self.xrd_calculator = XRDCalculator(wavelength=self.wavelength)
        self.shape_func = lambda x: np.exp(-4 * np.log(2) * (x / self.fwhm) ** 2) * (2 * np.sqrt(np.log(2) / np.pi) / self.fwhm)
    
    def simulate(self, structure: Structure):
        xrd_pattern = self.xrd_calculator.get_pattern(structure, two_theta_range=(self.two_theta_min, self.two_theta_max), scaled=True)
        x = xrd_pattern.x
        y = xrd_pattern.y
        
        x_broad = np.linspace(self.two_theta_min, self.two_theta_max, self.num_points)
        y_broad = np.zeros_like(x_broad)
        for xi, yi in zip(x, y):
            xi_contribute = self.shape_func(x_broad - xi)
            y_broad += yi * xi_contribute
        
        return x_broad, y_broad

        
