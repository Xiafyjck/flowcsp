import numpy as np

from pymatgen.analysis.structure_matcher import StructureMatcher

def rwp(y_calc, y_obs, epsilon=0.01):
    weights = 1 / np.maximum(y_obs, epsilon)
    numerator = np.sum(weights * (y_calc - y_obs) ** 2)
    denominator = np.sum(weights * y_obs ** 2)
    rwp = np.sqrt(numerator / denominator)
    return rwp

def rmsd(structure_calc, structure_obs):
    matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)
    rmsd = matcher.get_rms_dist(structure_calc, structure_obs)
    return rmsd
