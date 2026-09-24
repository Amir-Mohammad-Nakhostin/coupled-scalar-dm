"""BAO compilation used by the scalar-tensor comparison scripts.

The values below are the user-specified diagonal-error compilation.  The
DESI DR2 subset is from DESI Collaboration 2025, arXiv:2503.14738.
"""

ALL_BAO = [
    # DESI DR2
    (0.295, "DV_over_rd", 7.942, 0.075, "DESI BGS"),
    (0.510, "DM_over_rd", 13.588, 0.167, "DESI LRG1"),
    (0.510, "DH_over_rd", 21.863, 0.425, "DESI LRG1"),
    (0.706, "DM_over_rd", 17.351, 0.177, "DESI LRG2"),
    (0.706, "DH_over_rd", 19.455, 0.330, "DESI LRG2"),
    (0.934, "DM_over_rd", 21.576, 0.152, "DESI LRG3+ELG1"),
    (0.934, "DH_over_rd", 17.641, 0.193, "DESI LRG3+ELG1"),
    (1.321, "DM_over_rd", 27.601, 0.318, "DESI ELG2"),
    (1.321, "DH_over_rd", 14.176, 0.221, "DESI ELG2"),
    (1.484, "DM_over_rd", 30.512, 0.760, "DESI QSO"),
    (1.484, "DH_over_rd", 12.817, 0.516, "DESI QSO"),
    (2.330, "DM_over_rd", 38.988, 0.531, "DESI Lya"),
    (2.330, "DH_over_rd", 8.632, 0.101, "DESI Lya"),
    # BOSS DR12
    (0.38, "DM_over_rd", 10.27, 0.15, "BOSS LOWZ"),
    (0.38, "DH_over_rd", 24.89, 0.58, "BOSS LOWZ"),
    (0.51, "DM_over_rd", 13.38, 0.18, "BOSS CMASS"),
    (0.51, "DH_over_rd", 22.43, 0.48, "BOSS CMASS"),
    (0.61, "DM_over_rd", 15.45, 0.22, "BOSS CMASS"),
    (0.61, "DH_over_rd", 20.94, 0.47, "BOSS CMASS"),
    # eBOSS + 6dFGS + MGS
    (0.106, "DV_over_rd", 3.06, 0.14, "6dFGS"),
    (0.15, "DV_over_rd", 4.47, 0.16, "SDSS MGS"),
    (0.70, "DM_over_rd", 17.86, 0.33, "eBOSS LRG"),
    (0.70, "DH_over_rd", 19.33, 0.53, "eBOSS LRG"),
    (0.85, "DM_over_rd", 19.5, 1.0, "eBOSS ELG"),
    (1.48, "DM_over_rd", 30.69, 0.80, "eBOSS QSO"),
    (2.33, "DM_over_rd", 37.6, 1.9, "eBOSS Lya"),
    (2.33, "DH_over_rd", 8.93, 0.28, "eBOSS Lya"),
]

__all__ = ["ALL_BAO"]
