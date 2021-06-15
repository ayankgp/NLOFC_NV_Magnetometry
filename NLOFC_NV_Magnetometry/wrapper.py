import os
import ctypes
from ctypes import c_int, c_long, c_double, POINTER, Structure
import subprocess

__doc__ = """
Python wrapper for response.c
Compile with:
gcc -O3 -shared -o response.so response.c -lm -fopenmp -lnlopt -fPIC
"""

subprocess.run(["gcc", "-O3", "-shared", "-o", "response.so", "response.c", "-lm", "-lnlopt", "-fopenmp", "-fPIC"])


class c_complex(Structure):
    """
    Complex double ctypes
    """
    _fields_ = [
        ('real', c_double),
        ('imag', c_double)
    ]


class OFCParameters(Structure):
    """
    SpectraParameters structure ctypes
    """
    _fields_ = [
        ('excitedNUM', c_int),
        ('ensembleNUM', c_int),
        ('freqNUM', c_int),
        ('chiNUM', c_int),
        ('combNUM', c_int),
        ('resolutionNUM', c_int),
        ('basisNUM', c_int),
        ('frequency', POINTER(c_double)),
        ('omega_chi', POINTER(c_double)),
        ('combGAMMA', c_double),
        ('freqDEL', c_double),
        ('termsNUM', c_int),
        ('indices', POINTER(c_long)),
        ('basisINDX', POINTER(c_long)),
        ('modulations', POINTER(c_double)),
        ('envelopeWIDTH', c_double),
        ('envelopeCENTER', c_double),
        ('frequencyMC_opt', POINTER(c_double)),
        ('frequencyMC_RF1', POINTER(c_double)),
        # ('frequencyMC_RF2', POINTER(c_double))
    ]


class OFCMolecule(Structure):
    """
    Parameters structure ctypes
    """
    _fields_ = [
        ('levelsNUM', c_int),
        ('energies', POINTER(c_double)),
        ('levels', POINTER(c_double)),
        ('gammaMATRIX', POINTER(c_double)),
        ('muMATRIX', POINTER(c_complex)),
        ('polarizationINDEX', POINTER(c_complex)),
        ('polarizationMOLECULE', POINTER(c_complex)),
        ('chi1DIST', POINTER(c_complex)),
        ('chi3DIST', POINTER(c_complex)),
        ('chi1INDEX', POINTER(c_complex)),
        ('chi3INDEX', POINTER(c_complex)),
        ('probabilities', POINTER(c_double)),
    ]


try:
    lib = ctypes.cdll.LoadLibrary(os.getcwd() + "/response.so")
except OSError:
    raise NotImplementedError(
        """
        The library is absent. You must compile the C shared library using the commands:
        gcc -O3 -shared -o response.so response.c -lm -lnlopt -fopenmp -fPIC
        """
    )


lib.CalculateOFCResponse_C.argtypes = (
    POINTER(OFCMolecule),
    POINTER(OFCParameters),
)
lib.CalculateOFCResponse_C.restype = None

lib.CalculateChi_C.argtypes = (
    POINTER(OFCMolecule),
    POINTER(OFCParameters),
)
lib.CalculateChi_C.restype = None


def CalculateOFCResponse(ofc_mol, ofc_params):
    return lib.CalculateOFCResponse_C(
        ofc_mol,
        ofc_params
    )


def CalculateChi(ofc_mol, ofc_params):
    return lib.CalculateChi_C(
        ofc_mol,
        ofc_params
    )