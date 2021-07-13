#!/usr/bin/env python

"""
CalculateSpectra.py:

Class containing C calls for spectra calculation and discriminating OFC-pulse generation.
Plots results obtained from C calls.
"""

__author__ = "Ayan Chattopadhyay"
__affiliation__ = "Princeton University"


# ---------------------------------------------------------------------------- #
#                      LOADING PYTHON LIBRARIES AND FILES                      #
# ---------------------------------------------------------------------------- #

from multiprocessing import cpu_count
from types import MethodType, FunctionType
from itertools import product
from functions import *
from wrapper import *
import numpy as np
import time
from scipy.interpolate import interp1d
import sys


class ADict(dict):
    """
    Appended Dictionary: where keys can be accessed as attributes: A['*'] --> A.*
    """

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            dict.__getattribute__(self, item)


class OFC:
    """
    Calculates the ofc response of the molecule
    """

    def __init__(self, ofc_variables, **kwargs):
        """
        __init__ function call to initialize variables from the keyword args for the class instance
         provided in __main__ and add new variables for use in other functions in this class, with
         data from SystemVars.
        """

        for name, value in kwargs.items():
            if isinstance(value, FunctionType):
                setattr(self, name, MethodType(value, self))
            else:
                setattr(self, name, value)

        self.muMATRIX = np.ascontiguousarray(self.muMATRIX)

        self.energies = np.ascontiguousarray(self.energies)
        self.levelsNUM = ofc_variables.levelsNUM
        self.frequency, self.freq12, self.field1FREQ, self.field2FREQ, self.field3FREQ = nonuniform_frequency_range_3(ofc_variables)
        print("freqDEL_start_py = ", self.frequency[3] - self.frequency[0])

        # self.omega_chi = np.linspace(0., 1. * ofc_variables.freqDEL * ofc_variables.combNUM, ofc_variables.chiNUM)
        self.omega_chi = np.linspace(6.8e-8, 7.125e-8, ofc_variables.chiNUM + 1)
        self.omega_chi = np.linspace(0, 1.e-5, ofc_variables.chiNUM + 1)
        self.polarizationEMPTY = np.zeros((ofc_variables.molNUM, ofc_variables.ensembleNUM, self.frequency.size), dtype=np.complex)
        self.polarizationFIELD = np.zeros((ofc_variables.molNUM, ofc_variables.ensembleNUM, self.frequency.size), dtype=np.complex)
        self.polarizationINDEX = np.zeros((ofc_variables.molNUM, self.frequency.size), dtype=np.complex)
        self.polarizationMOLECULE = np.zeros((ofc_variables.molNUM, ofc_variables.ensembleNUM, self.frequency.size), dtype=np.complex)
        self.polarizationTOTALEMPTY = np.zeros((ofc_variables.molNUM, self.frequency.size), dtype=np.complex)
        self.polarizationTOTALFIELD = np.zeros((ofc_variables.molNUM, self.frequency.size), dtype=np.complex)
        self.polarizationTOTALEMPTY_DIST = np.zeros((ofc_variables.basisNUM, ofc_variables.basisNUM,
                                                     ofc_variables.basisNUM, ofc_variables.molNUM, self.frequency.size), dtype=np.complex)
        self.polarizationTOTALFIELD_DIST = np.zeros((ofc_variables.basisNUM, ofc_variables.basisNUM,
                                                     ofc_variables.basisNUM, ofc_variables.molNUM, self.frequency.size),
                                                    dtype=np.complex)
        self.chi1DIST = np.zeros((ofc_variables.molNUM, ofc_variables.ensembleNUM, self.omega_chi.size), dtype=np.complex)
        self.chi3DIST = np.zeros((ofc_variables.molNUM, ofc_variables.ensembleNUM, self.omega_chi.size, self.omega_chi.size), dtype=np.complex)
        self.chi1INDEX = np.zeros((ofc_variables.molNUM, self.omega_chi.size), dtype=np.complex)
        self.chi3INDEX = np.zeros((ofc_variables.molNUM, self.omega_chi.size, self.omega_chi.size), dtype=np.complex)
        self.polINDX = np.empty((ofc_variables.basisNUM, ofc_variables.basisNUM, ofc_variables.basisNUM))
        self.basisINDX = np.empty(3, dtype=int)
        self.indices = np.empty(3, dtype=int)

        self.chi3MATRIX = np.empty((ofc_variables.molNUM, self.omega_chi.size, self.omega_chi.size), dtype=np.complex)

        # ------------------------------------------------------------------------------------------------------------ #
        #                       DECLARE NEW SET OF VARIABLES FOR ALL N MOLECULES IN ENSEMBLE                           #
        # ------------------------------------------------------------------------------------------------------------ #


    def create_ofc_molecule(self, ofc_molecule, indices):
        ofc_molecule.levelsNUM = self.levelsNUM
        ofc_molecule.energies = self.energies[indices].ctypes.data_as(POINTER(c_double))
        ofc_molecule.levels = self.levels[indices].ctypes.data_as(POINTER(c_double))
        ofc_molecule.gammaMATRIX = np.ascontiguousarray(self.gammaMATRIX[indices]).ctypes.data_as(POINTER(c_double))
        ofc_molecule.muMATRIX = self.muMATRIX[indices].ctypes.data_as(POINTER(c_complex))
        ofc_molecule.polarizationINDEX = self.polarizationINDEX[indices].ctypes.data_as(POINTER(c_complex))
        ofc_molecule.polarizationMOLECULE = self.polarizationMOLECULE[indices].ctypes.data_as(POINTER(c_complex))
        ofc_molecule.chi1DIST = self.chi1DIST[indices].ctypes.data_as(POINTER(c_complex))
        ofc_molecule.chi3DIST = self.chi3DIST[indices].ctypes.data_as(POINTER(c_complex))
        ofc_molecule.chi1INDEX = self.chi1INDEX[indices].ctypes.data_as(POINTER(c_complex))
        ofc_molecule.chi3INDEX = self.chi3INDEX[indices].ctypes.data_as(POINTER(c_complex))
        ofc_molecule.probabilities = self.probabilities[indices].ctypes.data_as(POINTER(c_double))
        return

    def create_ofc_parameters(self, ofc_parameters, ofc_variables):
        ofc_parameters.excitedNUM = ofc_variables.excitedNUM
        ofc_parameters.ensembleNUM = ofc_variables.ensembleNUM
        ofc_parameters.freqNUM = len(self.frequency)
        ofc_parameters.chiNUM = ofc_variables.chiNUM
        ofc_parameters.combNUM = ofc_variables.combNUM
        ofc_parameters.resolutionNUM = ofc_variables.resolutionNUM
        ofc_parameters.basisNUM = ofc_variables.basisNUM
        ofc_parameters.frequency = self.frequency.ctypes.data_as(POINTER(c_double))
        ofc_parameters.omega_chi = self.omega_chi.ctypes.data_as(POINTER(c_double))
        ofc_parameters.combGAMMA = ofc_variables.combGAMMA
        ofc_parameters.freqDEL = ofc_variables.freqDEL
        ofc_parameters.termsNUM = ofc_variables.termsNUM
        ofc_parameters.indices = self.indices.ctypes.data_as(POINTER(c_long))
        ofc_parameters.basisINDX = self.basisINDX.ctypes.data_as(POINTER(c_long))
        ofc_parameters.modulations = np.zeros(3, dtype=int).ctypes.data_as(POINTER(c_double))
        ofc_parameters.envelopeWIDTH = ofc_variables.envelopeWIDTH
        ofc_parameters.envelopeCENTER = ofc_variables.envelopeCENTER
        ofc_parameters.frequencyMC_opt = ofc_variables.frequencyMC_opt.ctypes.data_as(POINTER(c_double))
        ofc_parameters.frequencyMC_RF1 = ofc_variables.frequencyMC_RF1.ctypes.data_as(POINTER(c_double))
        # ofc_parameters.frequencyMC_RF2 = ofc_variables.frequencyMC_RF2.ctypes.data_as(POINTER(c_double))
        return

    def calculate_ofc_system(self, ofc_variables):
        ofc_parameters = OFCParameters()
        self.create_ofc_parameters(ofc_parameters, ofc_variables)
        molENSEMBLE = [OFCMolecule() for _ in range(ofc_variables.molNUM)]

        basisRNG = int((ofc_parameters.basisNUM - (ofc_parameters.basisNUM % 2)) / 2)
        for I_ in range(-basisRNG, basisRNG + (ofc_parameters.basisNUM % 2)):
            ofc_parameters.basisINDX[0] = I_
            self.polINDX[I_] = I_
            for J_ in range(-basisRNG, basisRNG + (ofc_parameters.basisNUM % 2)):
                ofc_parameters.basisINDX[1] = J_
                self.polINDX[J_] = J_
                for K_ in range(-basisRNG, basisRNG + (ofc_parameters.basisNUM % 2)):
                    ofc_parameters.basisINDX[2] = K_
                    self.polINDX[K_] = K_
                    fig, ax = plt.subplots(nrows=ofc_variables.molNUM, ncols=2, sharex=True, sharey=True, figsize=(22, 11))
                    for molINDX in range(ofc_variables.molNUM):
                        self.create_ofc_molecule(molENSEMBLE[molINDX], molINDX)
                        for i, modulations in enumerate(list(product(*(3 * [[ofc_variables.omegaM1, ofc_variables.omegaM2, ofc_variables.omegaM3]])))):
                            if i == 5:
                                print("Molecule ", molINDX + 1, ": ", i, modulations)
                                for mINDX, nINDX, vINDX in ofc_variables.modulationINDXlist:
                                    ofc_parameters.indices[0] = mINDX
                                    ofc_parameters.indices[1] = nINDX
                                    ofc_parameters.indices[2] = vINDX
                                    ofc_parameters.modulations = np.asarray(modulations).ctypes.data_as(POINTER(c_double))
                                    mu_product = self.muMATRIX[molINDX][0, mINDX] * self.muMATRIX[molINDX][mINDX, nINDX] * \
                                                 self.muMATRIX[molINDX][nINDX, vINDX] * self.muMATRIX[molINDX][vINDX, 0]
                                    self.polarizationMOLECULE[molINDX][:] = 0.
                                    CalculateOFCResponse(molENSEMBLE[molINDX], ofc_parameters)
                                    self.polarizationMOLECULE[molINDX] *= mu_product
                                    for ensembleINDX in range(ofc_variables.ensembleNUM):
                                        if (i == 5) or (i == 7) or (i == 11) or (i == 15) or (i == 19) or (i == 21):
                                            self.polarizationEMPTY[molINDX][ensembleINDX] += self.polarizationMOLECULE[molINDX][
                                                ensembleINDX]
                                        else:
                                            self.polarizationFIELD[molINDX][ensembleINDX] += self.polarizationMOLECULE[molINDX][
                                                ensembleINDX]

                        for ensembleINDX in range(ofc_variables.ensembleNUM):
                            self.polarizationTOTALEMPTY[molINDX] += self.polarizationEMPTY[molINDX][ensembleINDX]*self.probabilities[molINDX][ensembleINDX]
                            self.polarizationTOTALFIELD[molINDX] += self.polarizationFIELD[molINDX][ensembleINDX]*self.probabilities[molINDX][ensembleINDX]


                        self.polarizationTOTALEMPTY_DIST[I_][J_][K_][molINDX] = self.polarizationTOTALEMPTY[molINDX]
                        self.polarizationTOTALFIELD_DIST[I_][J_][K_][molINDX] = self.polarizationTOTALFIELD[molINDX]

                        if ofc_variables.molNUM > 1:
                            ax[molINDX, 0].plot(self.frequency / timeFACTOR, self.polarizationTOTALEMPTY[molINDX].real, 'r')
                            ax[molINDX, 0].plot(self.frequency / timeFACTOR, self.polarizationTOTALFIELD[molINDX].real, 'k')
                            ax[molINDX, 1].plot(self.frequency / timeFACTOR, self.polarizationTOTALEMPTY[molINDX].imag, 'r')
                            ax[molINDX, 1].plot(self.frequency / timeFACTOR, self.polarizationTOTALFIELD[molINDX].imag, 'k')
                            ax[molINDX, 0].grid()
                            ax[molINDX, 1].grid()
                        else:
                            ax[0].plot(self.frequency / timeFACTOR, self.polarizationTOTALEMPTY[molINDX].real, 'r')
                            ax[0].plot(self.frequency / timeFACTOR, self.polarizationTOTALFIELD[molINDX].real, 'k')
                            ax[1].plot(self.frequency / timeFACTOR, self.polarizationTOTALEMPTY[molINDX].imag, 'r')
                            ax[1].plot(self.frequency / timeFACTOR, self.polarizationTOTALFIELD[molINDX].imag, 'k')
                            ax[0].grid()
                            ax[1].grid()

                        # self.polarizationFIELD *= 0.
                        # self.polarizationEMPTY *= 0.
                    plt.close(fig)

    def calculate_susceptibilities(self, ofc_variables):
        ofc_parameters = OFCParameters()
        self.create_ofc_parameters(ofc_parameters, ofc_variables)
        molENSEMBLE = [OFCMolecule() for _ in range(ofc_variables.molNUM)]

        for molINDX in range(ofc_variables.molNUM):
            self.create_ofc_molecule(molENSEMBLE[molINDX], molINDX)
            CalculateChi(molENSEMBLE[molINDX], ofc_parameters)


if __name__ == '__main__':

    from matplotlib.cm import get_cmap
    import pickle
    # --------------------------------------------------------- #
    #                       LIST OF CONSTANTS                   #
    # --------------------------------------------------------- #

    energyFACTOR = 1./27.211385
    timeFACTOR = 2.418884e-5
    wavelength2freqFACTOR = 1239.84
    cm_inv2evFACTOR = 1.23984e-4
    magneticFACTOR = 2.002319 * 8.7941e-2 * timeFACTOR

    vibrationalENERGYghz_1 = 2.88
    vibrationalENERGYghz_2 = 1.42
    electronicENERGYnm = 637

    # ------------------------------------------------------------------------------------------ #
    #                       MOLECULAR CONSTANTS, VARIABLES, VECTORS & MATRICES                   #
    # ------------------------------------------------------------------------------------------ #

    molNUM = 3
    levelsNUM = 6
    ensembleNUM = 1
    groundNUM = 3
    excitedNUM = levelsNUM - groundNUM

    probabilities = np.asarray([[0.10, 0.15, 0.20, 0.25, 0.20, 0.5, 0.5]]*molNUM)

    # ------------------ MOLECULAR ENERGY LEVEL STRUCTURE ------------------ #

    magnetic_fields = np.asarray([1.0 + 0.3 * i for i in range(molNUM)]) * 1e-2    # Magnetic Fields are in tesla


    def get_energies_due_to_magnetic_field(elec_gap, vib_gap1, vib_gap2, mu_field):
        """
        elec_gap: Energy Gap between electronic manifolds in nm
        vib_gap_1: Gap between ground vibrational modes (with zero magnetic field)
        vib_gap_2: Gap between excited vibrational modes (with zero magnetic field)
        mu_field: Magnetic field strength for a given molecule
        """

        elec_gap_2ev = energyFACTOR * wavelength2freqFACTOR / elec_gap
        vib_gap1_2ev = 2. * np.pi * timeFACTOR * vib_gap1 / 1000.
        vib_gap2_2ev = 2. * np.pi * timeFACTOR * vib_gap2 / 1000.

        mu_gap_2ev = magneticFACTOR * mu_field
        return [0.00, vib_gap1_2ev - mu_gap_2ev, vib_gap1_2ev + mu_gap_2ev, elec_gap_2ev,
                elec_gap_2ev + vib_gap2_2ev - mu_gap_2ev, elec_gap_2ev + vib_gap2_2ev + mu_gap_2ev]

    energies = np.empty((molNUM, levelsNUM))

    electronicENERGYthz = [electronicENERGYnm + .04 * en for en in range(-3, -3 + ensembleNUM)]
    levels = np.asarray([[get_energies_due_to_magnetic_field(elecEN, vibrationalENERGYghz_1, vibrationalENERGYghz_2, mu) for elecEN in electronicENERGYthz] for mu in magnetic_fields])

    # print(levels)
    print(levels / (2. * np.pi * timeFACTOR / 1000.))

    # ------------------------ INITIAL DENSITY MATRIX ---------------------- #
    rho_0 = np.zeros((levelsNUM, levelsNUM), dtype=np.complex)
    rho_0[0, 0] = 1 + 0j

    MUelec = [1.]*molNUM
    MUvibr = [0.01]*molNUM

    gammaFACTOR = 1.
    gammaPOPD = np.asarray([1]*molNUM) * timeFACTOR * gammaFACTOR * 1.e-9
    gammaVIBR = np.asarray([1]*molNUM) * timeFACTOR * gammaFACTOR * 2.5e-6
    gammaELEC = np.asarray([1]*molNUM) * timeFACTOR * gammaFACTOR * 10e-3

    muMATRIX = np.empty((molNUM, levelsNUM, levelsNUM), dtype=np.complex)
    for _ in range(molNUM):
        muMATRIX[_] = np.asarray(
            [
                [0.0 + 0.0j, MUvibr[_] + 0.0j, MUvibr[_] + 0.0j, MUelec[_] + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [MUvibr[_] + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, MUelec[_] + 0.0j, 0.0 + 0.0j],
                [MUvibr[_] + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, MUelec[_] + 0.0j],
                [MUelec[_] + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, MUvibr[_] + 0.0j, MUvibr[_] + 0.0j],
                [0.0 + 0.0j, MUelec[_] + 0.0j, 0.0 + 0.0j, MUvibr[_] + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, MUelec[_] + 0.0j, MUvibr[_] + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]
            ]
        )

    gammaMATRIXpopd = [np.ones((levelsNUM, levelsNUM), dtype=np.float) * gammaPOPD[i] for i in range(molNUM)]
    gammaMATRIXdephasing = [np.ones((levelsNUM, levelsNUM), dtype=np.float) * gammaVIBR[i] for i in range(molNUM)]
    for i in range(molNUM):
        np.fill_diagonal(gammaMATRIXpopd[i], 0.0)
        gammaMATRIXpopd[i] = np.tril(gammaMATRIXpopd[i]).T
        np.fill_diagonal(gammaMATRIXdephasing[i], 0.0)
        for j in range(groundNUM):
            for k in range(groundNUM, levelsNUM):
                gammaMATRIXdephasing[i][j, k] = gammaELEC[i]
                gammaMATRIXdephasing[i][k, j] = gammaELEC[i]
    gammaMATRIX = gammaMATRIXdephasing[:]
    for k in range(molNUM):
        for n in range(levelsNUM):
            for m in range(levelsNUM):
                for i in range(levelsNUM):
                    gammaMATRIX[k][n][m] += 0.5 * (gammaMATRIXpopd[k][n][i] + gammaMATRIXpopd[k][m][i])
        np.fill_diagonal(gammaMATRIX[k], 0.0)

    guessLOWER = np.zeros(ensembleNUM)
    guessUPPER = np.ones(ensembleNUM)

    # ---------------------------------------------------------------------------------------------------------------- #
    #              READ csv-DATA FILES INTO WAVELENGTH & ABSORPTION MATRICES: (SIZE) N x wavelengthNUM                 #
    # ---------------------------------------------------------------------------------------------------------------- #

    plot_colors = ['r', 'b', 'k']

    # -------------------------------------------#
    #              OFC PARAMETERS                #
    # -------------------------------------------#

    nu_0 = levels[1,0,2] - levels[0,0,2]
    nu_bar = levels[0,0,2]
    print(nu_0, nu_bar)
    rangeFREQ = np.asarray([nu_bar - nu_0, nu_bar + molNUM * nu_0])

    combNUM = 1000
    resolutionNUM = 3

    freqDEL = (molNUM + 1) * nu_0 / combNUM
    print(freqDEL)
    unit = freqDEL / 10
    omegaM1 = unit * 4
    omegaM2 = unit * 9
    omegaM3 = unit * 3

    # combGAMMA = 1e-15 * timeFACTOR
    combGAMMA = freqDEL / 100000
    termsNUM = 3
    envelopeWIDTH = 50000
    envelopeCENTER = 0
    chiNUM = 1000

    SystemArgs = dict(
        gammaMATRIXpopd=gammaMATRIXpopd,
        gammaMATRIXdephasing=gammaMATRIXdephasing,
        gammaMATRIX=gammaMATRIX,
        muMATRIX=muMATRIX,
        energies=energies,
        levels=levels,
        probabilities=probabilities,
    )

    SystemVars = ADict(
        molNUM=molNUM,
        levelsNUM=levelsNUM,
        excitedNUM=excitedNUM,
        ensembleNUM=ensembleNUM,
        threadNUM=cpu_count(),
        rho_0=rho_0,
        spectra_timeAMP=5000,
        spectra_timeDIM=1000,
        spectra_fieldAMP=8e-6,
        guessLOWER=guessLOWER,
        guessUPPER=guessUPPER,
        iterMAX=1,
        combNUM=combNUM,
        basisNUM=1,
        resolutionNUM=resolutionNUM,
        omegaM1=omegaM1,
        omegaM2=omegaM2,
        omegaM3=omegaM3,
        combGAMMA=combGAMMA,
        freqDEL=freqDEL,
        termsNUM=termsNUM,
        envelopeWIDTH=envelopeWIDTH,
        envelopeCENTER=envelopeCENTER,
        modulationINDXlist=[(3, 5, 2), (3, 4, 1)],
        chiNUM=chiNUM,
        frequencyMC_opt=np.linspace(2955., 2960., chiNUM) * timeFACTOR,
        frequencyMC_RF1=np.linspace(.01e-7, 5e-7, chiNUM),
        rangeFREQ=rangeFREQ,
    )

    if False:
        system = OFC(SystemVars, **SystemArgs)
        system.calculate_susceptibilities(SystemVars)

        np.set_printoptions(precision=10)

        fig_r, axes_r = plt.subplots(nrows=molNUM, ncols=1, sharex=True, sharey=True)
        fig_r.suptitle('log |$\chi^{(3)}$|')
        fig_i, axes_i = plt.subplots(nrows=molNUM, ncols=1, sharex=True, sharey=True)
        fig_i.suptitle('Imaginary Chi(3)')

        i = 0
        for ax in axes_r.flat:
            im_r = ax.imshow(np.log10(abs(system.chi3DIST[i][1])), extent=(
                SystemVars.frequencyMC_RF1.min() / timeFACTOR * 1000.,
                SystemVars.frequencyMC_RF1.max() / timeFACTOR * 1000.,
                SystemVars.frequencyMC_opt.max() / (timeFACTOR * 2. * np.pi),
                SystemVars.frequencyMC_opt.min() / (timeFACTOR * 2. * np.pi),), aspect='auto', cmap='hot')

            ax.set_ylabel(" Optical field (in THz)")
            ax.set_xlabel(" RF field (in GHz)")

            print(abs(system.chi3DIST[i][1].real).argmin(), abs(system.chi3DIST[i][1].real).argmax())
            print(abs(system.chi3DIST[i][1].real).min(), abs(system.chi3DIST[i][1].real).max())
            print(abs(system.chi3DIST[i][1].imag).argmin(), abs(system.chi3DIST[i][1].imag).argmax())
            print(abs(system.chi3DIST[i][1].imag).min(), abs(system.chi3DIST[i][1].imag).max())
            i += 1
        fig_r.colorbar(im_r, ax=axes_r.ravel().tolist())


        i = 0
        for ax in axes_i.flat:
            im_i = ax.imshow(np.log10(abs(system.chi3DIST[i][1].imag)), extent=(
                SystemVars.frequencyMC_RF1.min() / timeFACTOR * 1000.,
                SystemVars.frequencyMC_RF1.max() / timeFACTOR * 1000.,
                SystemVars.frequencyMC_opt.max() / (timeFACTOR * 2. * np.pi),
                SystemVars.frequencyMC_opt.min() / (timeFACTOR * 2. * np.pi),), aspect='auto', cmap='hot')
            print(abs(system.chi3DIST[i][1].imag).argmin(), abs(system.chi3DIST[i][1].imag).argmax())
            print(abs(system.chi3DIST[i][1].imag).min(), abs(system.chi3DIST[i][1].imag).max())
            i += 1

        fig_i.colorbar(im_i, ax=axes_i.ravel().tolist())
        del system

    def f_spline(x, y):
        f = interp1d(x, y, kind='quadratic')
        x_new = np.linspace(x[0], x[-1], num=5000, endpoint=True)
        y_new = f(x_new)
        return x_new, y_new

    if True:
        start = time.time()
        system = OFC(SystemVars, **SystemArgs)
        system.calculate_ofc_system(SystemVars)

        fig, ax = plt.subplots(nrows=molNUM, ncols=2, sharex=True, sharey=True)
        fig.suptitle("Gradient Response")
        fig2, ax2 = plt.subplots(nrows=molNUM, ncols=2, sharex=True, sharey=True)
        fig2.suptitle("Response")

        cmap = get_cmap("Set1")
        colors = cmap.colors

        maxPOL = np.abs(system.polarizationTOTALEMPTY).max()
        for i in range(molNUM):
            system.polarizationTOTALEMPTY[i].real /= maxPOL
            system.polarizationTOTALEMPTY[i].imag /= maxPOL

        field1, field2, field3 = plot_field_pol_params(system, SystemVars, rangeFREQ)
        field1 /= field1.max()
        field2 /= field2.max()
        field3 /= field3.max()

        for i in range(molNUM):
            diff = system.polarizationTOTALEMPTY[i] - system.polarizationTOTALEMPTY[(i + 1) % molNUM]

            ax[i,0].plot(system.frequency / (timeFACTOR * 2. * np.pi / 1000.), diff.real, color=colors[i], linewidth=.75)
            # x, y = f_spline(system.frequency[1:-1:30] / (timeFACTOR * 2. * np.pi), diff.real[1:-1:30])
            # ax[i,0].plot(x, y, 'k', linewidth=1.75)

            ax[i,1].plot(system.frequency / (timeFACTOR * 2. * np.pi / 1000.), diff.imag, color=colors[i], linewidth=.75)
            # x, y = f_spline(system.frequency[1:-1:30] / (timeFACTOR * 2. * np.pi), diff.imag[1:-1:30])
            # ax[i,1].plot(x, y, 'k', linewidth=1.75)

            render_axis(ax[i, 0], gridLINE='')
            render_axis(ax[i, 1], gridLINE='')

            ax2[i,0].plot(system.frequency / (timeFACTOR * 2. * np.pi / 1000.), system.polarizationTOTALEMPTY[i].real, 'r*-', linewidth=.75)
            # ax2[i,0].plot(system.field1FREQ / (timeFACTOR * 2. * np.pi / 1000.), field1, 'y')
            # ax2[i,0].plot(system.field2FREQ / (timeFACTOR * 2. * np.pi / 1000.), field2, 'g')
            # ax2[i,0].plot(system.field3FREQ / (timeFACTOR * 2. * np.pi / 1000.), field3, 'm')
            # x, y = f_spline(system.frequency[1:-1:30] / (timeFACTOR * 2. * np.pi), system.polarizationTOTALEMPTY[i].real[1:-1:30])
            # ax2[i,0].plot(x, y, 'k', linewidth=1.75)

            ax2[i,1].plot(system.frequency / (timeFACTOR * 2. * np.pi / 1000.), system.polarizationTOTALEMPTY[i].imag, 'b*-', linewidth=.75)
            # x, y = f_spline(system.frequency[1:-1:30] / (timeFACTOR * 2. * np.pi), system.polarizationTOTALEMPTY[i].imag[1:-1:30])
            # ax2[i,1].plot(x, y, 'k', linewidth=1.75)

            render_axis(ax2[i, 0], gridLINE='')
            render_axis(ax2[i, 1], gridLINE='')
            ax2[i, 0].set_ylim(-1.2, 1.2)
            ax2[i, 0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            ax2[i, 1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            ax[i, 0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            ax[i, 1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))


        for i in range(molNUM):
            for j in range(ensembleNUM):
                print(np.abs(system.polarizationEMPTY[i][j].imag).max())
        print("Time taken for response calculation: ", time.time() - start)


        with open('Pickle/S3.pickle', 'wb') as f:
            pickle.dump(
                {
                    "pol3field": system.polarizationTOTALFIELD,
                    "pol3empty": system.polarizationTOTALEMPTY,
                    "field1FREQ": system.field1FREQ,
                    "field2FREQ": system.field2FREQ,
                    "field3FREQ": system.field3FREQ,
                    "frequency": system.frequency
                },
                f)

        with open('Pickle/pol3_args.pickle', 'wb') as f:
            pickle.dump(
                {
                    "molNUM": molNUM,
                    "combNUM": combNUM,
                    "resolutionNUM": resolutionNUM,
                    "omegaM1": omegaM1,
                    "omegaM2": omegaM2,
                    "omegaM3": omegaM3,
                    "freqDEL": freqDEL,
                    "combGAMMA": combGAMMA,
                    "termsNUM": termsNUM,
                    "envelopeWIDTH": envelopeWIDTH,
                    "envelopeCENTER": envelopeCENTER,
                    "chiNUM": chiNUM,
                    "rangeFREQ": rangeFREQ
                },
                f)
    plt.show()
