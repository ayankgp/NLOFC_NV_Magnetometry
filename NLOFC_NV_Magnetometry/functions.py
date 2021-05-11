import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


def get_experimental_spectra(molecule_file, wavelengthMIN, wavelengthMAX, wavelengthNUM):
    """
    Obtain experimental spectral data
    :param wavelengthMIN: Minimum of interpolated wavelength range
    :param wavelengthMAX: Maximum of interpolated wavelength range
    :param wavelengthNUM: Number of points in interpolation
    :param molecule_file: file with experimental spectra of a given molecule
    :return: wavelength, normalized absorption of molecule
    """
    # with open(molecule_file, encoding='utf16') as f:
    with open(molecule_file) as f:
        data = pd.read_csv(f, delimiter=',')

    wavelength = data.values[:, 0]
    absorption = data.values[:, 1]

    func = interp1d(wavelength, absorption, kind='quadratic')
    wavelength = np.linspace(wavelengthMIN, wavelengthMAX, wavelengthNUM)
    absorption = func(wavelength)
    absorption -= absorption.min()
    absorption *= 100. / absorption.max()
    absorption = savgol_filter(absorption, 5, 3)

    return wavelength, absorption


# ==================================================================================================================== #
#                                                                                                                      #
#                                                  FUNCTIONS FOR PLOTTING                                              #
#   ---------------------------------------------------------------------------------------------------------------    #
# ==================================================================================================================== #


def render_axis(axis, labelSIZE='xx-large', labelCOLOR='k', gridCOLOR='r', gridLINE='--', rotation=0):
    """
    Style plots for better representation
    :param rotation: angle of rotation for tick-labels
    :param axis: axis to be rendered
    :param labelSIZE: size of ticks in plot
    :param labelCOLOR: color code for labels
    :param gridCOLOR: color code for grid lines
    :param gridLINE: line style for grid lines
    :return:
    """
    """
    Style plots for better representation
    :param axis: axes class of plot
    """
    plt.rc('font', weight='bold')
    axis.tick_params(bottom=True, top=True, left=True, right=True)
    axis.get_xaxis().set_tick_params(which='both', direction='in', width=1.25, labelrotation=0, labelsize=labelSIZE)
    axis.get_yaxis().set_tick_params(
        which='both', direction='in', width=1.25, labelcolor=labelCOLOR, labelsize=labelSIZE)
    axis.grid(color=gridCOLOR, linestyle=gridLINE, linewidth=0.5, alpha=0.5, b=None, which='both', axis='both')

    for label in axis.get_xmajorticklabels():
        label.set_rotation(rotation)
        label.set_horizontalalignment("right")
    return


def plot_dynamics(axis, x, y, xlabel, ylabel):
    """
    Plot y vs. x with labels ylabel and xlabel on axis.
    :param axis: axis for plotting dynamics
    :param x: x-axis variable
    :param y: y-axis variable
    :param xlabel: label for x-axis
    :param ylabel: label for y-axis
    :return:
    """
    return


def nonuniform_frequency_range_3(params):
    """
    Generating frequency axis
    :param params:
    :param range1:
    :param range2:
    :return:
    """
    range1 = params.rangeFREQ[0]
    range2 = params.rangeFREQ[1]
    timeFACTOR = 2.418884e-5

    pointsFREQpolarization = np.linspace(range1, range2, 10*params.combNUM + 1)[:, np.newaxis]
    pointsFREQcomb = np.linspace(range1, range2, 10*params.combNUM + 1)[:, np.newaxis]
    resolution = np.linspace(-0.001 * params.freqDEL, 0.001 * params.freqDEL, params.resolutionNUM)

    frequency_123 = params.omegaM1 + params.omegaM2 - params.omegaM3 + pointsFREQpolarization + resolution
    # frequency_123 = params.omegaM1 + params.omegaM2 - params.omegaM3 + pointsFREQpolarization
    field1FREQ = params.omegaM1 + pointsFREQcomb + resolution
    field2FREQ = params.omegaM2 + pointsFREQcomb + resolution
    field3FREQ = params.omegaM3 + pointsFREQcomb + resolution

    # frequency = np.sort(np.hstack([frequency_123.flatten(), field1FREQ.flatten(), field2FREQ.flatten(), field3FREQ.flatten()]))
    frequency = np.sort(np.hstack([frequency_123.flatten()]))
    print(frequency)
    field1FREQ = np.ascontiguousarray(field1FREQ.flatten())
    field2FREQ = np.ascontiguousarray(field2FREQ.flatten())
    field3FREQ = np.ascontiguousarray(field3FREQ.flatten())

    return frequency, frequency_123.flatten(), field1FREQ, field2FREQ, field3FREQ


def plot_field_pol_params(system, SystemVars, rangeFREQ):
    range1 = rangeFREQ[0]
    range2 = rangeFREQ[1]
    omega1MOD = system.field1FREQ[:, np.newaxis]
    omega2MOD = system.field2FREQ[:, np.newaxis]
    omega3MOD = system.field3FREQ[:, np.newaxis]

    omegaCOMB = np.linspace(range1, range2, SystemVars.combNUM + 1)[np.newaxis, :]

    field1 = (SystemVars.combGAMMA / (
            (omega1MOD - SystemVars.omegaM1 - omegaCOMB) ** 2 + SystemVars.combGAMMA ** 2)).sum(axis=1)
    field2 = (SystemVars.combGAMMA / (
            (omega2MOD - SystemVars.omegaM2 - omegaCOMB) ** 2 + SystemVars.combGAMMA ** 2)).sum(axis=1)
    field3 = (SystemVars.combGAMMA / (
            (omega3MOD - SystemVars.omegaM3 - omegaCOMB) ** 2 + SystemVars.combGAMMA ** 2)).sum(axis=1)

    return field1, field2, field3