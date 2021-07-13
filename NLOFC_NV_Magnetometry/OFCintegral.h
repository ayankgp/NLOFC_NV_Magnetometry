#include "auxiliary.h"
#include "structures.h"
#define ENERGY_FACTOR 1. / 27.211385
#define WAVELENGTH2FREQ 1239.84

//====================================================================================================================//
//                                                                                                                    //
//                                              INTEGRALS OF SPECTROSCOPIC TERMS                                      //
//   ---------------------------------------------------------------------------------------------------------------  //
//    Given a set of modulations (\omegaM1; \omegaM2; \omegaM3) and permuted indices (m, n, v) calculate the          //
//    non-linear OFC spectroscopic integrals for each spectroscopic term occurring in the susceptibility function     //
//    \chi_3 using analytical forms developed via solving the Cauchy integral with frequencies in the upper z-plane   //
//====================================================================================================================//


//====================================================================================================================//
//                                                                                                                    //
//                                                  INTEGRAL OF TYPE A-1                                              //
//   ---------------------------------------------------------------------------------------------------------------  //
//      I1 = 1/(ABC) + 1/(ABD) + 1/(BCE) - 1/(ADE*)                                                                   //
//      where:                                                                                                        //
//      A -> {\omega} + \omegaM_i + m_i(\Delta \omega) + \Omega_b + i\tau                                             //
//      B -> \omegaM_k + m_k(\Delta \omega) + \Omega_a + i\tau                                                        //
//      C -> \omegaM_k + \omegaM_j + (m_k + m_j)(\Delta \omega) + \Omega_b + 2i\tau                                   //
//      D -> {\omega} + \omegaM_i - \omegaM_j + (m_i - m_j)(\Delta \omega) + \Omega_a + 2i\tau                        //
//      E -> -{\omega} + \omegaM_k + \omegaM_j - \omegaM_i + (m_k + m_j - m_i)(\Delta \omega) + 3i\tau                //
//                                                                                                                    //
//====================================================================================================================//

void pol3(ofc_molecule* ofc_mol, ofc_parameters* ofc_params, const cmplx wg_c, const cmplx wg_b, const cmplx wg_a, const int sign)
{
    double freqDEL = ofc_params->freqDEL;
    int termsNUM = ofc_params->termsNUM;
    double combGAMMA = ofc_params->combGAMMA;

    double omegaM_k = ofc_params->modulations[0];
    double omegaM_j = ofc_params->modulations[1];
    double omegaM_i = ofc_params->modulations[2];

    int m_k_0 = ceil((- omegaM_k - crealf(wg_a))/freqDEL);
    int m_j_0 = ceil((- omegaM_k - omegaM_j - crealf(wg_b) )/freqDEL) - m_k_0;

    for(int out_i = 0; out_i < ofc_params->freqNUM; out_i++)
    {
        const double omega = ofc_params->frequency[out_i];
        int m_i_0 = m_k_0 + m_j_0 - ceil((omega - omegaM_k - omegaM_j + omegaM_i)/freqDEL);
        cmplx result = 0. + 0. * I;
        for(int m_i = m_i_0 - termsNUM; m_i < m_i_0 + termsNUM; m_i++)
        {
//            double c_i = exp(-pow(m_i + ofc_params->envelopeCENTER, 2) / (2. * powf(ofc_params->envelopeWIDTH, 2.)));
//            c_i = 1.;
            const cmplx term_A = omega + omegaM_i + m_i * freqDEL + wg_b + combGAMMA * I;
            for(int m_j = m_j_0 - termsNUM; m_j < m_j_0 + termsNUM; m_j++)
            {
//                double c_j = exp(-pow(m_j + ofc_params->envelopeCENTER, 2) / (2. * powf(ofc_params->envelopeWIDTH, 2.)));
//                c_j = 1.;
                const cmplx term_D = omega + omegaM_i - omegaM_j + (m_i - m_j) * freqDEL + wg_a + 2 * I * combGAMMA;
                for(int m_k = m_k_0 - termsNUM; m_k < m_k_0 + termsNUM; m_k++)
                {
//                    double c_k = exp(-pow(m_k + ofc_params->envelopeCENTER, 2) / (2. * powf(ofc_params->envelopeWIDTH, 2.)));
//                    c_k = 1.;
                    const cmplx term_B = omegaM_k + m_k * freqDEL + wg_a + combGAMMA * I;
                    const cmplx term_C = omegaM_k + omegaM_j + (m_k + m_j) * freqDEL +  wg_b + 2 * I * combGAMMA;
                    const cmplx term_E = omega - (omegaM_k + omegaM_j - omegaM_i) - (m_k + m_j - m_i) * freqDEL + 3 * I * combGAMMA;
                    const cmplx term_E_star = - omega + (omegaM_k + omegaM_j - omegaM_i) + (m_k + m_j - m_i) * freqDEL + 3 * I * combGAMMA;
//                    result += c_i * c_j * c_k * (1./(term_A * term_D * term_E_star) + (1./(term_B * term_C * term_E)));
                    result += (1./(term_A * term_D * term_E_star) + (1./(term_B * term_C * term_E)));
                }

            }
        }

        ofc_mol->polarizationINDEX[out_i] += M_PI*M_PI*I*sign*result/(omega + wg_c);
//        printf("%d %g + I %g \n", out_i, creal(ofc_mol->polarizationINDEX[out_i]), cimag(ofc_mol->polarizationINDEX[out_i]));
    }


}

void CalculatePol3Response(ofc_molecule* ofc_mol, ofc_parameters* ofc_params)
{
    int levelsNUM;
    long l, m, n, v;

    levelsNUM = ofc_mol->levelsNUM;
    l = 0;
    m = ofc_params->indices[0];
    n = ofc_params->indices[1];
    v = ofc_params->indices[2];
    printf("%ld, %ld, %ld \n", m, n, v);

    cmplx wg_nl = ofc_mol->energies[n] - ofc_mol->energies[l] + I * ofc_mol->gammaMATRIX[n * levelsNUM + l];
    cmplx wg_vl = ofc_mol->energies[v] - ofc_mol->energies[l] + I * ofc_mol->gammaMATRIX[v * levelsNUM + l];
    cmplx wg_ml = ofc_mol->energies[m] - ofc_mol->energies[l] + I * ofc_mol->gammaMATRIX[m * levelsNUM + l];
    cmplx wg_nv = ofc_mol->energies[n] - ofc_mol->energies[v] + I * ofc_mol->gammaMATRIX[n * levelsNUM + v];
    cmplx wg_mv = ofc_mol->energies[m] - ofc_mol->energies[v] + I * ofc_mol->gammaMATRIX[m * levelsNUM + v];
    cmplx wg_vm = ofc_mol->energies[v] - ofc_mol->energies[m] + I * ofc_mol->gammaMATRIX[v * levelsNUM + m];
    cmplx wg_vn = ofc_mol->energies[v] - ofc_mol->energies[n] + I * ofc_mol->gammaMATRIX[v * levelsNUM + n];
    cmplx wg_mn = ofc_mol->energies[m] - ofc_mol->energies[n] + I * ofc_mol->gammaMATRIX[m * levelsNUM + n];
    cmplx wg_nm = ofc_mol->energies[n] - ofc_mol->energies[m] + I * ofc_mol->gammaMATRIX[n * levelsNUM + m];

    //==========================================================================================//
    //  THE FOLLOWING 8 CALLS ARE FOR THE 8 SPECTROSCOPIC TERMS: (a1), (a2), ...., (d1), (d2)   //                                                                         //
    //==========================================================================================//

    pol3(ofc_mol, ofc_params, -conj(wg_vl), -conj(wg_nl), -conj(wg_ml), -1);
//    pol3(ofc_mol, ofc_params, -conj(wg_nv), -conj(wg_mv),        wg_vl,  1);
//    pol3(ofc_mol, ofc_params, -conj(wg_nv),        wg_vm, -conj(wg_ml),  1);
//    pol3(ofc_mol, ofc_params, -conj(wg_mn),        wg_nl,        wg_vl, -1);
//    pol3(ofc_mol, ofc_params,        wg_vn, -conj(wg_nl), -conj(wg_ml),  1);
//    pol3(ofc_mol, ofc_params,        wg_nm, -conj(wg_mv),        wg_vl, -1);
//    pol3(ofc_mol, ofc_params,        wg_nm,        wg_vm, -conj(wg_ml), -1);
//    pol3(ofc_mol, ofc_params,        wg_ml,        wg_nl,        wg_vl,  1);

//    pol3(ofc_mol, ofc_params, conj(wg_vl), conj(wg_nl), conj(wg_ml), -1);
//    pol3(ofc_mol, ofc_params, conj(wg_nv), conj(wg_mv),        -wg_vl,  1);
//    pol3(ofc_mol, ofc_params, conj(wg_nv),        -wg_vm, conj(wg_ml),  1);
//    pol3(ofc_mol, ofc_params, conj(wg_mn),        -wg_nl,        -wg_vl, -1);
//    pol3(ofc_mol, ofc_params,        -wg_vn, conj(wg_nl), conj(wg_ml),  1);
//    pol3(ofc_mol, ofc_params,        -wg_nm, conj(wg_mv),        -wg_vl, -1);
//    pol3(ofc_mol, ofc_params,        -wg_nm,        -wg_vm, conj(wg_ml), -1);
//    pol3(ofc_mol, ofc_params,        -wg_ml,        -wg_nl,        -wg_vl,  1);

    //========== TERMS CORRESPONDING TO Chi^(3)(w2, w1, w3) ==============//
//    pol3(ofc_mol, ofc_params, -conj(wg_vl), -conj(wg_ml), -conj(wg_nl), -1);
//    pol3(ofc_mol, ofc_params, -conj(wg_nv),        wg_vl, -conj(wg_mv),  1);
//    pol3(ofc_mol, ofc_params, -conj(wg_nv), -conj(wg_ml),        wg_vm,  1);
//    pol3(ofc_mol, ofc_params, -conj(wg_mn),        wg_vl,        wg_nl, -1);
//    pol3(ofc_mol, ofc_params,        wg_vn, -conj(wg_ml), -conj(wg_nl),  1);
//    pol3(ofc_mol, ofc_params,        wg_nm,        wg_vl, -conj(wg_mv), -1);
//    pol3(ofc_mol, ofc_params,        wg_nm, -conj(wg_ml),        wg_vm, -1);
//    pol3(ofc_mol, ofc_params,        wg_ml,        wg_vl,        wg_nl,  1);

}

void Chi1(ofc_molecule* ofc_mol, ofc_parameters* ofc_params)
{
    int m, n, v, y, z, l, levelsNUM;

    levelsNUM = ofc_mol->levelsNUM;
    l = 0;
    m = 1;
    n = 2;
    v = 3;
    y = 4;
    z = 5;

    cmplx wg_ml = ofc_mol->energies[m] - ofc_mol->energies[l] + I * ofc_mol->gammaMATRIX[m * levelsNUM + l];
    cmplx wg_nl = ofc_mol->energies[n] - ofc_mol->energies[l] + I * ofc_mol->gammaMATRIX[n * levelsNUM + l];
    cmplx wg_vl = ofc_mol->energies[v] - ofc_mol->energies[l] + I * ofc_mol->gammaMATRIX[v * levelsNUM + l];
    cmplx wg_yl = ofc_mol->energies[y] - ofc_mol->energies[l] + I * ofc_mol->gammaMATRIX[y * levelsNUM + l];
    cmplx wg_zl = ofc_mol->energies[z] - ofc_mol->energies[l] + I * ofc_mol->gammaMATRIX[z * levelsNUM + l];

    cmplx mu_ml = ofc_mol->muMATRIX[m * levelsNUM + l];
    cmplx mu_lm = ofc_mol->muMATRIX[l * levelsNUM + m];
    cmplx mu_nl = ofc_mol->muMATRIX[n * levelsNUM + l];
    cmplx mu_ln = ofc_mol->muMATRIX[l * levelsNUM + n];
    cmplx mu_vl = ofc_mol->muMATRIX[v * levelsNUM + l];
    cmplx mu_lv = ofc_mol->muMATRIX[l * levelsNUM + v];
    cmplx mu_yl = ofc_mol->muMATRIX[y * levelsNUM + l];
    cmplx mu_ly = ofc_mol->muMATRIX[l * levelsNUM + y];
    cmplx mu_zl = ofc_mol->muMATRIX[z * levelsNUM + l];
    cmplx mu_lz = ofc_mol->muMATRIX[l * levelsNUM + z];

    //==========================================================================================//
    //  THE FOLLOWING 8 CALLS ARE FOR THE 8 SPECTROSCOPIC TERMS: (a1), (a2), ...., (d1), (d2)   //                                                                         //
    //==========================================================================================//

    for(int out_i = 0; out_i < ofc_params->chiNUM; out_i++)
    {
        const double omega = ofc_params->omega_chi[out_i];
        cmplx result = 0. + 0. * I;
        {
            result += mu_lm * mu_ml * (1./(conj(wg_ml) - omega) + 1./(wg_ml + omega));
            result += mu_ln * mu_nl * (1./(conj(wg_nl) - omega) + 1./(wg_nl + omega));
            result += mu_lv * mu_vl * (1./(conj(wg_vl) - omega) + 1./(wg_vl + omega));
            result += mu_ly * mu_yl * (1./(conj(wg_yl) - omega) + 1./(wg_yl + omega));
            result += mu_lz * mu_zl * (1./(conj(wg_zl) - omega) + 1./(wg_zl + omega));
        }
        ofc_mol->chi1INDEX[out_i] += result;
    }

}


void Chi3terms(ofc_molecule* ofc_mol, ofc_parameters* ofc_params, int m, int n, int v)
{
    int l = 0;
    int levelsNUM;
    double omega_p, omega_q, omega_r, omega_pqr, omega_pq;
    levelsNUM = ofc_mol->levelsNUM;
    cmplx wg_ml = ofc_mol->energies[m] - ofc_mol->energies[l] + I * ofc_mol->gammaMATRIX[m * levelsNUM + l];
    cmplx wg_nl = ofc_mol->energies[n] - ofc_mol->energies[l] + I * ofc_mol->gammaMATRIX[n * levelsNUM + l];
    cmplx wg_vl = ofc_mol->energies[v] - ofc_mol->energies[l] + I * ofc_mol->gammaMATRIX[v * levelsNUM + l];
    cmplx wg_nv = ofc_mol->energies[n] - ofc_mol->energies[v] + I * ofc_mol->gammaMATRIX[n * levelsNUM + v];
    cmplx wg_mv = ofc_mol->energies[m] - ofc_mol->energies[v] + I * ofc_mol->gammaMATRIX[m * levelsNUM + v];
    cmplx wg_vm = ofc_mol->energies[v] - ofc_mol->energies[m] + I * ofc_mol->gammaMATRIX[v * levelsNUM + m];
    cmplx wg_vn = ofc_mol->energies[v] - ofc_mol->energies[n] + I * ofc_mol->gammaMATRIX[v * levelsNUM + n];
    cmplx wg_mn = ofc_mol->energies[m] - ofc_mol->energies[n] + I * ofc_mol->gammaMATRIX[m * levelsNUM + n];
    cmplx wg_nm = ofc_mol->energies[n] - ofc_mol->energies[m] + I * ofc_mol->gammaMATRIX[n * levelsNUM + m];

//    printf("%g + i %g \n", creal(wg_ml), cimag(wg_ml));
//    printf("%g + i %g \n", creal(wg_nl), cimag(wg_nl));
//    printf("%g + i %g \n", creal(wg_vl), cimag(wg_vl));
//    printf("%g + i %g \n", creal(wg_nv), cimag(wg_nv));
//    printf("%g + i %g \n", creal(wg_mv), cimag(wg_mv));
//    printf("%g + i %g \n", creal(wg_vm), cimag(wg_vm));
//    printf("%g + i %g \n", creal(wg_vn), cimag(wg_vn));
//    printf("%g + i %g \n", creal(wg_mn), cimag(wg_mn));
//    printf("%g + i %g \n \n", creal(wg_nm), cimag(wg_nm));

    for(int out_i = 0; out_i < ofc_params->chiNUM; out_i++)
    {
//        omega_p = ENERGY_FACTOR * WAVELENGTH2FREQ / ofc_params->frequencyMC[out_i * 3 + 0];
//        omega_q = ENERGY_FACTOR * WAVELENGTH2FREQ / ofc_params->frequencyMC[out_i * 3 + 1];
//        omega_r = ENERGY_FACTOR * WAVELENGTH2FREQ / ofc_params->frequencyMC[out_i * 3 + 2];
        for(int out_j = 0; out_j < ofc_params->chiNUM; out_j++)
        {
            omega_q = ofc_params->frequencyMC_opt[out_i];
            omega_p = ofc_params->frequencyMC_RF1[out_j];
            omega_r = omega_p;

            omega_pqr = omega_p + omega_q - omega_r;
            omega_pq = omega_p + omega_q;

            cmplx result = 0. + 0. * I;
            result += 1./((conj(wg_vl) - omega_pqr) * (conj(wg_nl) - omega_pq) * (conj(wg_ml) - omega_p));     // (a_1)
            result += 1./((conj(wg_nv) - omega_pqr) * (conj(wg_mv) - omega_pq) * (wg_vl + omega_p));     // (a_2)
            result += 1./((conj(wg_nv) - omega_pqr) * (wg_vm + omega_pq) * (conj(wg_ml) - omega_p));     // (b_1)
            result += 1./((conj(wg_mn) - omega_pqr) * (wg_nl + omega_pq) * (wg_vl + omega_p));     // (b_2)
            result += 1./((wg_vn + omega_pqr) * (conj(wg_nl) - omega_pq) * (conj(wg_ml) - omega_p));     // (c_1)
            result += 1./((wg_nm + omega_pqr) * (conj(wg_mv) - omega_pq) * (wg_vl + omega_p));     // (c_2)
            result += 1./((wg_nm + omega_pqr) * (wg_vm + omega_pq) * (conj(wg_ml) - omega_p));     // (d_1)
            result += 1./((wg_ml + omega_pqr) * (wg_nl + omega_pq) * (wg_vl + omega_p));     // (d_2)
            ofc_mol->chi3INDEX[out_i * ofc_params->chiNUM + out_j] += result;
//            ofc_mol->chi3INDEX[out_i * ofc_params->chiNUM + out_j] = result;
        }
    }
}


void Chi3(ofc_molecule* ofc_mol, ofc_parameters* ofc_params)
{
    long m, n, v, l;

//    Chi3terms(ofc_mol, ofc_params, 1, 2, 3);
//    Chi3terms(ofc_mol, ofc_params, 1, 3, 2);
//    Chi3terms(ofc_mol, ofc_params, 2, 1, 3);
//    Chi3terms(ofc_mol, ofc_params, 2, 3, 1);
//    Chi3terms(ofc_mol, ofc_params, 3, 1, 2);
//    Chi3terms(ofc_mol, ofc_params, 3, 2, 1);
    Chi3terms(ofc_mol, ofc_params, 2, 5, 3);
//    Chi3terms(ofc_mol, ofc_params, 2, 3, 5);
//    Chi3terms(ofc_mol, ofc_params, 3, 2, 5);
//    Chi3terms(ofc_mol, ofc_params, 3, 5, 2);
//    Chi3terms(ofc_mol, ofc_params, 5, 2, 3);
//    Chi3terms(ofc_mol, ofc_params, 5, 3, 2);
    Chi3terms(ofc_mol, ofc_params, 1, 4, 3);
//    Chi3terms(ofc_mol, ofc_params, 1, 3, 4);
//    Chi3terms(ofc_mol, ofc_params, 3, 1, 4);
//    Chi3terms(ofc_mol, ofc_params, 3, 4, 1);
//    Chi3terms(ofc_mol, ofc_params, 4, 1, 3);
//    Chi3terms(ofc_mol, ofc_params, 4, 3, 1);


}