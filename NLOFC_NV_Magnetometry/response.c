//===================================================================//
//                                                                   //
//              Linear Spectra ----> Molecular Parameters            //
//   Molecular Spectra + Field Parameters ----> Nonlinear Response   //
//      Optimal Nonlinear Response ----> Optimal Field parameters    //
//                                                                   //
//                @author  A. Chattopadhyay                          //
//    @affiliation Princeton University, Dept. of Chemistry          //
//           @version Updated last on Dec 14 2018                    //
//                                                                   //
//===================================================================//

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <nlopt.h>
#include <time.h>
#include <omp.h>
#include "OFCintegral.h"
#define ERROR_BOUND 1.0E-8
#define NLOPT_XTOL 1.0E-6

void copy_ofc_molecule(ofc_molecule* original, ofc_molecule* copy, ofc_parameters* ofc_params)
//-------------------------------------------------------------------//
//    MAKING A DEEP COPY OF AN INSTANCE OF THE MOLECULE STRUCTURE    //
//-------------------------------------------------------------------//
{
    int ensembleNUM = ofc_params->ensembleNUM;
    int levelsNUM = original->levelsNUM;
    int freqNUM = ofc_params->freqNUM;
    int chiNUM = ofc_params->chiNUM;

    copy->levelsNUM = original->levelsNUM;
    copy->energies = (double*)malloc(levelsNUM*sizeof(double));
    copy->gammaMATRIX = (double*)malloc(levelsNUM*levelsNUM*sizeof(double));
    copy->muMATRIX = (cmplx*)malloc(levelsNUM*levelsNUM*sizeof(cmplx));
    copy->polarizationINDEX = (cmplx*)malloc(freqNUM*sizeof(cmplx));
    copy->polarizationMOLECULE = (cmplx*)malloc(ensembleNUM*freqNUM*sizeof(cmplx));
    copy->chi1DIST = (cmplx*)malloc(ensembleNUM*chiNUM*sizeof(cmplx));
    copy->chi3DIST = (cmplx*)malloc(ensembleNUM*chiNUM*chiNUM*sizeof(cmplx));
    copy->chi1INDEX = (cmplx*)malloc(chiNUM*sizeof(cmplx));
    copy->chi3INDEX = (cmplx*)malloc(chiNUM*chiNUM*sizeof(cmplx));
    copy->probabilities = (double*)malloc(ensembleNUM*sizeof(double));


    memset(copy->energies, 0, original->levelsNUM*sizeof(double));
    memcpy(copy->gammaMATRIX, original->gammaMATRIX, levelsNUM*levelsNUM*sizeof(double));
    memcpy(copy->muMATRIX, original->muMATRIX, levelsNUM*levelsNUM*sizeof(cmplx));
    memcpy(copy->polarizationINDEX, original->polarizationINDEX, freqNUM*sizeof(cmplx));
    memcpy(copy->polarizationMOLECULE, original->polarizationMOLECULE, ensembleNUM*freqNUM*sizeof(cmplx));
    memcpy(copy->chi1DIST, original->chi1DIST, ensembleNUM*chiNUM*sizeof(cmplx));
    memcpy(copy->chi3DIST, original->chi3DIST, ensembleNUM*chiNUM*chiNUM*sizeof(cmplx));
    memcpy(copy->chi1INDEX, original->chi1INDEX, chiNUM*sizeof(cmplx));
    memcpy(copy->chi3INDEX, original->chi3INDEX, chiNUM*chiNUM*sizeof(cmplx));
    memcpy(copy->probabilities, original->probabilities, ensembleNUM*sizeof(double));
}


void free_ofc_molecule(ofc_molecule* mol)
//-------------------------------------------------------------------//
//    MAKING A DEEP COPY OF AN INSTANCE OF THE MOLECULE STRUCTURE    //
//-------------------------------------------------------------------//
{
    free(mol->energies);
    free(mol->gammaMATRIX);
    free(mol->muMATRIX);
    free(mol->polarizationINDEX);
    free(mol->polarizationMOLECULE);
    free(mol->chi1DIST);
    free(mol->chi3DIST);
    free(mol->chi1INDEX);
    free(mol->chi3INDEX);
    free(mol->probabilities);
    free(mol);
}


void CalculateOFCResponse_C(ofc_molecule* ofc_mol, ofc_parameters* ofc_params)
//------------------------------------------------------------//
//          CALCULATING OFC RESPONSE FOR MOLECULE             //
//------------------------------------------------------------//
{
    // ---------------------------------------------------------------------- //
    //      UPDATING THE PURE DEPHASING MATRIX & ENERGIES FOR MOLECULE        //
    // ---------------------------------------------------------------------- //

    ofc_molecule** ensemble = (ofc_molecule**)malloc(ofc_params->ensembleNUM * sizeof(ofc_molecule*));
    for(int i=0; i<ofc_params->ensembleNUM; i++)
    {
        ensemble[i] = (ofc_molecule*)malloc(sizeof(ofc_molecule));
        copy_ofc_molecule(ofc_mol, ensemble[i], ofc_params);
        for(int j=0; j<ofc_mol->levelsNUM; j++)
        {
//            ensemble[i]->energies[j] = ofc_mol->levels[ofc_params->excitedNUM*i+j];
            ensemble[i]->energies[j] = ofc_mol->levels[ofc_mol->levelsNUM * i + j];
        }
        print_double_vec(ensemble[i]->energies, ofc_mol->levelsNUM);
    }


    // ---------------------------------------------------------------------- //
    //                   CREATING THE ENSEMBLE OF MOLECULES                   //
    // ---------------------------------------------------------------------- //

    #pragma omp parallel for num_threads(11)
    for(int j=0; j<ofc_params->ensembleNUM; j++)
    {
        CalculatePol3Response(ensemble[j], ofc_params);
        for(int i=0; i<ofc_params->freqNUM; i++)
        {
            ofc_mol->polarizationMOLECULE[j * ofc_params->freqNUM + i] = ensemble[j]->polarizationINDEX[i];
        }
        free_ofc_molecule(ensemble[j]);
    }
}


void CalculateChi_C(ofc_molecule* ofc_mol, ofc_parameters* ofc_params)
//------------------------------------------------------------//
//          CALCULATING Chi1 RESPONSE FOR MOLECULE            //
//------------------------------------------------------------//
{
    ofc_molecule** ensemble = (ofc_molecule**)malloc(ofc_params->ensembleNUM * sizeof(ofc_molecule*));
    for(int i=0; i<ofc_params->ensembleNUM; i++)
    {
        ensemble[i] = (ofc_molecule*)malloc(sizeof(ofc_molecule));
        copy_ofc_molecule(ofc_mol, ensemble[i], ofc_params);
        for(int j=0; j<ofc_mol->levelsNUM; j++)
        {
            ensemble[i]->energies[j] = ofc_mol->levels[ofc_mol->levelsNUM * i + j];
        }
    }

    // ---------------------------------------------------------------------- //
    //                   CREATING THE ENSEMBLE OF MOLECULES                   //
    // ---------------------------------------------------------------------- //


    for(int j=0; j<ofc_params->ensembleNUM; j++)
    {
        Chi1(ensemble[j], ofc_params);
        Chi3(ensemble[j], ofc_params);
        for(int i=0; i<ofc_params->chiNUM; i++)
        {
            ofc_mol->chi1DIST[j * ofc_params->chiNUM + i] = ensemble[j]->chi1INDEX[i];
            for(int k=0; k<ofc_params->chiNUM; k++)
            {
                ofc_mol->chi3DIST[j * ofc_params->chiNUM * ofc_params->chiNUM + i * ofc_params->chiNUM + k] = ensemble[j]->chi3INDEX[i * ofc_params->chiNUM + k];
            }
        }
        free_ofc_molecule(ensemble[j]);
    }
}