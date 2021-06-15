
typedef struct ofc_parameters{
    int excitedNUM;
    int ensembleNUM;
    int freqNUM;
    int chiNUM;
    int combNUM;
    int resolutionNUM;
    int basisNUM;
    double* frequency;
    double* omega_chi;
    double combGAMMA;
    double freqDEL;
    int termsNUM;
    long* indices;
    long* basisINDX;
    double* modulations;
    double envelopeWIDTH;
    double envelopeCENTER;
    double* frequencyMC_opt;
    double* frequencyMC_RF1;
//    double* frequencyMC_RF2;
} ofc_parameters;

typedef struct ofc_molecule{
    int levelsNUM;
    double* energies;
    double* levels;
    double* gammaMATRIX;
    cmplx* muMATRIX;
    cmplx* polarizationINDEX;
    cmplx* polarizationMOLECULE;
    cmplx* chi1DIST;
    cmplx* chi3DIST;
    cmplx* chi1INDEX;
    cmplx* chi3INDEX;
    double* probabilities;
    cmplx* chi3MATRIX;
} ofc_molecule;
