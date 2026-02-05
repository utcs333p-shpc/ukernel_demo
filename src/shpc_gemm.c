#include "assignment.h"


void shpc_dgemm( int m, int n, int k,                                            
        double *A, int rsA, int csA,                                
        double *B, int rsB, int csB,                                
        double *C, int rsC, int csC )
{
    
    shpc_ukernel_4x4(k, A, rsA, csA, B, rsB, csB, C, rsC, csC);    

}

void shpc_ukernel_4x4( int k,                                            
        double *A, int rsA, int csA,                                
        double *B, int rsB, int csB,                                
        double *C, int rsC, int csC )
{
    
    /* Declare vector registers to hold 4x4 C and load them */
    __m256d gamma_0123_0 = _mm256_loadu_pd( &C[0*rsC + 0*csC] );
    __m256d gamma_0123_1 = _mm256_loadu_pd( &C[0*rsC + 1*csC] );
    __m256d gamma_0123_2 = _mm256_loadu_pd( &C[0*rsC + 2*csC] );
    __m256d gamma_0123_3 = _mm256_loadu_pd( &C[0*rsC + 3*csC] );

    for ( int p=0; p<k; p++ )
    {
        /* Declare vector register for load/broadcasting beta( p,j ) */
        __m256d beta_p_j;

        /* Declare a vector register to hold the current column of A and load
           it with the four elements of that column. */
        __m256d alpha_0123_p = _mm256_loadu_pd( &A[ 0*rsA + p*csA ] );

        /* Load/broadcast beta( p,0 ). */
        beta_p_j = _mm256_broadcast_sd( &B[ p*rsB + 0*csB ] );

        /* update the first column of C with the current column of A times
           beta ( p,0 ) */
        gamma_0123_0 = _mm256_fmadd_pd( alpha_0123_p, beta_p_j, gamma_0123_0 );

    }

    /* Store the updated results */
    _mm256_storeu_pd( &C[0*rsC + 0*csC], gamma_0123_0 );
    _mm256_storeu_pd( &C[0*rsC + 1*csC], gamma_0123_1 );
    _mm256_storeu_pd( &C[0*rsC + 2*csC], gamma_0123_2 );
    _mm256_storeu_pd( &C[0*rsC + 3*csC], gamma_0123_3 );    

}
   
