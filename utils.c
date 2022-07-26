#include "wallii.h"

/* uniform distribution, (0..1] */
static double drand()
{
    return (rand()+1.0)/(RAND_MAX+1.0);
}

/* normal distribution, centered on 0, std dev 1 */
double random_normal()
{
    return sqrt(-2*log(drand())) * cos(2*M_PI*drand());
}
