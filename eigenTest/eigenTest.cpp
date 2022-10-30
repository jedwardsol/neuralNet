// https://eigen.tuxfamily.org/dox/

#include <cmath>

#include "include/print.h"

#include <Eigen/Dense>
 
using Eigen::MatrixXd;


double identity(double x)
{
    return x;
}

double sigmoid(double x)
{
    return 1/(1+std::exp(-x));
}

 
int main()
{
    MatrixXd layer
    {
        {0.1},
        {0.9},
        {0.3},
        {0.4}
    };

    MatrixXd weights
    {
        {0.1,-0.2,0.3,0.4},
        {0.5,0.6,-0.7,0.8}
    };

    MatrixXd biases
    {
        { 0.1},
        {-0.2},
    };


    MatrixXd layer1 = (weights*layer+biases).unaryExpr(&sigmoid);


    std::cout << layer1 << std::endl;
}