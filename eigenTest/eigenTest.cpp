// https://eigen.tuxfamily.org/dox/

#include <cmath>

#include "include/print.h"

#include <Eigen/Dense>
 
using Eigen::MatrixXd;
using Eigen::Matrix;


double identity(double x)
{
    return x;
}

double sigmoid(double x)
{
    return 1/(1+std::exp(-x));
}

 
void dynamic()
{
    MatrixXd weights
    {
        {0.1,-0.2,0.3,0.4},
        {0.5,0.6,-0.7,0.8}
    };

    MatrixXd layer
    {
        {0.1},
        {0.9},
        {0.3},
        {0.4}
    };


    MatrixXd biases
    {
        { 0.1},
        {-0.2},
    };


    MatrixXd layer1 = (weights*layer+biases).unaryExpr(&sigmoid);


    std::cout << layer1 << std::endl;
}

void fixed()
{
    Matrix<double,2,4> weights
    {
        {0.1,-0.2,0.3,0.4},
        {0.5,0.6,-0.7,0.8}
    };

    Matrix<double,4,1> layer
    {
        {0.1},
        {0.9},
        {0.3},
        {0.4}
    };


    Matrix<double,2,1> biases
    {
        { 0.1},
        {-0.2},
    };


    Matrix<double,2,1> layer1 = (weights*layer+biases).unaryExpr(&sigmoid);


    std::cout << layer1 << std::endl;
}

int main()
{
    dynamic();
    fixed();

}