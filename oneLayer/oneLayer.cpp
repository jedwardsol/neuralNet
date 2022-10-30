// one hidden layer

#define EIGEN_STACK_ALLOCATION_LIMIT 0

#include <cmath>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <random>
#include <fstream>
#include "include/print.h"
#include "include/thrower.h"
#include <chrono>
namespace chr=std::chrono;
#include "idx/idx.h"


constexpr int  inputLayerSize{28*28};
constexpr int  outputLayerSize{10};
constexpr int  hiddenLayerSize{32}; //{(inputLayerSize + outputLayerSize)/2};

using InputLayer   = Eigen::Matrix<double,inputLayerSize,1>;

using Weights1     = Eigen::Matrix<double,hiddenLayerSize,inputLayerSize>;
using Biases1      = Eigen::Matrix<double,hiddenLayerSize,1>;

using HiddenLayer  = Eigen::Matrix<double,hiddenLayerSize,1>;

using Weights2     = Eigen::Matrix<double,outputLayerSize,hiddenLayerSize>;
using Biases2      = Eigen::Matrix<double,outputLayerSize,1>;


using OutputLayer  = Eigen::Matrix<double,outputLayerSize,1>;

double sigmoid(double x)
{
    return 1/(1+std::exp(-x));
}


InputLayer      inputLayer{};

Weights1        weights1{};                 // 2Mb
Biases1         biases1{};      

HiddenLayer     hiddenLayer{};  

Weights2        weights2{};     
Biases2         biases2{};      

OutputLayer     outputLayer{};


template <typename M>
void write(M const &matrix, std::string const &filename)
{
    std::ofstream   file{filename,std::ios::binary};

    if(!file)
    {
        throw_system_error("open " + filename);
    }

    file.write(reinterpret_cast<char const*>(matrix.data()), matrix.rows()*matrix.cols()*sizeof(M::Scalar));

    if(!file)
    {
        throw_system_error("write " + filename);
    }
}

template <typename M>
void read(M &matrix, std::string const &filename)
{
    std::ifstream   file{filename,std::ios::binary};

    if(!file)
    {
        throw_system_error("open " + filename);
    }

    file.read(reinterpret_cast<char*>(matrix.data()), matrix.rows()*matrix.cols()*sizeof(M::Scalar));

    if(!file)
    {
        throw_system_error("read " + filename);
    }
}


void randomise()
{
    weights1.setRandom();
    biases1.setRandom();    
    weights2.setRandom();
    biases2.setRandom();

    write(weights1,"matrices\\1l_weights1");
    write(biases1, "matrices\\1l_biases1");
    write(weights2,"matrices\\1l_weights2");
    write(biases2, "matrices\\1l_biases2");
}


void costs()
{
    read(weights1,"matrices\\1l_weights1");
    read(biases1, "matrices\\1l_biases1");
    read(weights2,"matrices\\1l_weights2");
    read(biases2, "matrices\\1l_biases2");

    auto labels = idx::readLabels("datasets\\train-labels.idx1-ubyte");
    auto images = idx::readImages("datasets\\train-images.idx3-ubyte");

    if(labels.size() != images.size())
    {
        print("sizes don't match\n");
    }

    auto start = chr::steady_clock::now();
    
    for(int i=0;i<images.size();i++)
    {
        auto const  label=labels[i];        
        auto const &image=images[i];        

        OutputLayer desiredOutput{};
        desiredOutput[label]=1.0;

        for(int pixel=0;pixel<image.pixels.size();pixel++)
        {
            inputLayer[pixel]=image.pixels[pixel]/256.0;
        }

        hiddenLayer = (weights1*inputLayer +biases1).unaryExpr(&sigmoid);
        outputLayer = (weights2*hiddenLayer+biases2).unaryExpr(&sigmoid);

        int maxIndex{};
        auto maxCoeff = outputLayer.maxCoeff(&maxIndex);
        auto cost = (outputLayer-desiredOutput).array().square().sum();

        print("{} : {} cost={}\n",label,maxIndex,cost);
    }

    auto end = chr::steady_clock::now();

    print("Duration {}\n",chr::duration_cast<chr::seconds>(end-start));
}


int main(int argc, char *argv[])
try
{
    auto conOut = GetStdHandle( STD_OUTPUT_HANDLE );
    SetConsoleOutputCP( CP_UTF8 );
    std::vector<std::string>    args(argv+1,argv+argc);

    if(args.empty())
    {
        throw_runtime_error("oneLayer random|costs\n");
    }

    if(args[0]=="random")
    {
        randomise();
    }
    else if(args[0]=="costs")
    {
        costs();
    }
    else
    {
        throw_runtime_error("oneLayer random|costs\n");
    }

}
catch(std::exception const &e)
{
    print("caught {}\n",e.what());
}