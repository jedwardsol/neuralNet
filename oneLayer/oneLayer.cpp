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
#include "include/matrixIO.h"

constexpr int  inputLayerSize{28*28};
constexpr int  outputLayerSize{10};
constexpr int  hiddenLayerSize{(inputLayerSize + outputLayerSize)/2};

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

double normalisePixel(uint8_t c)
{
    return c/256.0;
}

Weights1        weights1{};                
Biases1         biases1{};      
Weights2        weights2{};     
Biases2         biases2{};      


void randomise()
{
    weights1.setRandom();
//  biases1.setRandom();    
    weights2.setRandom();
//  biases2.setRandom();

    write(weights1,"matrices\\1l_weights1");
    write(biases1, "matrices\\1l_biases1");
    write(weights2,"matrices\\1l_weights2");
    write(biases2, "matrices\\1l_biases2");
}


void analyse(std::string const &labelFile,std::string const &imageFile,bool checkResult)
{
    read(weights1,"matrices\\1l_weights1");
    read(biases1, "matrices\\1l_biases1");
    read(weights2,"matrices\\1l_weights2");
    read(biases2, "matrices\\1l_biases2");

    auto labels = idx::readLabels(labelFile);
    auto images = idx::readImages(imageFile);

    if(labels.size() != images.size())
    {
        print("sizes don't match\n");
    }

    auto start = chr::steady_clock::now();
    
    int total{};
    int correct{};

    for(int i=0;i<images.size();i++)
    {
        auto const  label=labels[i];        
        auto const &image=images[i];        

        InputLayer      inputLayer{};
        std::ranges::transform(image.pixels, inputLayer.begin(), normalisePixel);

        HiddenLayer     hiddenLayer = (weights1*inputLayer +biases1).unaryExpr(&sigmoid);
        OutputLayer     outputLayer = (weights2*hiddenLayer+biases2).unaryExpr(&sigmoid);

        OutputLayer     desiredOutput{};
        desiredOutput[label]=1.0;

        int             maxIndex{};
        auto            maxCoeff = outputLayer.maxCoeff(&maxIndex);

        total++;
        if(label==maxIndex)
        {
            correct++;
        }

        if(!checkResult)
        {
            auto            costs = (outputLayer-desiredOutput).array().square();
            auto            cost  = costs.sum();

            print("{} : {} cost={}\n",label,maxIndex,cost);
        }
    }

    auto end = chr::steady_clock::now();

    if(checkResult)
    {
        print("correct = {}/{} = {}%\n",correct,total, correct*100.0/total);
    }

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
        throw_runtime_error("oneLayer random|costs|train|test\n");
    }

    if(args[0]=="random")
    {
        randomise();
    }
    else if(args[0]=="costs")
    {
        analyse("datasets\\train-labels.idx1-ubyte",
                "datasets\\train-images.idx3-ubyte",
              false);
    }
    else if(args[0]=="train")
    {
    }
    else if(args[0]=="test")
    {
        analyse("datasets\\t10k-labels.idx1-ubyte",
                "datasets\\t10k-images.idx3-ubyte",
                true);
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