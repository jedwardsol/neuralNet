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

constexpr double    trainingRate{0.01};

constexpr int       inputLayerSize{28*28};
constexpr int       outputLayerSize{10};
constexpr int       hiddenLayerSize{(inputLayerSize + outputLayerSize)/2};

using InputLayer   = Eigen::Matrix<double,inputLayerSize,1>;

using Weights1     = Eigen::Matrix<double,hiddenLayerSize,inputLayerSize>;
using Biases1      = Eigen::Matrix<double,hiddenLayerSize,1>;

using HiddenLayer  = Eigen::Matrix<double,hiddenLayerSize,1>;

using Weights2     = Eigen::Matrix<double,outputLayerSize,hiddenLayerSize>;
using Biases2      = Eigen::Matrix<double,outputLayerSize,1>;

using OutputLayer  = Eigen::Matrix<double,outputLayerSize,1>;

std::array<char8_t const*,5> blocks
{
    u8" ",
    u8"░",
    u8"▒",
    u8"▓",
    u8"█",
};


double sigmoid(double x)
{
    return 1.0/(1.0+std::exp(-x));
}


double sigmoid_derivative(double x)
{
	return x * (1.0 - x);
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


void analyse(std::string const &labelFile,std::string const &imageFile, bool showMistakes)
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
    double totalCost{};

    for(int i=0;i<images.size();i++)
    {
        auto const  label=labels[i];        
        auto const &image=images[i];        

        InputLayer      inputLayer{};
        std::ranges::transform(image.pixels, inputLayer.begin(), normalisePixel);

        HiddenLayer     hiddenLayer = (weights1*inputLayer +biases1).unaryExpr(&sigmoid);
        OutputLayer     outputLayer = (weights2*hiddenLayer+biases2).unaryExpr(&sigmoid);

        OutputLayer     desiredOutput;
        desiredOutput.fill(0);
        desiredOutput[label]=1.0;

        int             maxIndex{};
        auto            maxCoeff = outputLayer.maxCoeff(&maxIndex);

        total++;
        if(label==maxIndex)
        {
            correct++;
        }
        else if(showMistakes)
        {
            print("Should be {}.  Net said {}\n",label,maxIndex);

            for(int row=0;row<image.height;row++)
            {
                for(int col=0;col<image.width;col++)
                {
                    auto pixel = image.pixels[row*image.width+col];
                    auto block = blocks[ pixel * 5 / 256];
                    print("{}",reinterpret_cast<char const*>(block));
                }
                print("\n");
            }


        }



        double  cost = (outputLayer-desiredOutput).array().square().sum();
        totalCost += cost;
    }

    auto end = chr::steady_clock::now();

    print("Correct  = {}/{} = {}%\n",correct,total, correct*100.0/total);
    print("Cost     = {}\n",totalCost);
    print("Duration = {}\n",chr::duration_cast<chr::seconds>(end-start));
}



void train()
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
    

    for(int index=0;index<images.size();index++)
    {
        print("{}\r",index);

        auto const  label=labels[index];        
        auto const &image=images[index];        

        InputLayer      inputLayer{};
        std::ranges::transform(image.pixels, inputLayer.begin(), normalisePixel);

        HiddenLayer     hiddenLayer = (weights1*inputLayer +biases1).unaryExpr(&sigmoid);
        OutputLayer     outputLayer = (weights2*hiddenLayer+biases2).unaryExpr(&sigmoid);

        OutputLayer     desiredOutput;
        desiredOutput.fill(0);
        desiredOutput[label]=1.0;

        // Calculate errors

        // error at outputNeuron[x] = outputDelta[x]  * dsigma(outputNeuron[x])

        OutputLayer     outputDelta  = (outputLayer-desiredOutput);
        OutputLayer     outputSlope  = outputDelta.array() * outputLayer.unaryExpr(&sigmoid_derivative).array();

        HiddenLayer     hiddenSlope{};

        // error at hiddenNeuron[x] = Σ (outputDelta[j] * weight x->j ) * dsigma(hiddenNeuron[x])
        for(int i=0;i<hiddenSlope.size();i++)
        {
            // TODO : fancy way to do this?
            hiddenSlope[i] = (outputDelta.array() * weights2.col(i).array()).sum() * sigmoid_derivative(hiddenLayer[i]);
        }

        // update weights

        // input -> hidden
        for(int hiddenNeuron=0;hiddenNeuron<weights1.rows();hiddenNeuron++)
        {
            for(int inputNeuron=0;inputNeuron<weights1.cols();inputNeuron++)
            {
                weights1(hiddenNeuron,inputNeuron) -= trainingRate * hiddenSlope[hiddenNeuron] * inputLayer[inputNeuron];
            }
        }

        // hidden -> output

        for(int outputNeuron=0;outputNeuron<weights2.rows();outputNeuron++)
        {
            for(int hiddenNeuron=0;hiddenNeuron<weights2.cols();hiddenNeuron++)
            {
                weights2(outputNeuron,hiddenNeuron) -= trainingRate * outputSlope[outputNeuron] * hiddenLayer[hiddenNeuron];
            }
        }


    }

    auto end = chr::steady_clock::now();


    write(weights1,"matrices\\1l_weights1");
    write(biases1, "matrices\\1l_biases1");
    write(weights2,"matrices\\1l_weights2");
    write(biases2, "matrices\\1l_biases2");


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
        throw_runtime_error("oneLayer random|costs|test|mistakes\n");
    }

    if(args[0]=="random")
    {
        randomise();
    }
    else if(args[0]=="costs")
    {
        analyse("datasets\\train-labels.idx1-ubyte",
                "datasets\\train-images.idx3-ubyte",false);
    }
    else if(args[0]=="train")
    {
        train();
    }
    else if(args[0]=="test")
    {
        analyse("datasets\\t10k-labels.idx1-ubyte",
                "datasets\\t10k-images.idx3-ubyte",false);
    }
    else if(args[0]=="mistakes")
    {
        analyse("datasets\\t10k-labels.idx1-ubyte",
                "datasets\\t10k-images.idx3-ubyte",true);
    }
    else
    {
        throw_runtime_error("oneLayer random|costs|train|test|mistakes\n");
    }
}
catch(std::exception const &e)
{
    print("caught {}\n",e.what());
}