#include "idx/idx.h"
#include "include/print.h"
#pragma comment(lib,"idx")

int main()
{
    auto labels = idx::readLabels("data\\train-labels.idx1-ubyte");

    print("{}\n",labels.size());
    for(auto label : labels)
    {
        print("{} ",label);

    }


}