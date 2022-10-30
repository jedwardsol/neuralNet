#include <windows.h>
#include "idx/idx.h"
#include "include/print.h"
#pragma comment(lib,"idx")
#include <array>
int main()
try
{
    auto conOut = GetStdHandle( STD_OUTPUT_HANDLE );
    SetConsoleOutputCP( CP_UTF8 );

    auto labels = idx::readLabels("data\\train-labels.idx1-ubyte");
    auto images = idx::readImages("data\\train-images.idx3-ubyte");

    if(labels.size() != images.size())
    {
        print("sizes don't match\n");
    }

    std::array<char8_t const*,5> blocks
    {
        u8" ",
        u8"░",
        u8"▒",
        u8"▓",
        u8"█",
    };


    for(int i=0;i<images.size();i++)
    {
        print("{} ",labels[i]);

        auto const &image=images[i];        

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

        print("\n");
    }


}
catch(std::exception const &e)
{
    print("caught {}\n",e.what());
}