#include "idx.h"

#include <string>
#include <vector>
#include <fstream>
#include <cstdint>
#include <bit>
#include "include/thrower.h"

namespace idx
{

std::vector<int>  readLabels(std::string const &filename)
{
    std::ifstream   file{filename,std::ios::binary};

    if(!file)
    {
        throw_system_error("file "+filename);
    }

    uint32_t        magic{};
    file.read(reinterpret_cast<char*>(&magic),sizeof(magic));

    uint32_t        numItems{};
    file.read(reinterpret_cast<char*>(&numItems),sizeof(magic));

    if(!file)
    {
        throw_runtime_error("failed to read magic & number of items "+filename);
    }

    if(magic != 0x01080000)
    {
        throw_runtime_error("magic number incorrect "+filename);
    }

    numItems = std::byteswap(numItems);

    std::vector<int>    labels(numItems);

    for(auto &label : labels)
    {
        label=file.get();
    }

    if(!file)
    {
        throw_runtime_error("failed to read data "+filename);
    }

    return labels;
}



std::vector<Image>  readImages(std::string const &filename)
{
    std::ifstream   file{filename,std::ios::binary};

    if(!file)
    {
        throw_system_error("file "+filename);
    }

    uint32_t        magic{};
    file.read(reinterpret_cast<char*>(&magic),sizeof(magic));

    uint32_t        numItems{};
    file.read(reinterpret_cast<char*>(&numItems),sizeof(numItems));

    uint32_t        height{};
    file.read(reinterpret_cast<char*>(&height),sizeof(height));

    uint32_t        width{};
    file.read(reinterpret_cast<char*>(&width),sizeof(width));


    if(!file)
    {
        throw_runtime_error("failed to read header "+filename);
    }

    if(magic != 0x03080000)
    {
        throw_runtime_error("magic number incorrect "+filename);
    }

    numItems = std::byteswap(numItems);
    height   = std::byteswap(height);
    width    = std::byteswap(width);

    std::vector<Image>    images;
    images.reserve(numItems);

    for(auto i=0u;i<numItems;i++)
    {
        Image   image(height,width);
        image.pixels.resize(height*width);
        file.read(reinterpret_cast<char*>(image.pixels.data()), height*width);

        images.push_back(std::move(image));
    }

    if(!file)
    {
        throw_runtime_error("failed to read data "+filename);
    }

    return images;
}



}