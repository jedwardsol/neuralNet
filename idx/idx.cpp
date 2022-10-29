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


    return labels;
}



}