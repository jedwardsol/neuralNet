#pragma once
#include "include/thrower.h"
#include <fstream>
#include <string>

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
