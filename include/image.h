#pragma once
#include <vector>
#include <cstdint>

struct Image
{
    int height;
    int width;
    std::vector<uint8_t>    pixels;

};
