#pragma once
#include <string>
#include <vector>
#include "include/image.h"

// http://yann.lecun.com/exdb/mnist/

namespace idx
{

std::vector<int>    readLabels(std::string const &filename);
std::vector<Image>  readImages(std::string const &filename);



}