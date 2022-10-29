#pragma once
#include <string>
#include <vector>

// http://yann.lecun.com/exdb/mnist/

namespace idx
{

std::vector<int>  readLabels(std::string const &filename);



}