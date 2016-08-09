#pragma once

#include "percepto/neural/LinearLayer.hpp"
#include "percepto/neural/FullyConnectedNet.hpp"
//#include "percepto/compo/SeriesWrapper.hpp"

#include "percepto/neural/HingeActivation.hpp"
#include "percepto/neural/SigmoidActivation.hpp"
#include "percepto/neural/NullActivation.hpp"

namespace percepto
{

/**
 * @brief A fully connected unit with rectified linear unit activation. 
 */
typedef FullyConnectedNet<LinearLayer, HingeActivation> ReLUNet;

typedef FullyConnectedNet<LinearLayer, SigmoidActivation> PerceptronNet;


}