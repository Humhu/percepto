#pragma once

#include "modprop/neural/LinearLayer.hpp"
#include "modprop/neural/FullyConnectedNet.hpp"
//#include "percepto/compo/SeriesWrapper.hpp"

#include "modprop/neural/HingeActivation.hpp"
#include "modprop/neural/SigmoidActivation.hpp"
#include "modprop/neural/NullActivation.hpp"

namespace percepto
{

/**
 * @brief A fully connected unit with rectified linear unit activation. 
 */
typedef FullyConnectedNet<LinearLayer, HingeActivation> ReLUNet;

typedef FullyConnectedNet<LinearLayer, SigmoidActivation> PerceptronNet;


}