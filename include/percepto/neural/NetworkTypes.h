#pragma once

#include "percepto/neural/LinearLayer.hpp"
#include "percepto/neural/FullyConnectedNet.hpp"
#include "percepto/compo/SeriesWrapper.hpp"

#include "percepto/neural/HingeActivation.hpp"
#include "percepto/neural/SigmoidActivation.hpp"
#include "percepto/neural/NullActivation.hpp"

namespace percepto
{

/**
 * @brief A fully connected unit with rectified linear unit activation. 
 */
typedef FullyConnectedNet<HingeActivation> ReLUNet;

/**
 * @brief A fully connected unit with sigmoid activation.
 */
typedef FullyConnectedNet<SigmoidActivation> PerceptronNet;

/**
 * @brief A fully connected unit without any activation units. Should be
 * composed with another unit as a final output layer.
 */
typedef LinearLayer<NullActivation> UnrectifiedLinearLayer;

/**
 * @brief A fully connected unit with sigmoid activations followed by a final
 * linear output layer.
 */
typedef SeriesWrapper<PerceptronNet, UnrectifiedLinearLayer> PerceptronRegressionNet;
}