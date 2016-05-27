#pragma once

#include "percepto/optim/ModularOptimizer.hpp"

// Convergence 
#include "percepto/optim/SimpleConvergence.hpp"

// Steppers
#include "percepto/optim/AdamStepper.hpp"

namespace percepto
{

typedef ModularOptimizer<AdamStepper, SimpleConvergence> AdamOptimizer;

}