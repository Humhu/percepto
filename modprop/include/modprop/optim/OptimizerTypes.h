#pragma once

#include "modprop/optim/ModularOptimizer.hpp"
#include "modprop/optim/NaturalOptimizer.hpp"

// Convergence 
#include "modprop/optim/SimpleConvergence.hpp"

// Steppers
#include "modprop/optim/AdamStepper.hpp"
#include "modprop/optim/DirectStepper.hpp"

namespace percepto
{

typedef ModularOptimizer<AdamStepper, SimpleConvergence> AdamOptimizer;
typedef ModularOptimizer<DirectStepper, SimpleConvergence> DirectOptimizer;
typedef NaturalOptimizer<SimpleConvergence> SimpleNaturalOptimizer;

}