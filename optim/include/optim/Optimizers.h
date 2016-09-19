#pragma once

#include "optim/ModularOptimizer.h"

#include "optim/AdamSearchDirector.h"
#include "optim/GradientSearchDirector.h"
#include "optim/NaturalSearchDirector.h"

#include "optim/FixedSearchStepper.h"
#include "optim/L1ConstrainedSearchStepper.h"
#include "optim/L2ConstrainedSearchStepper.h"
#include "optim/BacktrackingSearchStepper.h"
#include "optim/ConstrainedBacktrackingSearchStepper.h"

#include "optim/GradientTerminationChecker.h"
#include "optim/IterationTerminationChecker.h"
#include "optim/RuntimeTerminationChecker.h"
#include "optim/ConstraintTerminationChecker.h"