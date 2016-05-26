import numpy as np
import covreg.regressors as creg
import covreg.optimization as copt
from numpy.random import multivariate_normal as mvn

lInputDim = 4
dInputDim = 6
outputDim = 3
mcOffset = 1E-6 * np.identity( outputDim )

numDatapoints = 1000
lWeight = 0
dWeight = 1E-3

# Initial model
initialModel = creg.create_default_modchol( lInputDim = lInputDim,
                                            dInputDim = dInputDim,
                                            outputDim = outputDim,
                                            offset = mcOffset )
initialParams = initialModel.GetParameters()

# Create model to sample random data
randLParams = np.random.rand( *initialParams.lParameters.shape )
randDParams = np.random.rand( *initialParams.dParameters.shape )
randomParams = creg.ModCholRegressorParameters( lParameters = randLParams,
                                                dParameters = randDParams )
randomModel = creg.ModCholRegressor( parameters = randomParams,
                                     offset = mcOffset )

# Generate features
data = []
for i in xrange(numDatapoints):
    lFeature = np.random.rand( lInputDim, 1 )
    dFeature = np.random.rand( dInputDim, 1 )
    features = creg.ModCholInput( lInput = lFeature,
                                  dInput = dFeature )
    randomCov = randomModel.Evaluate( features )
    sample = np.random.multivariate_normal( np.zeros(outputDim), randomCov )
    datum = copt.DirectLikelihoodData( sample = sample, input = features )
    data.append( datum )

optParams = { "abs_function_tol" : 1E-6,
              "abs_parameter_tol" : 1E-6,
              "max_runtime_secs" : 600 }

retval = copt.fit_direct_likelihood( model = initialModel, 
                                     dataset = data, 
                                     lWeight = lWeight,
                                     dWeight = dWeight,
                                     optimizationParams = optParams )

print "Optimization took %f seconds." % retval.totalElapsedSecs
print "Final params:"
print retval.finalParameters
print "True params:"
print randomParams