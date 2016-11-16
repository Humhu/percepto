#!/usr/bin/env python

import rospy, sys, cma
import numpy as np
import cPickle as pickle
from percepto_msgs.srv import GetCritique, GetCritiqueRequest, GetCritiqueResponse
from collections import namedtuple

def PrintArray( a ):
    return np.array_str( a, max_line_width=sys.maxint )

def PrintCES( ces ):
    return 'Input: %s Output: %f' % ( PrintArray( ces.input ), ces.output )

class CMAOptimization:
    """Covariance Matrix Adaptation (CMA) evolution strategy optimizer. 

    Uses the implementation in the python CMA package.
    
    Interfaces with an optimization problem through the GetCritique service.

    ROS Parameters
    --------------
    ~critic_service : string
        The GetCritique service topic for the optimization task.
    
    ~initial_mean : N-length numeric array
        The CMA search initial mean. The dimensionality N of the optimization input 
        is inferred from this array.

    ~initial_std_dev : N-length numeric array or scalar (default 1.0)
        The CMA search initial standard deviations. If an array it must be N elements,
        one for each input dimension.

    ~num_restarts : integer (default 0)
        Number of top-level reinitializations to perform.

    ~output_log_path : string file path
        The file to pickle the execution trace and results to.

    ~random_seed : integer (optional, default system time)
        The random number generator seed. Defaults to system time if not specified.

    ~population_size : integer (optional, default 4+int(3*log(N)))
        The number of samples to collect at each iteration.

    ~diagonal_only : boolean, (optional, default true after 0*100*N/sqrt(popsize) iters)

    ~bounds : length 2 numeric array (optional, default [None,None])
        Lower and upper bounds for input dimensions.

    ~convergence/max_iters : numeric (default Inf)

    ~convergence/max_time : numeric (default Inf)

    ~convergence/output_change : numeric (default -Inf)
    
    ~convergence/input_change : numeric (default -Inf)
   """

    def __init__( self ):

        # Create critique service proxy
        critique_topic = rospy.get_param( '~critic_service' )
        rospy.wait_for_service( critique_topic )
        self.critique_service = rospy.ServiceProxy( critique_topic, GetCritique, True )

        # Read initialization
        self.init_mean = rospy.get_param( '~initial_mean' )
        self.init_stds = rospy.get_param( '~initial_std_dev', 1.0 )

        # Custom termination conditions
        self.max_iters = rospy.get_param( '~convergence/max_iters', float('Inf') )
        self.max_runtime = float(rospy.get_param( '~convergence/max_time', float('Inf') ) )

        # Read optimization parameters
        self.num_restarts = rospy.get_param( '~num_restarts', 0 )

        # I/O initialization
        output_path = rospy.get_param( '~output_log_path' )
        self.output_log = open( output_path, 'wb' )
        if self.output_log is None:
            raise RuntimeError( 'Could not open output log at path: ' + output_path )
        self.loggers = []

        self.cma_options = cma.CMAOptions()
        
        self.ReadCMAParam( ros_key='~random_seed', cma_key='seed' )
        self.ReadCMAParam( ros_key='~population_size', cma_key='popsize' )
        self.ReadCMAParam( ros_key='~diagonal_only', cma_key='CMA_diagonal' )
        self.cma_options['tolfun'] = float( rospy.get_param( '~convergence/output_change', -float('Inf') ) )
        self.cma_options['tolx'] = float( rospy.get_param( '~convergence/input_change', -float('Inf') ) )

        if rospy.has_param( '~bounds' ):
            raw = rospy.get_param( '~bounds' )
            bounds = [ float(r) for r in raw ]
            self.cma_options['bounds'] = bounds

        # TODO Make an option
        self.cma_options['verb_disp'] = 1
        self.cma_options['verb_plot'] = 1
        
    def __del__( self ):
        if self.output_log is not None:
            self.output_log.close()

    def ReadCMAParam( self, ros_key, cma_key ):
        if rospy.has_param( ros_key ):
            val = rospy.get_param( ros_key )
            self.cma_options[cma_key] = val
            self.Log( cma_key + ': ' + str( val ) )

    def OptimizationIteration( self ):
        """Execute CMA with a random initialization once.

        Returns an array of [iter_num, inputs, outputs].
        """
        # Initialize state
        self.start_time = rospy.Time.now()
        self.iter_counter = 0

        rounds = []
        optimizer = cma.CMAEvolutionStrategy( self.init_mean, self.init_stds, self.cma_options )
        while not optimizer.stop() and not rospy.is_shutdown():
            
            if self.HasTerminated():
                break
            inputs = optimizer.ask()
            outputs = [ self.EvaluateInput( inval ) for inval in inputs ]

            # CMA feedback and visualization
            optimizer.tell( inputs, outputs )
            optimizer.logger.add()
            optimizer.disp()
            cma.show()
            optimizer.logger.plot()

            # Record iteration
            rounds.append( [self.iter_counter, inputs, outputs] )
            self.iter_counter += 1

        self.loggers.append( optimizer.logger )
        return rounds

    def EvaluateInput( self, inval ):
        """Query the optimization function.
        """
        # Be able to bail when needed
        if rospy.is_shutdown():
            sys.exit(-1)

        req = GetCritiqueRequest()
        req.input = inval
        try:
            res = self.critique_service.call( req )
        except rospy.ServiceException:
            raise RuntimeError( 'Could not evaluate item: ' + PrintArray( inval ) )
        sample = CrossEntropySample( input=inval, output=res.critique )
        self.Log( 'Evaluated: ' + PrintCES( sample ) )

        # Critique is a reward so we have to negate it to get a cost
        return -res.critique

    def HasTerminated( self ):
        """Check the custom termination conditions.
        """
        # Check runtime
        runtime_exceeded = (rospy.Time.now() - self.start_time).to_sec() > self.max_runtime
        if runtime_exceeded:
            print 'Runtime exceeded.'
            return True

        # Check number of iterations
        iters_exceeded = self.iter_counter >= self.max_iters
        if iters_exceeded:
            print 'Max iterations exceeded'
            return True

        return False

    def Execute( self ):

        best_solution = cma.BestSolution()

        results = []
        for trial in range( 0, self.num_restarts + 1 ):
            rospy.loginfo( 'Executing trial %d...', trial )
            results.append( self.OptimizationIteration() )

        pickle.dump( self.output_log )

if __name__ == '__main__':
    rospy.init_node( 'cma_optimizer' )
    cepo = CMAOptimization()
    cepo.Execute()
