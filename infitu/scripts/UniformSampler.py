#!/usr/bin/env python

import rospy, sys, pickle
import numpy as np
from itertools import izip
from percepto_msgs.srv import GetCritique, GetCritiqueRequest

def wait_for_service( srv ):
    rospy.loginfo( 'Waiting for service %s', srv )
    rospy.wait_for_service( srv )
    rospy.loginfo( 'Service now available %s', srv )

class UniformSampler:

    def __init__( self ):

         # Seed RNG if specified
        seed = rospy.get_param('~random_seed', None)
        if seed is None:
            rospy.loginfo( 'No random seed specified. Using default behavior.' )
        else:
            rospy.loginfo( 'Initializing with random seed: ' + str(seed) )
            np.random.seed( seed )

        self.save_period = rospy.get_param( '~save_period', 1 )

        # Reward model and bandit
        self.num_samples_per = rospy.get_param( '~num_samples' );
        self.input_dim = rospy.get_param( '~input_dimension' )
        input_lower = rospy.get_param( '~input_lower_bound' )
        input_upper = rospy.get_param( '~input_upper_bound' )

        burn_num = rospy.get_param( '~init_burn_count', None )
        if burn_num is not None:
            rospy.loginfo( 'Burning %d initial samples', burn_num )
            np.random.uniform( low=input_lower, high=input_upper,
                               size=(burn_num, self.input_dim) )

        num_inputs_to_test = rospy.get_param( '~num_inputs_to_test' )
        variation = rospy.get_param( '~input_variation' )
        offset = np.array( rospy.get_param( '~input_offset' ) )
        self.test_inputs = np.random.uniform( low=-variation, high=variation,
                                              size=(num_inputs_to_test, self.input_dim ) ) + offset
        self.test_inputs[ self.test_inputs < input_lower ] = input_lower
        self.test_inputs[ self.test_inputs > input_upper ] = input_upper

        self.rounds = []
        self.prog_path = rospy.get_param( '~progress_path', None )
        self.out_path = rospy.get_param( '~output_path' )

    def evaluate( self, eval_cb, inval ):
        results = []
        for i in range( self.num_samples_per ):
            rospy.loginfo( 'Sample %d/%d...', i+1, self.num_samples_per )
            results.append( eval_cb( inval ) )
        return results

    def execute( self, eval_cb ):
        while len( self.rounds ) < len( self.test_inputs ):
            rospy.loginfo( 'Input %d/%d', len( self.rounds )+1, len( self.test_inputs ) )
            x = self.test_inputs[ len( self.rounds ) ]
            trial = self.evaluate( eval_cb, x )
            self.rounds.append( (x, trial) )
            self.save( 'in progress' )
        self.save( 'completed' )

    def save( self, status ):
        if len( self.rounds ) % self.save_period != 0:
            return

        if self.prog_path is not None:
            rospy.loginfo( 'Saving progress at %s...', self.prog_path )
            prog = open( self.prog_path, 'wb' )
            pickle.dump( self, prog )
            prog.close()

        rospy.loginfo( 'Saving output at %s...', self.out_path )
        out = open( self.out_path, 'wb' )
        pickle.dump( (status, self.rounds), out )
        out.close()

def evaluate_input( proxy, inval ):

    req = GetCritiqueRequest()
    req.input = inval

    try:
        res = proxy.call( req )
    except rospy.ServiceException:
        rospy.logerr( 'Could not evaluate item: ' + np.array_str( inval ) )
        return None
    
    msg = 'Evaluated input: %s\n' % np.array_str( inval, max_line_width=sys.maxint )
    msg += 'Critique: %f\n' % res.critique
    msg += 'Feedback:\n'
    feedback = {}
    for (name,value) in izip( res.feedback_names, res.feedback_values ):
        msg += '\t%s: %f\n' % ( name, value )
        feedback[name] = value
    rospy.loginfo( msg )

    return (res.critique, feedback)

if __name__=='__main__':
    rospy.init_node( 'uniform_sampler' )
    
        # See if we're resuming
    if rospy.has_param( '~load_path' ):
        data_path = rospy.get_param( '~load_path' )
        data_log = open( data_path, 'rb' )
        rospy.loginfo( 'Found load data at %s...', data_path )
        usamp = pickle.load( data_log )
    else:
        rospy.loginfo( 'No resume data specified. Starting new sampler...' )
        usamp = UniformSampler()

    # Create interface to optimization problem
    critique_topic = rospy.get_param( '~critic_service' )
    rospy.loginfo( 'Waiting for service %s...', critique_topic )
    rospy.wait_for_service( critique_topic )
    rospy.loginfo( 'Connected to service %s.', critique_topic )
    critique_proxy = rospy.ServiceProxy( critique_topic, GetCritique, True )
    eval_cb = lambda x : evaluate_input( critique_proxy, x )

    usamp.execute( eval_cb )
