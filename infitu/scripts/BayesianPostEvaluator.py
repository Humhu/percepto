#!/usr/bin/env python

import pickle, rospy, sys
import numpy as np
from itertools import izip

from percepto_msgs.srv import GetCritique, GetCritiqueRequest

class BayesianPostEvaluator:
    """Postprocessing node for Bayesian optimization results.
    """

    def __init__( self ):
        self.trials_per_input = rospy.get_param( '~trials_per_input' )

        results_path = rospy.get_param( '~results_path' )
        results_file = open( results_path, 'rb' )
        results_data = pickle.load( results_file )
        self.test_inputs = []
        for dat in results_data[2]:
            # Each point is (round_num, input, est_value, bounds)
            self.test_inputs.append( dat[1] )

        self.save_period = rospy.get_param( '~save_period', 1 )
        self.rounds = []
        self.prog_path = rospy.get_param( '~progress_path', None )
        self.out_path = rospy.get_param( '~output_path' )

    def query( self, eval_cb, inval ):
        req = GetCritiqueRequest()
        req.input = inval
        critiques = []
        feedbacks = {}

        while len(critiques) < self.trials_per_input:
            rospy.loginfo( 'Trial input %d/%d', len(critiques), self.trials_per_input )
            (critique, feedback) = eval_cb( inval )
            
            critiques.append( critique )
            for (n,v) in feedback.iteritems():
                if n not in feedbacks:
                    feedbacks[n] = []
                feedbacks[n].append(v)

        return [critiques, feedback]

    def execute( self, eval_cb ):

        while len( self.rounds ) < len( self.test_inputs ):
            i = len( self.rounds )
            inval = self.test_inputs[i]
            rospy.loginfo( 'Testing input %d/%d value: %s', 
                           i, len( self.test_inputs ), np.array_str( inval ) )

            self.rounds.append( [inval] + self.query( eval_cb, inval ) )
            self.save()
        self.save() #Redundant...

    def save( self ):
        if len( self.rounds ) % self.save_period != 0:
            return

        if self.prog_path is not None:
            rospy.loginfo( 'Saving progress at %s...', self.prog_path )
            prog = open( self.prog_path, 'wb' )
            pickle.dump( self, prog )
            prog.close()

        rospy.loginfo( 'Saving output at %s...', self.out_path )
        out = open( self.out_path, 'wb' )
        pickle.dump( self.rounds, out )
        out.close()

def evaluate_input( proxy, inval):

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

if __name__ == '__main__':

    rospy.init_node( 'bayesian_optimizer' )

    # See if we're resuming
    if rospy.has_param( '~load_path' ):
        data_path = rospy.get_param( '~load_path' )
        data_log = open( data_path, 'rb' )
        rospy.loginfo( 'Found load data at %s...', data_path )
        bopt = pickle.load( data_log )
    else:
        rospy.loginfo( 'No resume data specified. Starting new optimization...' )
        bopt = BayesianPostEvaluator()

    # Create interface to optimization problem
    critique_topic = rospy.get_param( '~critic_service' )
    rospy.loginfo( 'Waiting for service %s...', critique_topic )
    rospy.wait_for_service( critique_topic )
    rospy.loginfo( 'Connected to service %s.', critique_topic )
    critique_proxy = rospy.ServiceProxy( critique_topic, GetCritique, True )
    eval_cb = lambda x : evaluate_input( critique_proxy, x )

    bopt.execute( eval_cb )
