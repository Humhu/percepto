#!/usr/bin/env python

import rospy, sys, random
import numpy as np
import cPickle as pickle
from percepto_msgs.srv import GetCritique, GetCritiqueRequest, GetCritiqueResponse
from collections import namedtuple
from itertools import izip

Trial = namedtuple( 'Trial', ['seed', 'samples'] )
Sample = namedtuple( 'Sample', ['input', 'outputs'] )

class RandomSearch(object):

    def __init__( self ):

        # Create critique service proxy
        critique_topic = rospy.get_param( '~critic_service' )
        rospy.wait_for_service( critique_topic )
        self.critique_service = rospy.ServiceProxy( critique_topic, GetCritique, True )

        # Read initialization
        self.x_mins = rospy.get_param( '~x_lower_bounds' )
        self.x_maxs = rospy.get_param( '~x_upper_bounds' )
        if len(self.x_mins) != len(self.x_maxs):
            raise ValueError( 'x mins and max must have same number of elements' )

        self.num_arms = rospy.get_param( '~num_arms' )
        self.num_pulls = rospy.get_param( '~num_pulls' )
        self.seeds = rospy.get_param( '~random_seeds' )

        # I/O initialization
        output_path = rospy.get_param( '~output_data_path' )
        self.output_log = open( output_path, 'w' )
        if self.output_log is None:
            raise RuntimeError( 'Could not open output log at path: ' + output_path )
        self.data = []

    def __del__( self ):
        if self.output_log is not None:
            self.output_log.close()

    def GenerateArm( self ):
        return [ random.uniform(low,upp) for (low,upp) in
                 izip( self.x_mins, self.x_maxs ) ]

    def QueryArm( self, x ):
        req = GetCritiqueRequest()
        req.input = x
        try:
            res = self.critique_service.call( req )
        except rospy.ServiceException:
            raise RuntimeError( 'Could not evaluate x: ' + str(x) )
        return res.critique

    def CollectSample( self ):
        arm = self.GenerateArm()
        pulls = [ self.QueryArm( arm ) for i in range( self.num_pulls ) ]
        return { 'arm':arm, 'pulls':pulls }

    def RunTrial( self, seed ):
        rospy.loginfo( 'Executing trial with seed: %d' % seed )
        random.seed(seed)
        samples = [ self.CollectSample() for i in range( self.num_arms )  ]
        return { 'seed':seed, 'samples':samples }

    def Execute( self ):
        results = [ self.RunTrial(s) for s in self.seeds ]
        pickle.dump( results, self.output_log )

if __name__ == '__main__':
    rospy.init_node( 'random_search' )
    cepo = RandomSearch()
    cepo.Execute()
