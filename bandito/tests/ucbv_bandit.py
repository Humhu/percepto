#! /usr/bin/env python
import rospy, random
import numpy as np
from itertools import izip

from bandito.bandits import BanditInterface
from bandito.arm_proposals import DiscreteArmProposal
from bandito.arm_selectors import UCBVSelector
from bandito.reward_models import EmpiricalRewardModel

from percepto_msgs.srv import GetCritique, GetCritiqueRequest

class UCBVBandit(object):
    """
    A simple test bandit node.
    """

    def __init__( self ):
        
        # Seed RNG if specified
        seed = rospy.get_param('~random_seed', None)
        if seed is None:
            rospy.loginfo('No random seed specified. Using system time.')
        else:
            rospy.loginfo('Initializing with random seed: ' + str(seed) )
        random.seed( seed )

        self.num_rounds = rospy.get_param('~num_rounds', float('Inf'))
        
        # Output log
        log_path = rospy.get_param('~output_log')
        self.out_log = open( log_path, 'w' )
        if self.out_log is None:
            raise IOError('Could not open output log at: ' + log_path)

        num_arms = rospy.get_param('~num_arms', 0)
        b = rospy.get_param('~reward_scale', 1.0)
        c = rospy.get_param('~criteria_c', 1.0)
        beta = rospy.get_param('~beta')

        # Print header
        self.out_log.write('Random seed: %s\n' % str(seed))
        self.out_log.write('Reward scale: %f\n' % b)
        self.out_log.write('Criteria c: %f\n' % c)
        self.out_log.write('Hardness beta: %f\n' % beta)
        self.out_log.write('Init arms: %d\n' % num_arms)
        self.out_log.write('Num rounds: %d\n' % self.num_rounds)

        self.arm_lower_lims = np.array(rospy.get_param('~arm_lower_limits'))
        self.arm_upper_lims = np.array(rospy.get_param('~arm_upper_limits'))
        if len( self.arm_lower_lims ) != len( self.arm_upper_lims ):
            raise ValueError( 'Lower and upper limits must have save length.' )
        self.arm_proposal = DiscreteArmProposal()
        for i in range( num_arms ):
            self.add_arm()

        self.reward_model = EmpiricalRewardModel()

        self.round_num = 1
        self.exp_func = lambda : UCBVSelector.default_exp_func( self.round_num )
        self.arm_selector = UCBVSelector( reward_histories = self.reward_model,
                                          exp_func = self.exp_func,
                                          reward_scale = b,
                                          c = c )

        self.bandit = BanditInterface( arm_proposal = self.arm_proposal,
                                       reward_model = self.reward_model,
                                       arm_selector = self.arm_selector )

        # Create critique service proxy
        critique_topic = rospy.get_param( '~critic_service' )
        rospy.wait_for_service( critique_topic )
        self.critique_service = rospy.ServiceProxy( critique_topic, GetCritique, True )

    def add_arm( self ):
        arm = tuple( [ random.uniform(low,upp) for (low,upp) 
                     in izip( self.arm_lower_lims, self.arm_upper_lims ) ] )
        self.arm_proposal.add_arm( arm )
        msg = 'Arm: %s\n' % str(arm)
        rospy.loginfo( msg )
        self.out_log.write( msg )
        self.out_log.flush()

    def evaluate_input( self, inval ):
        req = GetCritiqueRequest()
        req.input = inval
        try:
            res = self.critique_service.call( req )
        except rospy.ServiceException:
            raise RuntimeError( 'Could not evaluate item: ' + str( inval ) )
        return res.critique

    def execute( self ):
        while not rospy.is_shutdown() and self.round_num < self.num_rounds:
            arm = self.bandit.ask()
            # TODO Arm adding logic

            rospy.loginfo( 'Round %d Evaluating arm %s' % (self.round_num,str(arm)) )
            reward = self.evaluate_input( arm )
            rospy.loginfo( 'Arm returned reward %f' % reward )
            self.bandit.tell( arm, reward )
            
            self.out_log.write( 'Round: %d Arm: %s Reward: %f\n' % 
                                (self.round_num, str(arm), reward) )
            self.out_log.flush()
            self.round_num += 1
        self.out_log.close()

if __name__=='__main__':
    rospy.init_node( 'bandit_node' )
    pbn = UCBVBandit()
    pbn.execute()
