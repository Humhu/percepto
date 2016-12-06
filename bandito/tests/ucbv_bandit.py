#! /usr/bin/env python
import dill
import cPickle as pickle
import rospy, random
import numpy as np
from itertools import izip

from bandito.bandits import BanditInterface
from bandito.arm_proposals import DiscreteArmProposal
from bandito.arm_selectors import UCBVSelector
from bandito.reward_models import EmpiricalRewardModel

from percepto_msgs.srv import GetCritique, GetCritiqueRequest

class UCBVBandit(object):
    """Multi-arm bandit optimization using the UCB-V selection criteria.

    Interfaces with an optimization problem through the GetCritique service.
    """

    def __init__( self ):
    
        # Seed RNG if specified
        seed = rospy.get_param('~random_seed', None)
        if seed is None:
            rospy.loginfo( 'No random seed specified. Using default behavior.' )
        else:
            rospy.loginfo( 'Initializing with random seed: ' + str(seed) )
            random.seed( seed )

        # Problem parameters
        self.num_rounds = rospy.get_param( '~num_rounds' )
        num_arms = rospy.get_param( '~num_arms' )

        # Arm sampling
        x_lower_lim = rospy.get_param('~arm_lower_limit')
        x_upper_lim = rospy.get_param('~arm_upper_limit')
        if rospy.has_param( '~arm_dim' ):
            dim = rospy.get_param( '~arm_dim' )
            x_lower_lim = (x_lower_lim,) * dim
            x_upper_lim = (x_upper_lim,) * dim

        self.arm_proposal = DiscreteArmProposal()
        for i in range( num_arms ):
            arm = tuple( [ random.uniform(low,upp) for (low,upp) 
                           in izip( x_lower_lim, x_upper_lim ) ] )
            self.arm_proposal.add_arm( arm )

        self.reward_model = EmpiricalRewardModel()

        # Optimization parameters
        b = rospy.get_param('~reward_scale', 1.0)
        c = rospy.get_param('~criteria_c', 1.0)
        self.exp_func = lambda : UCBVSelector.default_exp_func( self.round_num )
        self.arm_selector = UCBVSelector( reward_histories = self.reward_model,
                                          exp_func = self.exp_func,
                                          reward_scale = b,
                                          c = c )
        
        self.bandit = BanditInterface( arm_proposal = self.arm_proposal,
                                       reward_model = self.reward_model,
                                       arm_selector = self.arm_selector )
        

        self.prog_path = rospy.get_param( '~progress_path', None )
        self.out_path = rospy.get_param( '~output_path')
        
        # Initialize state
        self.round_num = 1
        self.rounds = []

    def save( self ):
        if self.prog_path is not None:
            rospy.loginfo( 'Saving progress at %s...', self.prog_path )
            prog = open( self.prog_path, 'wb' )
            pickle.dump( self, prog )
            prog.close()

        rospy.loginfo( 'Saving output at %s...', self.out_path )
        out = open( self.out_path, 'wb' )
        data = (self.arm_proposal.get_arms(), self.rounds)
        pickle.dump( data, out )
        out.close()

    def is_done( self ):
        return self.round_num > self.num_rounds

    def execute( self, eval_cb ):
        """Begin/resume execution.
        """
        while not rospy.is_shutdown() and not self.is_done():

            arm = self.bandit.ask()
            arm_ind = self.arm_proposal.get_arm_by_ind( arm )

            rospy.loginfo( 'Round %d Evaluating arm %s' % (self.round_num, arm_ind ) )
            reward, feedback = eval_cb( arm )
            self.bandit.tell( arm, reward )
            
            self.rounds.append( (self.round_num, arm_ind, reward, feedback) )
            self.save()

            self.round_num += 1

def evaluate_input( proxy, inval, num_retries=1 ):
    """Query the optimization function.

    Parameters
    ----------
    proxy : rospy.ServiceProxy
        Service proxy to call the GetCritique service for evaluation.

    inval : numeric array
        Input values to evaluate.

    Return
    ------
    reward : numeric
        The reward of the input values
    feedback : list
        List of feedback
    """
    req = GetCritiqueRequest()
    req.input = inval

    for i in range(num_retries+1):
        try:
            res = proxy.call( req )
            break
        except rospy.ServiceException:
            rospy.logerr( 'Could not evaluate item: ' + np.array_str( inval ) )
    
    reward = res.critique
    rospy.loginfo( 'Evaluated input: %s\noutput: %f\n feedback: %s', 
                   str( inval ),
                   reward,
                   str( res.feedback ) )
    return (reward, res.feedback)

if __name__ == '__main__':
    rospy.init_node( 'ucb_v_bandit_optimizer' )

    # See if we're resuming
    if rospy.has_param( '~load_path' ):
        data_path = rospy.get_param( '~load_path' )
        data_log = open( data_path, 'rb' )
        rospy.loginfo( 'Found load data at %s...', data_path )
        cmaopt = pickle.load( data_log )
    else:
        rospy.loginfo( 'No resume data specified. Starting new optimization...' )
        cmaopt = UCBVBandit()

    # Create interface to optimization problem
    critique_topic = rospy.get_param( '~critic_service' )
    rospy.loginfo( 'Waiting for service %s...', critique_topic )
    rospy.wait_for_service( critique_topic )
    rospy.loginfo( 'Connected to service %s.', critique_topic )
    critique_proxy = rospy.ServiceProxy( critique_topic, GetCritique, True )
    eval_cb = lambda x : evaluate_input( critique_proxy, x )

    # Register callback so that the optimizer can save progress
    cmaopt.execute( eval_cb )
