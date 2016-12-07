#! /usr/bin/env python
import rospy, random
import numpy as np
from itertools import izip, product

from bandito.bandits import BanditInterface
from bandito.arm_proposals import NullArmProposal
from bandito.arm_selectors import *
from bandito.reward_models import GaussianProcessRewardModel

from percepto_msgs.srv import GetCritique, GetCritiqueRequest

from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

class GPUCBBandit(object):
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

        beta = rospy.get_param('~beta')
        init_num_samples = rospy.get_param('~init_hyperparam_samples', 30)

        # Print header
        self.out_log.write('Random seed: %s\n' % str(seed))
        self.out_log.write('Exploration beta: %f\n' % beta)
        self.out_log.write('Num rounds: %d\n' % self.num_rounds)

        x_lower = rospy.get_param('~arm_lower_limit')
        x_upper = rospy.get_param('~arm_upper_limit')
        arm_dim = rospy.get_param( '~arm_dim' )

        self.white = WhiteKernel( 1.0, (1e-3, 1e3) )
        self.kernel_base = ConstantKernel( 1.0, (1e-3, 1e1) ) * RBF( 1.0, (1e-3, 1e-1) )
        self.kernel_noisy = self.kernel_base  + self.white
        self.reward_model = GaussianProcessRewardModel( kernel = self.kernel_noisy,
                                                        kernel_noiseless = self.kernel_base,
                                                        hyperparam_min_samples = init_num_samples,
                                                        hyperparam_refine_ll_delta = 3.0,
                                                        alpha = 0 )

        self.round_num = 1
        self.pulls = []

        self.arm_selector = CMAOptimizerSelector( reward_model = self.reward_model,
                                                  dim = arm_dim,
                                                  beta = beta,
                                                  bounds = [ x_lower, x_upper ],
                                                  popsize = 30 )

        self.arm_proposal = NullArmProposal()
        self.bandit = BanditInterface( arm_proposal = self.arm_proposal,
                                       reward_model = self.reward_model,
                                       arm_selector = self.arm_selector )

        # Create critique service proxy
        critique_topic = rospy.get_param( '~critic_service' )
        rospy.wait_for_service( critique_topic )
        self.critique_service = rospy.ServiceProxy( critique_topic, GetCritique, True )

        # # Plotting
        # if len( self.x_lower_lim ) == 2:
        #     self.plot_update_period = rospy.get_param('~plot_update_period', 1)
        #     self.plot_enable = True

        #     self.reward_fig = plt.figure()
        #     self.reward_ax = self.reward_fig.gca(projection='3d')
        #     self.reward_fig.show()

        #     self.pulls_fig = plt.figure()
        #     self.pulls_ax = self.pulls_fig.gca()
        #     self.pulls_ax.set_aspect('equal')
        #     self.pulls_ax_cb = None
        #     self.pulls_fig.show()

        #     self.vars_fig = plt.figure()
        #     self.vars_ax = self.vars_fig.gca()
        #     self.vars_ax.set_aspect('equal')
        #     self.vars_ax_cb = None
        #     self.vars_fig.show()

        #     x_values = np.linspace( self.x_lower_lim[0], self.x_upper_lim[0], 20 )
        #     y_values = np.linspace( self.x_lower_lim[1], self.x_upper_lim[1], 20 )
        #     self.X_vals, self.Y_vals = np.meshgrid( x_values, y_values )
        #     self.plot_arms = zip( self.X_vals.flat, self.Y_vals.flat )
        # else:
        #     print 'Can only plot reward function for 2D arms.'
        #     self.plot_enable = False

    def evaluate_input( self, inval ):
        req = GetCritiqueRequest()
        req.input = inval
        try:
            res = self.critique_service.call( req )
        except rospy.ServiceException:
            raise RuntimeError( 'Could not evaluate item: ' + str( inval ) )
        self.pulls.append( inval )
        return res.critique, res.feedback

    def execute( self ):
        while not rospy.is_shutdown() and self.round_num < self.num_rounds:
            arm = self.bandit.ask()

            rospy.loginfo( 'Round %d Evaluating arm %s' % (self.round_num,str(arm)) )
            reward, feedback = self.evaluate_input( arm )
            rospy.loginfo( 'Arm returned reward: %f feedback: %s',
                           reward, str(feedback) )
            self.bandit.tell( arm, reward )
            
            self.out_log.write( 'Round: %d Arm: %s Reward: %f\n' % 
                                (self.round_num, str(arm), reward) )
            self.out_log.flush()
            self.round_num += 1

            print 'Theta: ' + str(self.reward_model.gp.theta)
            # if self.plot_enable:
                # self.update_plot()

        self.out_log.close()

    def update_plot( self ):
        
        # NOTE We don't redraw the full GP too often since it's slow
        if self.round_num % self.plot_update_period == 0 or not hasattr(self, 'reward_means'):
            estimates = [ self.reward_model.query( arm ) for arm in self.plot_arms ]
            self.reward_means = np.reshape( np.asarray([ est[0] for est in estimates ]), self.X_vals.shape )
            self.reward_vars = np.reshape( np.asarray([ est[1] for est in estimates ]), self.X_vals.shape )
            self.reward_stds = np.sqrt( self.reward_vars )
            #self.pull_ests = [ self.reward_model.query( pull ) for pull in self.pulls ]

        # Visualize pulls in 2D
        pull_x = [ pull[0] for pull in self.pulls ]
        pull_y = [ pull[1] for pull in self.pulls ]
        
        self.pulls_ax.clear()
        pap = self.pulls_ax.pcolor( self.X_vals, self.Y_vals, self.reward_means )
        if self.pulls_ax_cb is None:
            self.pulls_ax_cb = self.pulls_fig.colorbar( pap, ax=self.pulls_ax )
        else:
            self.pulls_ax_cb.update_bruteforce( pap )
        self.pulls_ax.plot( pull_x[0:-1], pull_y[0:-1], 'k.', markersize=10, mec='w' )
        self.pulls_ax.plot( pull_x[-1], pull_y[-1], 'k.', markersize=40, mec='w' )
        self.pulls_ax.set_title('Estimated mean reward')
        self.pulls_fig.canvas.draw()
        
        self.vars_ax.clear()
        pap = self.vars_ax.pcolor( self.X_vals, self.Y_vals, self.reward_stds )
        if self.vars_ax_cb is None:
            self.vars_ax_cb = self.vars_fig.colorbar( pap, ax=self.vars_ax )
        else:
            self.vars_ax_cb.update_bruteforce( pap )
        self.vars_ax.plot( pull_x[0:-1], pull_y[0:-1], 'k.', markersize=10, mec='w' )
        self.vars_ax.plot( pull_x[-1], pull_y[-1], 'k.', markersize=40, mec='w' )
        self.vars_ax.set_title('Mean reward estimate SD')
        self.vars_fig.canvas.draw()

        # Visualize pulls in 3D
        self.reward_ax.clear()
        self.reward_ax.grid(False)
        self.reward_ax.plot_surface( self.X_vals, self.Y_vals, self.reward_means, 
                                     cmap=cm.coolwarm, alpha=1.0, rstride=1, cstride=1, 
                                     linewidth=0.1 )
        self.reward_ax.plot_surface( self.X_vals, self.Y_vals, self.reward_means + 3*self.reward_stds, 
                                     cmap=cm.coolwarm, alpha=0.4, rstride=1, cstride=1, 
                                     linewidth=0 )
        self.reward_ax.plot_surface( self.X_vals, self.Y_vals, self.reward_means - 3*self.reward_stds, 
                                     cmap=cm.coolwarm, alpha=0.4, rstride=1, cstride=1, 
                                     linewidth=0 )
        
        #pull_z = [ est[0] for est in pull_ests]
        #self.reward_ax.scatter( pull_x[-1], pull_y[-1], pull_z[-1], c='r', s=20 )
        #self.reward_ax.scatter( pull_x[0:-1], pull_y[0:-1], pull_z[0:-1], c='k', s=5 )

        self.reward_fig.canvas.draw()

if __name__=='__main__':
    rospy.init_node( 'bandit_node' )
    pbn = GPUCBBandit()
    pbn.execute()
