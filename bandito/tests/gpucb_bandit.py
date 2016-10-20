#! /usr/bin/env python
import rospy, random
import numpy as np
from itertools import izip, product

from bandito.bandits import BanditInterface
from bandito.arm_proposals import *
from bandito.arm_selectors import *
from bandito.reward_models import GaussianProcessRewardModel

from percepto_msgs.srv import GetCritique, GetCritiqueRequest

from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from gp_extras.kernels import HeteroscedasticKernel

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
        self.arm_proposal = UniformArmProposal( bounds=zip(self.arm_lower_lims, self.arm_upper_lims),
                                                num_arms=num_arms )

        # TODO Think about how to choose the prototypes in higher dims...
        #x_values = np.linspace( self.arm_lower_lims[0], self.arm_upper_lims[0], 3 )
        #y_values = np.linspace( self.arm_lower_lims[1], self.arm_upper_lims[1], 3 )
        #X,Y = np.meshgrid( x_values, y_values )
        #prototypes = np.asarray( zip( X.flat, Y.flat ) )        

        self.white = WhiteKernel( 1.0, (1e-3, 1e3) )
        #self.hetero = HeteroscedasticKernel( prototypes, sigma_2=1.0, sigma_2_bounds=(1e-3, 1e3),
        #                                     gamma=1.0, gamma_bounds="fixed" )
        self.kernel_base = ConstantKernel( 1.0, (1e-3, 1e1) ) * RBF( 1e-2, (1e-3, 1e-1) )
        self.kernel_noisy = self.kernel_base  + self.white
        self.reward_model = GaussianProcessRewardModel( kernel = self.kernel_noisy,
                                                        kernel_noiseless = self.kernel_base,
                                                        hyperparam_min_samples = 100,
                                                        hyperparam_refine_ll_delta = 3.0,
                                                        alpha = 0 )

        self.round_num = 1
        self.pulls = []
        # self.beta_func = lambda : UCBSelector.finite_beta_func( self.round_num, num_arms, 1e-2 )
        # self.arm_selector = UCBSelector( beta_func = self.beta_func,
        #                                 reward_model = self.reward_model )
        self.arm_selector = CMAOptimizerSelector( reward_model = self.reward_model,
                                                  dim = len( self.arm_lower_lims ),
                                                  beta = 1.0,
                                                  bounds = [-1, 1],
                                                  popsize = 30 )

        self.bandit = BanditInterface( arm_proposal = self.arm_proposal,
                                       reward_model = self.reward_model,
                                       arm_selector = self.arm_selector )

        # Create critique service proxy
        critique_topic = rospy.get_param( '~critic_service' )
        rospy.wait_for_service( critique_topic )
        self.critique_service = rospy.ServiceProxy( critique_topic, GetCritique, True )

        # Plotting
        if len( self.arm_lower_lims ) == 2:
            self.plot_enable = True

            self.reward_fig = plt.figure()
            self.reward_ax = self.reward_fig.gca(projection='3d')
            self.reward_fig.show()

            self.pulls_fig = plt.figure()
            self.pulls_ax = self.pulls_fig.gca()
            self.pulls_ax.set_aspect('equal')
            self.pulls_ax_cb = None
            self.pulls_fig.show()

            self.vars_fig = plt.figure()
            self.vars_ax = self.vars_fig.gca()
            self.vars_ax.set_aspect('equal')
            self.vars_ax_cb = None
            self.vars_fig.show()

            x_values = np.linspace( self.arm_lower_lims[0], self.arm_upper_lims[0], 20 )
            y_values = np.linspace( self.arm_lower_lims[1], self.arm_upper_lims[1], 20 )
            self.X_vals, self.Y_vals = np.meshgrid( x_values, y_values )
            self.plot_arms = zip( self.X_vals.flat, self.Y_vals.flat )
        else:
            print 'Can only plot reward function for 2D arms.'
            self.plot_enable = False

    def evaluate_input( self, inval ):
        req = GetCritiqueRequest()
        req.input = inval
        try:
            res = self.critique_service.call( req )
        except rospy.ServiceException:
            raise RuntimeError( 'Could not evaluate item: ' + str( inval ) )
        self.pulls.append( inval )
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

            print 'Theta: ' + str(self.reward_model.gp.theta)
            if self.plot_enable:
                self.update_plot()

        self.out_log.close()

    def update_plot( self ):
        
        # NOTE We don't redraw the full GP too often since it's slow
        if self.round_num % 10 == 0 or not hasattr(self, 'reward_means'):
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
        self.pulls_ax.plot( pull_x[-1], pull_y[-1], 'kx', markersize=20, mew=4 )
        self.pulls_ax.set_title('Estimated mean reward')
        self.pulls_fig.canvas.draw()
        
        self.vars_ax.clear()
        pap = self.vars_ax.pcolor( self.X_vals, self.Y_vals, self.reward_stds )
        if self.vars_ax_cb is None:
            self.vars_ax_cb = self.vars_fig.colorbar( pap, ax=self.vars_ax )
        else:
            self.vars_ax_cb.update_bruteforce( pap )
        self.vars_ax.plot( pull_x[0:-1], pull_y[0:-1], 'k.', markersize=10, mec='w' )
        self.vars_ax.plot( pull_x[-1], pull_y[-1], 'kx', markersize=20, mew=4 )
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
