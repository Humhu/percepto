#!/usr/bin/env python

import dill
import pickle
import rospy, sys, math
import numpy as np

from percepto_msgs.srv import GetCritique, GetCritiqueRequest, GetCritiqueResponse
import optim.genetic as optgen
from itertools import izip

# TODO Update
class GeneticOptimizer:
    """Evolutionary optimization optimizer.

    Interfaces with an optimization problem through the GetCritique service.
    """
    def __init__(self):
        # Seed RNG if specified
        seed = rospy.get_param('~random_seed', None)
        if seed is None:
            rospy.loginfo('No random seed specified. Using default behavior.')
        else:
            rospy.loginfo('Initializing with random seed: ' + str(seed))
            np.random.seed(seed)

        self.save_period = rospy.get_param('~save_period', 1)

        self.input_dim = rospy.get_param('~input_dimension')
        self.input_lower = rospy.get_param('~input_lower_bound')
        self.input_upper = rospy.get_param('~input_upper_bound')
        if not np.iterable(self.input_lower):
            self.input_lower = [self.input_lower]*self.input_dim
        self.input_lower = np.asarray(self.input_lower)
        if not np.iterable(self.input_upper):
            self.input_upper = [self.input_upper]*self.input_dim
        self.input_upper = np.asarray(self.input_upper)

        checker_func = self.check_input

        self.prog_path = rospy.get_param('~progress_path', None)
        self.out_path = rospy.get_param('~output_path')

        crossover_rate = rospy.get_param('~crossover_rate', 0.5)
        crossover_func = lambda x, y: optgen.uniform_crossover(x, y, crossover_rate)

        mutate_cov = float(rospy.get_param('~mutate_cov', 0.1))
        mutate_func = lambda x: optgen.gaussian_mutate(x, mutate_cov)

        selection_k = rospy.get_param('~selection_k', None)
        selection_func = lambda N, w: optgen.tournament_selection(N, w, selection_k)

        crossover_prob = rospy.get_param('~crossover_prob', 0.6)
        init_popsize = rospy.get_param('~init_popsize')
        run_popsize = rospy.get_param('~run_popsize', init_popsize)
        elitist = rospy.get_param('~elitist', False)
        verbose = rospy.get_param('~verbose', False)

        self.max_iters = rospy.get_param('~convergence/max_iters', 100)
        self.iter_counter = 0

        self.optimizer = optgen.GeneticOptimizer(crossover_func=crossover_func,
                                                 mutate_func=mutate_func,
                                                 selection_func=selection_func,
                                                 checker_func=checker_func,
                                                 prob_cx=crossover_prob,
                                                 popsize=run_popsize,
                                                 # elitist=elitist,
                                                 verbose=verbose)
        
        initial_pop = [self.sample_input() for i in range(init_popsize)]
        self.optimizer.initialize(initial_pop)

        self.rounds = []
        self.prog_path = rospy.get_param('~progress_path', None)
        self.out_path = rospy.get_param('~output_path')

    def check_input(self, x):
        x[x > self.input_upper] = self.input_upper[x > self.input_upper]
        x[x < self.input_lower] = self.input_lower[x < self.input_lower]
        return x

    def sample_input(self):
        out = np.random.rand(self.input_dim)
        return out * (self.input_upper - self.input_lower) + self.input_lower

    def save(self):
        if self.prog_path is not None:
            rospy.loginfo('Saving progress at %s...', self.prog_path)
            prog = open(self.prog_path, 'wb')
            pickle.dump(self, prog)
            prog.close()

        rospy.loginfo('Saving output at %s...', self.out_path)
        out = open(self.out_path, 'wb')
        pickle.dump(self.rounds, out)
        out.close()

    def is_done(self):
        return self.iter_counter > self.max_iters

    def execute(self, eval_cb):
        while not self.is_done() and not rospy.is_shutdown():
            rospy.loginfo('Beginning iteration %d/%d', self.iter_counter+1, self.max_iters)
            queries = self.optimizer.ask()
            critiques = []
            for i, x in enumerate(queries):
                rospy.loginfo('Evaluating %d/%d...', i+1, len(queries))
                critiques.append(eval_cb(x)[0])
            fitnesses = critiques
            self.optimizer.tell(fitnesses)

            self.rounds.append((queries, fitnesses))
            self.save()
            self.iter_counter += 1
        
        self.rounds.append(self.optimizer.ask())
        self.save()

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
    rospy.init_node('genetic_optimization')

    # See if we're resuming
    if rospy.has_param('~load_path'):
        data_path = rospy.get_param('~load_path')
        data_log = open(data_path, 'rb')
        rospy.loginfo('Found load data at %s...', data_path)
        bopt = pickle.load(data_log)
    else:
        rospy.loginfo( 'No resume data specified. Starting new optimization...' )
        bopt = GeneticOptimizer()

    # Create interface to optimization problem
    critique_topic = rospy.get_param('~critic_service')
    rospy.loginfo('Waiting for service %s...', critique_topic)
    rospy.wait_for_service(critique_topic)
    rospy.loginfo('Connected to service %s.', critique_topic)
    critique_proxy = rospy.ServiceProxy( critique_topic, GetCritique, True )
    eval_cb = lambda x : evaluate_input(critique_proxy, x)

    bopt.execute(eval_cb)