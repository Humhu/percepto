#!/usr/bin/env python

import sys
import pickle
import rospy

from percepto_msgs.srv import RunOptimization


class BlockOptimizer(object):
    """Iteratively calls subordinate optimizers to perform block optimization.

    Uses the RunOptimization service to interface with subordinates.
    """

    def __init__(self):
        self.order = list(rospy.get_param('~order'))
        self.blocks = [None] * len(self.order)
        block_info = dict(rospy.get_param('~blocks'))
        for name, topic in block_info.iteritems():
            try:
                ind = self.order.index(name)
            except ValueError:
                rospy.logerr('Could not find block %s in order %s',
                             name, str(self.order))
                sys.exit(-1)
            self.blocks[ind] = rospy.ServiceProxy(topic, RunOptimization, True)

        self.use_warm_starts = rospy.get_param('~use_warm_starts')
        self.max_iters = rospy.get_param('~convergence/max_iters')
        self.rounds = []

    def is_finished(self):
        return len(self.rounds) >= self.max_iters

    @property
    def round_index(self):
        return len(self.rounds) % len(self.blocks)

    def execute(self, out_file):
        while not rospy.is_shutdown() and not self.is_finished():
            block = self.order[self.round_index]
            rospy.loginfo('Optimizing %s...', block)
            try:
                res = self.blocks[self.round_index].call(self.use_warm_starts)
            except rospy.ServiceException:
                rospy.logerr('Could not optimize %s', block)
                continue

            if not res.success:
                rospy.logwarn('Optimization failed!')
                continue
            obj = res.objective
            sol = res.solution

            self.rounds.append((block, obj, sol))
        pickle.dump(self.rounds, out_file)


if __name__ == '__main__':
    rospy.init_node('block_optimizer')

    out_path = rospy.get_param('~output_path')
    out_file = open(out_path, 'w')

    optimizer = BlockOptimizer()
    try:
        optimizer.execute(out_file)
    except rospy.ROSInterruptException:
        pass

    out_file.close()
