#!/usr/bin/env python

import rospy
import optim
from percepto_msgs.srv import GetCritique


class BlockOptimizationWrapper(object):
    """Wraps a GetCritique call and provides multiple GetCritique services
    corresponding to blocks of the base GetCritique server.
    """

    def __init__(self):
        interface_info = rospy.get_param('~interface')
        self.interface = optim.CritiqueInterface(**interface_info)

        self.servers = []
        block_info = dict(rospy.get_param('~blocks'))
        self.current_params = {}
        self.blocks = {}
        for b_name, b_info in block_info.iteritems():
            params = b_info['parameters']
            init = b_info['initial']

            # Add all parameters to current param dict
            if b_name in self.blocks:
                raise RuntimeError('Block %s repeated!'% b_name)
            self.blocks[b_name] = params

            for param, ival in zip(params, init):
                if param in self.current_params:
                    raise RuntimeError('Parameter %s repeated!' % param)
                self.current_params[param] = ival

            rospy.loginfo('Block %s with params %s init %s',
                          b_name, str(params), str(init))

            def cb(req):
                return self.block_callback(block=b_name,
                                           values=req.input,
                                           names=req.names)
            topic = '~get_%s_critique' % b_name
            self.servers.append(rospy.Service(topic, GetCritique, cb))

    def block_callback(self, block, values, names):
        if len(names) == 0:
            names = self.blocks[block]

        block_params = self.blocks[block]
        if any([(n not in block_params) for n in names]):
            rospy.logerr('Received parameters %s not in block %s consisting of %s',
                         str(names), block, str(block_params))
            return None

        for n, v in zip(names, values):
            self.current_params[n] = v

        return self.get_critique()

    def get_critique(self):
        n = self.current_params.keys()
        x = [self.current_params[ni] for ni in n]
        return self.interface.raw_call(x)


if __name__ == '__main__':
    rospy.init_node('block_optimization_wrapper')
    wrapper = BlockOptimizationWrapper()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
