#!/usr/bin/env python
import rospy
from threading import Lock
from percepto_msgs.msg import RewardStamped
from infitu.srv import SetRecording

class RewardAccumulator:
    """Listens to and sums or integrates a stream of rewards."""

    def __init__( self ):
        self.Reset()
        self.lock = Lock()

        self.default_on_empty = rospy.get_param( '~default_on_empty', False )
        self.default_value = float( rospy.get_param( '~default_value', 0 ) )

        self.time_integrate = rospy.get_param( '~mode/time_integrate', False )
        self.normalize_time = rospy.get_param( '~mode/normalize_time', False )
        self.normalize_count = rospy.get_param( '~mode/normalize_count', False )

        buff_size = rospy.get_param( '~buffer_size', 100 )
        self.sub = rospy.Subscriber( 'reward', RewardStamped, self.RewardCallback,
                                     queue_size = buff_size )

        self.recording_service = rospy.Service( '~set_recording', 
                                                SetRecording, 
                                                self.RecordCallback )

    def Reset( self ):
        self.acc = 0
        self.duration = 0
        self.count = 0
        self.last_reward = None

    def RewardCallback( self, msg ):
        with self.lock:
            if self.last_reward is None:
                self.last_reward = msg
            dt = ( msg.header.stamp - self.last_reward.header.stamp ).to_sec()
            self.count += 1
            self.duration += dt

            if self.time_integrate:
                self.acc += ( msg.reward + self.last_reward.reward ) * 0.5 * dt
            else:
                self.acc += msg.reward

            self.last_reward = msg

    def RecordCallback( self, req ):
        with self.lock:
            # If we are starting recording, reset state
            if req.enable_recording:
                self.Reset()
                return 0

            if self.default_on_empty and self.count == 0:
                return self.default_value

            # Else return the state
            evaluation = self.acc
            if self.normalize_time:
                if self.duration == 0:
                    evaluation = self.default_value
                else:
                    evaluation = evaluation / self.duration
            if self.normalize_count:
                if self.count == 0:
                    evaluation = self.default_value
                else:
                    evaluation = evaluation / self.count
            return evaluation

if __name__=="__main__":
    rospy.init_node( "reward_accumulator" )
    try:
        ra = RewardAccumulator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
