"""Software for storing, loading, and transferring tensorflow parameters
"""

import tensorflow as tf
import paramiko
import rospy

from adel.msg import CheckpointNotification


class ModelSaver(object):
    """Convenience class wrapping tf.train.Saver and paramiko.SSHClient
    to combine saving, restoring, and transferring checkpoint files.

    Parameters
    ==========
    variables : list of tf.Variables 
        The variables to store/restore
    store_path : string (default '/tmp')
        The base path to store checkpoint files at
    sess : tf.Session or None (default None)
        If given, sets the default session for the ModelSaver
    """

    def __init__(self, variables, store_path='/tmp', sess=None):
        self.store_path = store_path
        self.saver = tf.train.Saver(variables)
        self.sess = None

        # For transferring to remote
        self.ssh = None
        self.remote_path = ''
        self.notification_pub = None

        # For listening to remote
        self.notification_sub = None

    def set_session(self, sess):
        """Set the default session for this ModelSaver
        """
        self.sess = sess

    def save(self, sess=None):
        """Saves the session using this saver. If no session specified,
        uses the default session.
        """
        if sess is None:
            sess = self.sess
        self.saver.save(sess=sess, save_path=self.store_path)

    def restore(self, path, sess=None):
        """Restores to the session using this saver. If no session specified,
        uses the default session.
        """
        if sess is None:
            sess = self.sess
        self.saver.restore(sess, path)

    def restore_latest(self, sess=None):
        """Restores from the latest checkpoint.
        """
        self.restore(sess, self.saver.last_checkpoints[-1])

    def enable_remote_transfer(self, remote_name, user_name, password,
                               port=22, remote_path='/tmp',
                               notification_topic='/checkpoint_status'):
        """Enable transfer to the remote through SSH and SFTP. Opens an
        SSH connection, enables SFTP, and advertises a ROS topic
        to notify the remote about transfers.

        Parameters
        ==========
        remote_name : string
            Remote hostname or IP
        user_name : string
            Username to use for logging into the remote
        password : string
            Password to use for logging into the remote
        port : int (default 22)
            Port to use for SSH
        remote_path : string (default '/tmp')
            Base path to use for transferring files to the remote
        notification_topic : string (default '/checkpoint_status')
        """
        if self.ssh is not None:
            self.ssh.close()

        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(hostname=remote_name,
                         username=user_name,
                         password=password,
                         port=port)
        self.ssh.open_sftp()
        self.remote_path = remote_path

        self.notification_pub = rospy.Publisher(notification_topic,
                                                CheckpointNotification,
                                                queue_size=10)

    def transfer_latest_remote(self):
        """Transfers the last saved checkpoint to the remote. Sends
        file to remote_path specified when enabling and sends
        a CheckpointNotification message with the file path.
        """
        if self.ssh is None:
            raise RuntimeError('SSH client is not connected! Cannot transfer.')

        if len(self.saver.last_checkpoints) == 0:
            print 'No checkpoints to transfer!'
            return

        latest = self.saver.last_checkpoints[-1]
        fname = latest.split('/')[-1]
        dest_file = '%s/%s' % (self.remote_path, fname)

        print 'Transferring %s over SFTP to %s...' % (latest, dest_file)
        self.ssh.put(latest, dest_file)
        print 'Transfer complete!'

        msg = CheckpointNotification()
        msg.header.frame_id = 'saver'
        msg.header.stamp = rospy.Time.now()
        msg.checkpoint_path = dest_file

    def enable_remote_receive(self, notification_topic):
        """Enables listening for remote transfers
        """
        self.notification_sub = rospy.Subscriber(notification_topic,
                                                 CheckpointNotification,
                                                 callback=self.receive_callback,
                                                 queue_size=10)

    def receive_callback(self, msg):
        print 'Restoring from received file %s' % msg.checkpoint_path
        self.restore(path=msg.checkpoint_path)