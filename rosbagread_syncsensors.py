import rclpy
from rclpy.node import Node
from message_filters import ApproximateTimeSynchronizer, Subscriber

from std_msgs.msg import String
from sensor_msgs.msg import LaserScan, Imu 
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import pickle
import signal
import datetime


class TopicQueuer:
    def __init__(self):
        self.Lscan=[]
        self.Limu=[]
        self.Lodom=[]
        self.Lcmd=[]
    
    def save(self):
        print("Saving")
        with open("houseScan_deleteit.pkl",'wb') as fh:
            pickle.dump({'scan':self.Lscan,'imu':self.Limu,'odom':self.Lodom,'cmd_vel':self.Lcmd},fh)        
            
    def scan_listener_callback(self,msg):
        T=datetime.datetime.fromtimestamp(msg.header.stamp.sec+1e-9*msg.header.stamp.nanosec)
        print("scan time = ",T.strftime("%Y-%m-%d %H:%M:%S.%f").rstrip('0'))
        self.Lscan.append({'ranges':msg.ranges,
    		'intensities':msg.intensities,
    		'range_max':msg.range_max,
    		'range_min':msg.range_min,
    		'scan_time':msg.scan_time,
    		'time_increment':msg.time_increment,
    		'angle_increment':msg.angle_increment,
    		'angle_max':msg.angle_max,
    		'angle_min':msg.angle_min,
        'time': T 
    		})
    def imu_listener_callback(self,msg):
        T=datetime.datetime.fromtimestamp(msg.header.stamp.sec+1e-9*msg.header.stamp.nanosec)
        print("imu time = ",T.strftime("%Y-%m-%d %H:%M:%S.%f").rstrip('0'))
        self.Limu.append({'w':[msg.angular_velocity.x,msg.angular_velocity.y,msg.angular_velocity.z],
    		'a':[msg.linear_acceleration.x,msg.linear_acceleration.y,msg.linear_acceleration.z],
            'time': T 
    		})
    def imu_scan_callback(self,imumsg,scanmsg):
        Timu=datetime.datetime.fromtimestamp(imumsg.header.stamp.sec+1e-9*imumsg.header.stamp.nanosec)
        Tscan=datetime.datetime.fromtimestamp(scanmsg.header.stamp.sec+1e-9*scanmsg.header.stamp.nanosec)
        print("----------------")
        print("imu time = ",Timu.strftime("%Y-%m-%d %H:%M:%S.%f").rstrip('0'))
        print("scan time = ",Tscan.strftime("%Y-%m-%d %H:%M:%S.%f").rstrip('0'))
        
    def odom_listener_callback(self,msg):
        T=datetime.datetime.fromtimestamp(msg.header.stamp.sec+1e-9*msg.header.stamp.nanosec)
        print("odom time = ",T.strftime("%Y-%m-%d %H:%M:%S.%f").rstrip('0'))
        self.Lodom.append({'trans':[msg.pose.pose.position.x,msg.pose.pose.position.y,msg.pose.pose.position.z],
    		'q':[msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,msg.pose.pose.orientation.z,msg.pose.pose.orientation.w],
            'time': T 
    		})
    def cmd_listener_callback(self,msg):
        # T=datetime.datetime.fromtimestamp(msg.header.stamp.sec+1e-9*msg.header.stamp.nanosec)
        # print(T.strftime("%Y-%m-%d %H:%M:%S.%f").rstrip('0'))
        self.Lcmd.append({'v':[msg.linear.x,msg.linear.y,msg.linear.z],
    		'Om':[msg.angular.x,msg.angular.y,msg.angular.z],
            # 'time': T 
    		})    
        




def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('turtlebotsubscriber')
    ts=TopicQueuer()
    node.create_subscription(LaserScan,'scan',ts.scan_listener_callback)
    node.create_subscription(Imu,'imu',ts.imu_listener_callback)
    node.create_subscription(Odometry,'odom',ts.odom_listener_callback)
    node.create_subscription(Twist,'cmd_vel',ts.cmd_listener_callback)
    
    # scan_sub = Subscriber(node,"odom", LaserScan)
    # imu_sub = Subscriber(node,"imu", Imu)

    # ats = ApproximateTimeSynchronizer([imu_sub, scan_sub], 5, 0.03)
    # ats.registerCallback(ts.imu_scan_callback)
    
    
    
    # minimal_subscriber = ScanSubscriber()
    # rclpy.get_default_context().on_shutdown(minimal_subscriber.save)
    # signal.signal(signal.SIGINT, minimal_subscriber.save)
    i =0 
    # rclpy.spin(minimal_subscriber)
    while True:
        i+=1
        try:
        	rclpy.spin_once(node,timeout_sec=0.01)
        except KeyboardInterrupt:
        	pass
        if i>500:
            break
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    # minimal_subscriber.save()
    # minimal_subscriber.destroy_node()
    
    print("saving and closeing")
    rclpy.shutdown()


if __name__ == '__main__':
    main()