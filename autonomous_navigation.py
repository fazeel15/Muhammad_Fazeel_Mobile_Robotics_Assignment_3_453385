#!/usr/bin/env python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import numpy as np
from scipy.ndimage import gaussian_filter1d

def gaussian(data):
    return gaussian_filter1d(data, sigma=.3)

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0.0
        self.integral = 0.0

    def compute(self, setpoint, measured_value, dt):
        error = setpoint - measured_value
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

class ExplorerNode(Node):
    def __init__(self):
        super().__init__('explorer_node')

        self.laser_sub = self.create_subscription(LaserScan, 'scan', self.laser_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        self.obstacle_threshold = 0.55 
        self.move_speed = 0.2
        self.rotation_speed = 0.45  # rad/s
        
        self.pid_controller = PIDController(kp=1.0, ki=0.0, kd=0.1)
        self.previous_time = self.get_clock().now()

    def laser_callback(self, msg: LaserScan):
        cmd = Twist()
        ranges = np.array(msg.ranges)
        ranges = ranges - 0.180

        current_time = self.get_clock().now()
        dt = (current_time - self.previous_time).nanoseconds / 1e9
        self.previous_time = current_time

        if all(ranges[0:20] > self.obstacle_threshold) and all(ranges[-20:] > self.obstacle_threshold):
            cmd.linear.x = self.move_speed
            cmd.angular.z = 0.0
        else:
            front_distance = min(min(ranges[0:20]), min(ranges[-20:]))
            error = self.obstacle_threshold - front_distance

            angular_z = self.pid_controller.compute(0, error, dt)

            cmd.linear.x = self.move_speed / 2.0
            cmd.angular.z = angular_z

        self.cmd_vel_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    explorer_node = ExplorerNode()
    rclpy.spin(explorer_node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()