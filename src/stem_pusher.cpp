#include <cmath>
#include <iostream>
#include <franka/exception.h>
#include <franka/robot.h>
#include "examples_common.h"

#include "ros/ros.h"
#include "std_msgs/Float64MultiArray.h"
#include <math.h> 


int main(int argc, char** argv) {
  try {
    ros::init(argc, argv, "robot_pose_pub");
    ros::NodeHandle n;
    ros::Publisher robot_pose_pub = n.advertise<std_msgs::Float64MultiArray>("robot_pose", 1000);

    franka::Robot robot(argv[1]);
    setDefaultBehavior(robot);

    // First move the robot to a suitable joint configuration
    // std::array<double, 7> q_goal = {{0, -M_PI_4, 0, -3 * M_PI_4, 0, M_PI_2, M_PI_4}};
    std::array<double, 7> q_goal = {{0.14360167152468625, 0.1753777056788718, -0.2196985128072391, -1.365035858023108, -0.15087520535954108, 3.1017061897913636, -2.018819763140546}};
    MotionGenerator motion_generator(0.5, q_goal);
    robot.control(motion_generator);

    // Set collision behavior.
    robot.setCollisionBehavior(
        {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}}, {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
        {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}}, {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
        {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}}, {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}},
        {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}}, {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}});

    std::array<double, 16> initial_pose;
    double time = 0.0;
    double position = 0.0;
    double velocity = 0.0;
    double acceleration = 0.0;
    double last_position = 0.0;
    
    robot.control([=, &time, &initial_pose](const franka::RobotState& robot_state,
                                         franka::Duration period) mutable -> franka::CartesianPose {
      time += period.toSec();
      std::vector<double> EE_pose_vec = {robot_state.O_T_EE[0], robot_state.O_T_EE[1], robot_state.O_T_EE[2], robot_state.O_T_EE[3], robot_state.O_T_EE[4], robot_state.O_T_EE[5], robot_state.O_T_EE[6],
                                   robot_state.O_T_EE[7], robot_state.O_T_EE[8], robot_state.O_T_EE[9], robot_state.O_T_EE[10], robot_state.O_T_EE[11], robot_state.O_T_EE[12], robot_state.O_T_EE[13],
                                  robot_state.O_T_EE[14], robot_state.O_T_EE[15]};
      std_msgs::Float64MultiArray robot_pose_msg;
      robot_pose_msg.data.clear();
      robot_pose_msg.data.resize(17);
      EE_pose_vec.push_back(0.0); //this is the flag to continue/end subscribing fo python script subscriber
      robot_pose_msg.data = EE_pose_vec;
      robot_pose_pub.publish(robot_pose_msg);

      if (time == 0.0) {
        initial_pose = robot_state.O_T_EE_c;
      }

      std::array<double, 16> new_pose = initial_pose;

      if (time < 1.0){
        acceleration = -0.5;
        velocity = acceleration * time;
        position = acceleration * time * time / 2 + initial_pose[13];
        last_position = position;
      } else if (time < 2.0){
        acceleration = -0.5;
        position = acceleration * time * time / 2 + velocity * time + last_position;
      }

      // new_pose[13] = position;

      if (time >= 30.0) {
        std::cout << std::endl << "Finished motion, shutting down example" << std::endl;
        return franka::MotionFinished(new_pose);
      }
      return new_pose;
    });
  } catch (const franka::Exception& e) {
    std::cout << e.what() << std::endl;
    return -1;
  }

  return 0;
}
