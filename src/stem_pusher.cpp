#include <cmath>
#include <iostream>
#include <fstream>
#include <franka/exception.h>
#include <franka/robot.h>
#include "examples_common.h"

#include "ros/ros.h"
#include "std_msgs/Float64MultiArray.h"
#include <math.h> 
#include <Eigen/Dense>
#include <unsupported/Eigen/EulerAngles>

#include <array>
#include <limits>

typedef Eigen::EulerAngles<double, Eigen::EulerSystem<Eigen::EULER_X, Eigen::EULER_Y, Eigen::EULER_Z> > Eulerss;


float rot_value = 0;
void wrist_rot_cb(const std_msgs::Float64MultiArray& rotation)
{
  rot_value = rotation.data[0];
  // std::cout << "rotation: " << rot_value << std::endl;
}

int main(int argc, char** argv) {
  try {
    ros::init(argc, argv, "robot_pose_pub");
    ros::NodeHandle n;
    ros::Publisher robot_pose_pub = n.advertise<std_msgs::Float64MultiArray>("robot_pose", 1000);
    ros::Subscriber wrist_rot = n.subscribe("/wrist_rotation", 1000, wrist_rot_cb);

    franka::Robot robot(argv[1]);
    setDefaultBehavior(robot);

    // First move the robot to a suitable joint configuration
    std::array<double, 7> q_goal = {{0, -M_PI_4, 0, -3 * M_PI_4, 0, M_PI_2, M_PI_4}};
    // std::array<double, 7> q_goal = {{0.14360167152468625, 0.1753777056788718, -0.2196985128072391, -1.365035858023108, -0.15087520535954108, 3.1017061897913636, -2.018819763140546}};
    // std::array<double, 7> q_goal = {{-0.00635813,0.0372822,-0.0373876,-1.30535,0.0145356,2.85452,0.787199}};
    // std::array<double, 7> q_goal = {{0.000189747,-0.348658,-0.0224457,-1.67147,-0.0154906,2.95908,0.788193}};
    MotionGenerator motion_generator(0.5, q_goal);
    robot.control(motion_generator);

    // Set collision behavior.
    robot.setCollisionBehavior(
        {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}}, {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
        {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}}, {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
        {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}}, {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}},
        {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}}, {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}});

    
    std::string filename = "/home/kiyanoush/Desktop/robot_y.csv";
    std::vector<double> y_save;
    
    std::array<double, 16> initial_pose;
    double time = 0.0;
    double T = 1.2;
    double a = 0.8 / pow(T, 2);
    
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

      if (time < T/2){
        new_pose[13] = - 0.5 * a * time * time + initial_pose[13];
      } else if (time < T){
        new_pose[13] = 0.5 * a * (time - T) * (time - T) - (0.2 - initial_pose[13]);
      }

      // Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
      // Eigen::Vector3d position(transform.translation());
      // Eigen::Matrix3d rotation(transform.rotation());
      // Eigen::Vector3d euler_angles = rotation.eulerAngles(0, 1, 2);

      // Eulerss euler_angles2;
      // euler_angles2 = {euler_angles(0), euler_angles(1), euler_angles(2)};
      // Eigen::Matrix3d rotationMatrix = euler_angles2.toRotationMatrix();

      // if (rot_value == 0){
      //     euler_angles[1] += 0.0001 * time;
      // }
      
      // new_pose[0] = rotationMatrix(0, 0);
      // new_pose[1] = rotationMatrix(1, 0);
      // new_pose[2] = rotationMatrix(2, 0);
      // new_pose[4] = rotationMatrix(0, 1);
      // new_pose[5] = rotationMatrix(1, 1);
      // new_pose[6] = rotationMatrix(2, 1);
      // new_pose[8] = rotationMatrix(0, 2);
      // new_pose[9] = rotationMatrix(1, 2);
      // new_pose[10] = rotationMatrix(2, 2);

      // std::cout << rotationMatrix(1, 2) << std::endl;
      y_save.push_back(new_pose[13]);

      ros::spinOnce();
      
      if (time >= T) {
        // std::cout << std::endl << "Finished motion, shutting down example" << std::endl;
        int b = 0;
        unsigned int milisecond = 500;
        std_msgs::Float64MultiArray robot_pose_msg;
        robot_pose_msg.data.clear();
        robot_pose_msg.data.resize(17);
        int n = EE_pose_vec.size();
        EE_pose_vec[n-1] = 1.0;
        robot_pose_msg.data = EE_pose_vec ;
        
        while(b<5000){  // 5 seconds - ish   
          usleep(milisecond);
          b++;
          robot_pose_pub.publish(robot_pose_msg);
          ros::spinOnce();
        }
        std::ofstream RobotPoseCSV;   
        RobotPoseCSV.open(filename);
        RobotPoseCSV << "y" << std::endl;
        for (int i = 0; i < y_save.size(); i++)
        {
          RobotPoseCSV << y_save[i] << std::endl;
        }
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