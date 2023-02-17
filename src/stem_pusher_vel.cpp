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


float rot_value = 0;
void wrist_rot_cb(const std_msgs::Float64MultiArray& rotation)
{
  rot_value = rotation.data[0];
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
    // std::array<double, 7> q_goal = {{0, -M_PI_4, 0, -3 * M_PI_4, 0, M_PI_2, M_PI_4}};
    // std::array<double, 7> q_goal = {{0.14360167152468625, 0.1753777056788718, -0.2196985128072391, -1.365035858023108, -0.15087520535954108, 3.1017061897913636, -2.018819763140546}};
    std::array<double, 7> q_goal = {{-0.00635813,0.0372822,-0.0373876,-1.30535,0.0145356,2.85452,0.787199}};
    // std::array<double, 7> q_goal = {{0.000189747,-0.348658,-0.0224457,-1.67147,-0.0154906,2.95908,0.788193}};
    MotionGenerator motion_generator(0.5, q_goal);
    robot.control(motion_generator);

    // Set collision behavior.
    robot.setCollisionBehavior(
        {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}}, {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
        {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}}, {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
        {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}}, {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}},
        {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}}, {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}});

    
    std::string filename = "/home/kiyanoush/Desktop/robot_v_y.csv";
    std::vector<double> v_y_save;
    
    double time = 0.0;
    double T = 1.2;
    double a = 0.8 / pow(T, 2);
    double v_y = 0;
    double w_z = 0;
    double T_rot;
    double alpha_z;
    double t_start_rot;
    bool rotate_old = false;
    bool rotate_new = false;
    
    robot.control([=, &time](const franka::RobotState& robot_state,
                                         franka::Duration period) mutable -> franka::CartesianVelocities {
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

      if (rot_value != 0){
          rotate_new = true;
      }

      if (rotate_old != rotate_new){
        std::cout << "start rotate now ..." << std::endl;
        t_start_rot = time;
        T_rot = T - time;
        // alpha_z = 2 * 3.14 / (3 * pow(T, 2)); // 30 deg wrist rotation
        alpha_z = 3.14 / pow(T, 2); // 45 degree wrist rotation
      }

      if (time < T/2){
        v_y = - a * time;
      } else if (time < T){
        v_y = a * time - a * T;
      }
      
      if (rotate_new == true){
        if ((time - t_start_rot) < T_rot/2){
          w_z = -alpha_z * (time - t_start_rot);
          
      } else if ((time - t_start_rot) < T_rot){
          w_z = alpha_z * (time - t_start_rot) - alpha_z * T_rot;
      }
        
      }
     
      v_y_save.push_back(v_y);

      rotate_old = rotate_new;
      
      ros::spinOnce();

      franka::CartesianVelocities output = {{0, v_y, 0, 0, 0, w_z}};
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
        RobotPoseCSV << "v_y" << std::endl;
        for (int i = 0; i < v_y_save.size(); i++)
        {
          RobotPoseCSV << v_y_save[i] << std::endl;
        }
        return franka::MotionFinished(output);
      }
      return output;
    });
  } catch (const franka::Exception& e) {
    std::cout << e.what() << std::endl;
    return -1;
  }
  return 0;
}