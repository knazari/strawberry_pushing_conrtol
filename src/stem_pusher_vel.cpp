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


float stem_pose = 0.0;
void wrist_rot_cb(const std_msgs::Float64MultiArray& rotation)
{
  stem_pose = rotation.data[0];
}

int main(int argc, char** argv) {
  try {
    ros::init(argc, argv, "robot_pose_pub");
    ros::NodeHandle n;
    ros::Publisher robot_pose_pub = n.advertise<std_msgs::Float64MultiArray>("robot_pose", 1000);
    ros::Publisher robot_vel_pub = n.advertise<std_msgs::Float64MultiArray>("robot_vel", 1000);
    ros::Subscriber wrist_rot = n.subscribe("/stem_pose", 1000, wrist_rot_cb);

    franka::Robot robot(argv[1]);
    setDefaultBehavior(robot);

    // First move the robot to a suitable joint configuration
    // std::array<double, 7> q_goal = {{0, -M_PI_4, 0, -3 * M_PI_4, 0, M_PI_2, M_PI_4}};
    // std::array<double, 7> q_goal = {{0.14360167152468625, 0.1753777056788718, -0.2196985128072391, -1.365035858023108, -0.15087520535954108, 3.1017061897913636, -2.018819763140546}};
    std::array<double, 7> q_goal = {{-0.00635813,0.0372822,-0.0373876,-1.30535,0.0145356,2.85452,0.787199}}; // strawberry stem
    // std::array<double, 7> q_goal = {{0.000189747,-0.348658,-0.0224457,-1.67147,-0.0154906,2.95908,0.788193}}; // behind stem to avoid contact
    MotionGenerator motion_generator(0.5, q_goal);
    robot.control(motion_generator);

    // Set collision behavior.
    robot.setCollisionBehavior(
        {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}}, {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
        {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}}, {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
        {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}}, {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}},
        {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}}, {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}});

    
    std::string filename = "/home/kiyanoush/Desktop/robot_cart_vel.csv";
    std::vector< std::vector<double> > robot_traj;
    
    double time = 0.0;
    double T = 2;
    double a = 0.8 / pow(T, 2);
    double v_y = 0;
    double w_z = 0;
    
    double stem_now = 0.0;
    double stem_prev = 0.0;
    double e = 0;
    double e_dot = 0;
    double w_z_des = 0;
    double k_p = 0.5;
    double k_d = 0.15;
    
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

      // linear motion bang-bang reference trajectory
      if (time < T/2){
        v_y = - a * time;
      } else if (time < T){
        v_y = a * time - a * T;
      } else if (time < 1.2 * T){ // wait for a bit at the end of motion and then return
        v_y = 0.0;
      } else if (time < 1.7 * T){ // here the task is finished but we move the robot to the initial pose
        v_y = a * (time - 1.2 * T);
      } else if (time < 2.2 * T){
        v_y = - a * time + 2.2 * a * T;
      }

      if (time < T){ // do it only for the forward path

        // rotational motion PID controller
        if (stem_pose != 0.0){
          stem_now = stem_pose;
          if (stem_now != stem_prev){
            e = -(stem_now - 0.5); // negative because rotation around z is positive in opposite
            e_dot = stem_now - stem_prev;
            if (stem_prev == 0){
              e_dot = 0.0;
            }
            w_z_des = k_p * e + k_d * e_dot;
            // w_z_des = k_p * e;
          }
          
          w_z = w_z + 0.0625 * (w_z_des - w_z);
          stem_prev = stem_now;

          std::cout << "first  : " << k_p * e << std::endl;
          std::cout << "second : " << k_d * e_dot << std::endl;

        }

        // log traj data
        std::vector<double> robot_vel_vec = {v_y, w_z, w_z_des};
        robot_traj.push_back(robot_vel_vec);

        std_msgs::Float64MultiArray robot_vel_msg;
        robot_vel_msg.data = robot_vel_vec;
        robot_vel_pub.publish(robot_vel_msg);
      }

      ros::spinOnce();

      franka::CartesianVelocities output = {{0, v_y, 0, 0, 0, w_z}};

      if (time > T and time < 2.2*T) {
        std_msgs::Float64MultiArray robot_pose_msg;
        robot_pose_msg.data.clear();
        robot_pose_msg.data.resize(17);
        int n = EE_pose_vec.size();
        EE_pose_vec[n-1] = 1.0;
        robot_pose_msg.data = EE_pose_vec ;
        robot_pose_pub.publish(robot_pose_msg);
        
      } else if (time > 2.2 * T){
        
        std::ofstream RobotPoseCSV;   
        RobotPoseCSV.open(filename);
        RobotPoseCSV << "v_y" << "," << "w_z" << "," << "w_z_des" << std::endl;
        for (int i = 0; i < robot_traj.size(); i++)
        {
          RobotPoseCSV << robot_traj[i][0] << "," << robot_traj[i][1] << "," << robot_traj[i][2]<< std::endl;
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