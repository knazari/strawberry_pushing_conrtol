##### THIS IS THE SCRIPT TO CREATE THE FIGURES FOR IROS2023 SUBMISSION.
##### FOR FIGURE 4 IN THE PAPER (SINGLE STRAWBERRY RESULTS) USE: TIP: REACTIVE 005, PROACTIVE: 002, AND OPENLOOP: 013 + 0.2
#####                                                            BASE: REACTIVE 006, PROACTIVE: 005 - 1.4, AND OPENLOOP: 015 - 0.4 (FOR ACTION USE REACTIVE: 007)



import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def drop_missed_frames(stem):
    stem_pose = np.copy(stem)
    zero_indeces = np.where(stem==0)[0]
    non_zero_indeces = np.where(stem!=0)[0]
    for i in zero_indeces:
        substitude_index = find_nearest(non_zero_indeces, i)
        stem_pose[i] = stem_pose[substitude_index]
    
    return stem_pose

def smooth(x,window_len=7,window='hanning'):
    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='valid')
    #return y
    return y[int((window_len/2-1)):-int((window_len/2))]

data_path = "/home/kiyanoush/Cpp_ws/src/haptic_finger_control/RT-Data"


# contact_initialised = 51 # this is for action plot
plt.rcParams["figure.figsize"] = (6.5, 5)
fig = plt.figure()
ax  = fig.add_subplot(111)
contact_initialised = 53

open_loop_stem_pose = np.load(data_path + "/proactive/final_tests" + "/024" + "/localisation_offline.npy")

reactive_robot_pose = np.load(data_path + "/proactive/final_tests" + "/025" +  "/robot_pose.npy")
reactive_robot_vel  = np.load(data_path + "/proactive/final_tests" + "/025" +  "/robot_velocity.npy")
reactive_stem_pose  = np.load(data_path + "/proactive/final_tests" + "/025" +  "/localisation_offline.npy")

proactive_robot_pose = np.load(data_path + "/proactive/final_tests" + "/025" +  "/robot_pose.npy")
proactive_robot_vel  = np.load(data_path + "/proactive/final_tests" + "/025" +  "/robot_velocity.npy")
proactive_stem_pose  = np.load(data_path + "/reactive" + "/032" + "/localisation.npy")

open_loop_stem_pose = drop_missed_frames(open_loop_stem_pose)
reactive_stem_pose = drop_missed_frames(reactive_stem_pose)
proactive_stem_pose = drop_missed_frames(proactive_stem_pose)

# normalise stem pose
open_loop_stem_pose = open_loop_stem_pose - 0.45
open_loop_stem_pose = np.where(open_loop_stem_pose < 0.0, open_loop_stem_pose, open_loop_stem_pose / 0.2)

reactive_stem_pose = reactive_stem_pose - 0.45
reactive_stem_pose = np.where(reactive_stem_pose < 0.0, reactive_stem_pose, reactive_stem_pose / 0.2)
reactive_stem_pose = np.where(reactive_stem_pose > 0.0, reactive_stem_pose, reactive_stem_pose / 0.4)

proactive_stem_pose = proactive_stem_pose - 0.45
proactive_stem_pose = np.where(proactive_stem_pose < 0.0, proactive_stem_pose, proactive_stem_pose / 0.2)
proactive_stem_pose = np.where(proactive_stem_pose > 0.0, proactive_stem_pose, proactive_stem_pose / 0.4)

# proactive_stem_pose = proactive_stem_pose - 0.46
# proactive_stem_pose = np.where(proactive_stem_pose < 0, proactive_stem_pose, proactive_stem_pose / 0.03)
# proactive_stem_pose = np.where(proactive_stem_pose > 0, proactive_stem_pose, proactive_stem_pose / 0.09)

ax.plot(np.arange(len(reactive_robot_vel[:contact_initialised+1]))/60, np.zeros(len(reactive_robot_vel[:contact_initialised+1])), c='b', linewidth=2, alpha=0.05)
ax.plot(contact_initialised/60 + np.arange(len(open_loop_stem_pose[contact_initialised:]))/60, open_loop_stem_pose[contact_initialised:]-0.2, c='g', linewidth=2)
ax.plot(contact_initialised/60 + np.arange(len(reactive_robot_vel[contact_initialised:]))/60, reactive_stem_pose[contact_initialised:]-0.2, c='b', linewidth=2)
ax.plot(contact_initialised/60 + np.arange(len(proactive_robot_vel[contact_initialised:]))/60, proactive_stem_pose[contact_initialised:]-0.26, c='r', linewidth=2)

# plt.plot(np.arange(len(reactive_robot_vel))/60, smooth(reactive_robot_vel[:-1, -1]*57), c='blue', linewidth=2)
# plt.plot(np.arange(len(proactive_robot_vel))/60, smooth(proactive_robot_vel[:-1, -1]*57), c='red', linewidth=2)
######################################################################
######################################################################
########################### SENSOR BASE ##############################
open_loop_stem_pose = np.load(data_path + "/proactive/final_tests" + "/026" +  "/localisation_offline.npy")

reactive_robot_pose = np.load(data_path + "/reactive/final_tests" + "/027" +  "/robot_pose.npy")
reactive_robot_vel  = np.load(data_path + "/reactive/final_tests" + "/027" +  "/robot_velocity.npy")
reactive_stem_pose  = np.load(data_path + "/reactive/final_tests" + "/027" +  "/localisation.npy")

proactive_robot_pose = np.load(data_path + "/proactive/final_tests" + "/030" +  "/robot_pose.npy")
proactive_robot_vel  = np.load(data_path + "/proactive/final_tests" + "/030" +  "/robot_velocity.npy")
proactive_stem_pose  = np.load(data_path + "/reactive" + "/033" + "/localisation.npy")

open_loop_stem_pose = drop_missed_frames(open_loop_stem_pose)
reactive_stem_pose = drop_missed_frames(reactive_stem_pose)
proactive_stem_pose = drop_missed_frames(proactive_stem_pose)

# normalise stem pose
open_loop_stem_pose = open_loop_stem_pose - 0.45
open_loop_stem_pose = np.where(open_loop_stem_pose < 0.0, open_loop_stem_pose, open_loop_stem_pose / 0.2)

reactive_stem_pose = reactive_stem_pose - 0.45
reactive_stem_pose = np.where(reactive_stem_pose < 0.0, reactive_stem_pose, reactive_stem_pose / 0.2)
reactive_stem_pose = np.where(reactive_stem_pose > 0.0, reactive_stem_pose, reactive_stem_pose / 0.4)

proactive_stem_pose = proactive_stem_pose - 0.45
proactive_stem_pose = np.where(proactive_stem_pose < 0.0, proactive_stem_pose, proactive_stem_pose / 0.2)
proactive_stem_pose = np.where(proactive_stem_pose > 0.0, proactive_stem_pose, proactive_stem_pose / 0.4)

ax.plot(np.arange(len(reactive_robot_vel[:contact_initialised+1]))/60, np.zeros(len(reactive_robot_vel[:contact_initialised+1])), c='b', linewidth=2, alpha=0.05)
ax.plot(contact_initialised/60 + np.arange(len(open_loop_stem_pose[contact_initialised:]))/60, open_loop_stem_pose[contact_initialised:]-1.35, "--", c='g', linewidth=2)
ax.plot(contact_initialised/60 + np.arange(len(reactive_robot_vel[contact_initialised+3:]))/60, reactive_stem_pose[contact_initialised+3:]-0.15, "--", c='b', linewidth=2)
ax.plot(contact_initialised/60 + np.arange(len(proactive_robot_vel[contact_initialised:])-2)/60, proactive_stem_pose[contact_initialised:, 0]-1.43, "--", c='r', linewidth=2)

###################################################################
###################################################################

ax.set_ylabel("Stem position (normalized)", fontsize=14)
ax.set_xlabel("Time (sec)", fontsize=14)
ax.axvspan(contact_initialised/60, contact_initialised/60 + 0.05, facecolor='b', alpha=0.15)
ax.text(contact_initialised/60 - 0.08, -0.4, "Contact initiated", rotation='vertical', fontsize=14)
ax.axhspan(-0.008, 0.008, xmin=0, xmax=1, facecolor='b', alpha=0.15)
ax.set_yticks(np.arange(-1, 1.1, 0.25))
ax.set_xticks([0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00])
ax.set_xticklabels([0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00], fontsize=12)
ax.set_yticklabels(np.arange(-1, 1.1, 0.25), fontsize=12)
ax.arrow(0.04, 0.03, 0.0, 0.1, width= 0.02, head_length=0.04, color='black')
ax.arrow(0.04, -0.04, 0.0, -0.1, width= 0.02, head_length=0.04, color='black')
ax.text(0.010, 0.23, "Sensor tip", rotation='vertical', fontsize=14)
ax.text(0.002, -0.85, "Sensor base", rotation='vertical', fontsize=14)
ax.set_ylim(-1, 1.1)
# ax.set_title("Stem pose", fontsize=20, pad=15)


cmap1 = plt.cm.hsv
cmap2 = plt.cm.RdYlGn
cmap3 = plt.cm.Greys
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color=cmap2(0.95), lw=4),
                Line2D([0], [0], color=cmap1(0.7), lw=4),
                Line2D([0], [0], color=cmap1(0.0), lw=4),
                Line2D([0], [0], color=cmap3(1.0), lw=2),
                Line2D([0], [0], color=cmap3(1.0), lw=2, linestyle='--')
                ]

ax.legend(custom_lines, ['Open-loop', 'PD-tactile', 'd-FPC', 'linear push', 'circular push'], fontsize=11, loc="lower right", bbox_to_anchor=(0.415,0.67), numpoints=2)
plt.savefig("/home/kiyanoush/Cpp_ws/src/haptic_finger_control/RT-Data/stem_pose_cluster.png")

# plt.figure()

# plt.plot(np.arange(len(reactive_robot_vel))/60, smooth(reactive_robot_vel[:-1, -1]*57), c='blue', linewidth=2, label="PD-tactile")
# plt.plot(np.arange(len(proactive_robot_vel))/60, smooth(proactive_robot_vel[:-1, -1]*57), c='red', linewidth=2, label="d-FPC")
# plt.ylabel("Angular velocity (deg/sec)", fontsize=14)
# plt.xlabel("Time (sec)", fontsize=14)
# plt.axvspan(contact_initialised/60, contact_initialised/60 + 0.05, facecolor='b', alpha=0.15)
# plt.text(contact_initialised/60 - 0.08, -7.5, "Contact initiated", rotation='vertical', fontsize=15)
# plt.legend(fontsize=14)
# plt.xticks([0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00], fontsize=12)
# plt.yticks([-8, -6, -4, -2, 0, 2], fontsize=12)
# # plt.title("Control action", fontsize=20, pad=15)

# plt.show()

# plt.savefig("/home/kiyanoush/Cpp_ws/src/haptic_finger_control/RT-Data/action.png")
