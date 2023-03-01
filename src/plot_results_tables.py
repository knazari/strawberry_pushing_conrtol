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


contact_initialised = 53

names = []
for n in range(1, 41):
    if n < 10:
        names.append("00" + str(n))
    else:
        names.append("0" + str(n))

stem_max_displacement = []
slip_instances = []
stem_displacement_integral = []
action_integral = []

for i in range(26, 30):

    stem_pose  = np.load(data_path + "/proactive/final_tests/" + names[i] +  "/localisation.npy")

    stem_pose = drop_missed_frames(stem_pose)

    stem_pose = stem_pose - 0.45
    stem_pose = np.where(stem_pose < 0.0, stem_pose, stem_pose / 0.2)
    stem_pose = np.where(stem_pose > 0.0, stem_pose, stem_pose / 0.4)
    stem_pose = stem_pose[contact_initialised:]

    max_displacement = abs(max(stem_pose) - min(stem_pose))
    stem_max_displacement.append(max_displacement)

    stem_vel = np.diff(stem_pose[:, 0])
    slip_ts = np.where(abs(stem_vel) > 0.02)[0]
    slip_instances.append(len(slip_ts))

    sum_displacement = 0
    sum_displacement += sum([stem*0.0166 for stem in abs(stem_pose)])
    stem_displacement_integral.append(sum_displacement)

    robot_vel  = np.load(data_path + "/proactive/final_tests/" + names[i] +  "/robot_velocity.npy")[:, -1] * 57
    sum_action = 0
    sum_action += sum([act*0.0166 for act in abs(robot_vel)])
    action_integral.append(sum_action)


# action_integral = np.array(action_integral)
# print("displacement integral mean: ", np.mean(action_integral))
# print("displacement integral std: ", np.std(action_integral))


stem_displacement_integral = np.array(stem_displacement_integral)
print("displacement integral mean: ", np.mean(stem_displacement_integral))
print("displacement integral std: ", np.std(stem_displacement_integral))


stem_max_displacement = np.array(stem_max_displacement)
slip_instances = np.array(slip_instances)
print("stem max displacement mean: ", np.mean(stem_max_displacement))
print("stem max displacement std: ", np.std(stem_max_displacement))

print("stem slip instances mean: ", np.mean(slip_instances))
print("stem slip instances std: ", np.std(slip_instances))
