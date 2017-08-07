#!/bin/bash
img_name="ros_gen_grasping_script"
docker rm $img_name
images_folder="/informatik2/wtm/datasets/KT Internal
Datasets/20170703_NICO_GraspTraining"
echo $images_folder
docker run \
-it \
--rm \
--name $img_name \
--link roscore_kinetic \
--net default \
-e ROS_MASTER_URI=http://roscore_kinetic:11311/ \
-v '$images_folder':/imgs \
meppe78/ros-core-kinetic \
bash -c '
for $img in /imgs
do
	echo "processing img file $img"
done
bash
'

#	rostopic pub frcnn/interface_input std_msgs/String "$input"