#!/bin/bash
docker rm ros_kinetic_txt_interface
docker run \
-it \
--rm \
--name ros_kinetic_txt_interface \
--link roscore_kinetic \
--net default \
-e ROS_MASTER_URI=http://roscore_kinetic:11311/ \
meppe78/ros-core-kinetic \
bash -c '
while true
do
	echo "
	Enter classes to display, separated by \",\".
	Enter \"all\" to display all classes.
	Enter \"mask\" to toggle object masking.
	Enter \"single\" to toggle display of only a single object with the highest score.
	Enter \"file:<path>/<to>/<file>\" to specify a file for output.
	Enter \"exit\" to exit."
	read input
	if [ "$input" = exit ]; then
	    break
	fi
	echo "You entered: $input"
	rostopic pub frcnn/interface_input std_msgs/String "$input"
done
bash
'
