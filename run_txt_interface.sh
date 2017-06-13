#!/bin/bash
docker rm ros_kinetic_enter_classes
docker run \
-it \
--rm \
--name ros_kinetic_enter_classes \
--link roscore_kinetic \
-e ROS_MASTER_URI=http://roscore_kinetic:11311/ \
meppe78/ros-core-kinetic \
bash -c '
while true
do
	echo "
	Enter classes to display, separated by \",\".
	Enter \"all\" to display all classes.
	Enter \"mask\" to toggle object masking.
	Enter \"file:<path>/<to>/<file>\" to specify a file for output.
	Enter \"exit\" to exit."
	read classes
	if [ "$classes" = exit ]; then
	    break
	fi
	echo "Will now only display the following classes: $classes"
	rostopic pub frcnn/classes std_msgs/String "$classes"
done
'
