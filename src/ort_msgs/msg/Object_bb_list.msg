# Several bounding boxes for several objects in one frame
string frame_id
int32 frame_timestamp
bool is_keyframe

# BB parameters (x,y,width,height)
float32[] bb_x
float32[] bb_y
float32[] bb_width
float32[] bb_height
# detected class
string[] class_name
string[] object_id
# detection score
float32[] score
