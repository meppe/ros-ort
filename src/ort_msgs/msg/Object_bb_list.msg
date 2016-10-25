# Several bounding boxes for several objects in one frame
string frame_id
int64 frame_timestamp
bool is_keyframe

# BB parameters (ul_x,ul_y,lr_x, lr_y)
float32[] bb_ul_x
float32[] bb_ul_y
float32[] bb_lr_x
float32[] bb_lr_y
# detected class
string[] class_name
string[] object_id
# detection score
float32[] score
