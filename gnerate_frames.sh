#!/usr/bin/env bash#!/usr/bin/env bash
data_root="/storage/data/nico2017/generate_frames"
mkdir resized_videos
mkdir video_frames

cd $1
for filename in *; do
    echo processing "$filename"
    ffmpeg -i "$filename" -vf scale=640:480,setdar=4:3 -c:v libx264 -preset fast -c:a aac "$filename"_resized.mp4 -hide_banner
    ffmpeg -i "$filename"_resized.mp4 -filter:v "crop=280:200:140:200" -c:v libx264 -preset fast -an "$filename"_resized_cropped.mp4 -hide_banner
    mv "$filename"_resized_cropped.mp4 "$data_root"/resized_videos/"$filename"_small.mp4
    rm "$filename"_resized.mp4
    mkdir "$data_root"/video_frames/"$filename"
    echo "ffmpeg -i $data_root/resized_videos/"$filename"_small.mp4 -r 1/1 $data_root/video_frames/"$filename"/frame_%03d.bmp"
    ffmpeg -i "$data_root"/resized_videos/"$filename"_small.mp4 -r 1/1 "$data_root"/video_frames/"$filename"/frame_%03d.jpg
done
cd ..


