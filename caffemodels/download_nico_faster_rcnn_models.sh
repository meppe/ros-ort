#!/bin/bash

#DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../" && pwd )"
#cd $DIR

mkdir nico_frcnn_models

FILE="nico_frcnn_models/ZF_faster_rcnn_nico.caffemodel"
URL="https://drive.google.com/uc?export=download&id=0B_Cy6UXhY3PIMHJfckpaQm41Snc"
#CHECKSUM=e2a75a32ed77a9d663e496387808b02f
##
#if [ -f $FILE ]; then
#  echo "File already exists. Checking md5 of $FILE..."
#  os=`uname -s`
#  if [ "$os" = "Linux" ]; then
#    checksum=`md5sum $FILE | awk '{ print $1 }'`
#  elif [ "$os" = "Darwin" ]; then
#    checksum=`cat $FILE | md5`
#  fi
#  if [ "$checksum" = "$CHECKSUM" ]; then
#    echo "Checksum is correct. No need to download."
#  else
#    echo "Checksum is incorrect. Need to download again."
#    echo "Downloading Faster R-CNN ZF model for NICO (226M)..."
#    wget $URL -O $FILE
#
#  fi
#else
#    echo "Downloading Faster R-CNN ZF model for NICO (226M)..."
#    wget $URL -O $FILE
#fi

echo "Downloading Faster R-CNN ZF model for NICO (226M)..."
wget $URL -O $FILE

#
#echo "Unzipping..."
#
#tar zxvf $FILE
#
echo "Done. Please run this command again to verify that checksum = $CHECKSUM."


#FILE="VGG16_faster_rcnn_nico.caffemodel"
#URL="https://drive.google.com/uc?export=download&id=0B_Cy6UXhY3PIVkRSRVhHLVlTUWM"
#CHECKSUM=04911bbb58aa2cce601760d021d5ed99
##
#if [ -f $FILE ]; then
#  echo "File already exists. Checking md5 of $FILE..."
#  os=`uname -s`
#  if [ "$os" = "Linux" ]; then
#    checksum=`md5sum $FILE | awk '{ print $1 }'`
#  elif [ "$os" = "Darwin" ]; then
#    checksum=`cat $FILE | md5`
#  fi
#  if [ "$checksum" = "$CHECKSUM" ]; then
#    echo "Checksum is correct. No need to download."
#    exit 0
#  else
#    echo "Checksum is incorrect. Need to download again."
#  fi
#fi

#echo "Downloading Faster R-CNN VGG16 model for NICO (226M)..."
#
#wget $URL -O nico_frcnn_models/$FILE
#
#echo "Unzipping..."
#
#tar zxvf $FILE
#
#echo "Done. Please run this command again to verify that checksum = $CHECKSUM."
