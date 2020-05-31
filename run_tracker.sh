#!/bin/bash

SPLIT=$1
DETS_DATAROOT=$2
POSE_DIR=$3
TRACKS_DUMP_DIR=$4
MIN_CONF=$5

python -u run_ab3dmot.py  --split $SPLIT \
	--dets_dataroot $DETS_DATAROOT \
	--pose_dir $POSE_DIR \
	--tracks_dump_dir $TRACKS_DUMP_DIR \
	--min_conf $MIN_CONF