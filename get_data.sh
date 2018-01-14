#!/bin/bash
# Downloads and unpacks the LFW (Labeled Faces in the Wild) dataset

DATA_URL='http://vis-www.cs.umass.edu/lfw/lfw.tgz'

wget $DATA_URL
tar -xvzf lfw.tgz && rm lfw.tgz
mkdir data && mv lfw data/ 
