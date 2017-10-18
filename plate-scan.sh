#!/bin/bash

indir=$PWD    # current working directory

FILES=$indir/*_083.jpg

for f in $FILES
do
  fb=`basename "$f"`
  echo -n "$fb  " >> $indir/plates.txt
  # take action on each file. $f store current file name
  alpr -n 1 $f | grep 'confidence' | xargs echo -n >> $indir/plates.txt
  echo >> $indir/plates.txt
done
