#!/bin/bash

# iterate over all .jpg files in current directory and run OpenALPR on each

DIR="$1"
dend=`basename "$DIR"`  # last part of path is date YYYY-MM-DD

date=$(date)
pfile=$DIR/plates.txt

echo "#  $DIR processed on $date" >> $pfile

find $DIR -maxdepth 1 -type f -iname "*.jpg" -print0 | while IFS= read -r -d $'\0' f; do
  base=$(basename "$f")
  alpr $f | while IFS= read -r res ; do  # iterate over each line of output from 'alpr'
    if [[ $res == *"No license plates found"* ]]; then
      res="#"
    fi
    echo "$dend $base $res" >> $pfile
  done  # loop over alpr output lines

done  # loop over filenames
