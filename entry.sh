#!/bin/bash -ex
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

$DIR/bootstrap.sh $DIR $DIR/venv

# Default path
current_path="/data/data_recording"

# Check if the disk /dev/sda1 exists
if lsblk /dev/sda1 > /dev/null 2>&1; then
    # Check if the disk is already mounted
    if ! mountpoint -q /mnt/; then
        sudo mount /dev/sda1 /mnt
        if mountpoint -q /mnt/; then
            echo "Successfully mounted /dev/sda1 at /mnt"
            current_path="/mnt/data_recording"
        else
            echo "Failed to mount /dev/sda1 at /mnt"
            echo "Saving in Amiga"
        fi
    else
        echo "/dev/sda1 is already mounted at /mnt"
        current_path="/mnt/data_recording"
    fi
else
    echo "/dev/sda1 does not exist. Skipping mount and saving in default path."
fi


sleep 1

$DIR/venv/bin/python $DIR/src/main.py $@ --path "$current_path" --port1 50051 --port2 50052 --exposure_time 10000 --iso 100

exit 0
