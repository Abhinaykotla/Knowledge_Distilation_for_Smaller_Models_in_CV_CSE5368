#!/bin/bash

mkdir -p places_standard_dataset/val_hires/
mkdir -p places_standard_dataset/visual_test_hires/

# Randomly sample images for test and visual test
OUT=$(python3 fetch_data/sampler.py)
echo "${OUT}"

# Copy for validation
while IFS= read -r path; do
    if [ -d "${path}" ]; then
        cp -r "${path}" places_standard_dataset/val_hires/
    else
        cp "${path}" places_standard_dataset/val_hires/
    fi
done < places_standard_dataset/original/test_random_files.txt

# Copy for visual test
while IFS= read -r path; do
    if [ -d "${path}" ]; then
        cp -r "${path}" places_standard_dataset/visual_test_hires/
    else
        cp "${path}" places_standard_dataset/visual_test_hires/
    fi
done < places_standard_dataset/original/val_random_files.txt
