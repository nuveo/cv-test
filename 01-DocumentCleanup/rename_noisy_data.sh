# command to add zeros to the images in noisy_data in order to correctly sort the images in python
cd noisy_data && ls | awk '/^([0-9]+)\.png$/ { printf("%s %04d.png\n", $0, $1) }' | xargs -n2 mv
