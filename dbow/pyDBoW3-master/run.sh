#!/bin/bash 
# Ensure that you have installed nvidia-docker and the latest nvidia graphics driver on host!

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")

# Build and run the image
echo "Building image..."
sudo nvidia-docker build --build-arg pass=$SPASSWORD -t dbow3 .
echo "Removing older image..."
sudo nvidia-docker rm -f dbow30
echo "Running image..."
sudo nvidia-docker run --privileged -d -v /dev:/dev --ipc=host -p 52022:22 --name dbow30 \
      -v $SCRIPTPATH/src:/dbow3/src \
      -v $SCRIPTPATH/docker_share:/dbow3/docker_share \
      orbslam

# Retrieve IP and port of Docker instance and container
CONTAINERIP=$(sudo nvidia-docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' orbslam0);
DOCKERIP=$(/sbin/ifconfig docker0 | grep 'inet addr:' | cut -d: -f2 | awk '{ print $1}')
echo "CONTAINER IP:":$CONTAINERIP
echo "DOCKER IP:":$DOCKERIP
DOCKERPORTSTRING=$(sudo nvidia-docker port dbow30 22)
DOCKERPORT=${DOCKERPORTSTRING##*:}
echo "DOCKER PUBLISHED PORT 22 -> :":$DOCKERPORT
echo "IdentityFile $SCRIPTPATH/.sshauth/dbow3.rsa" >> ~/.ssh/config
ssh-keygen -f ~/.ssh/known_hosts -R [$DOCKERIP]:$DOCKERPORT
echo "Login password is: ":$SPASSWORD
#ssh -o StrictHostKeyChecking=no root@$DOCKERIP -X -p $DOCKERPORT
# ssh  -X -p $DOCKERPORT -v -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@$DOCKERIP
echo "LOGIN LIKE: ssh -o StrictHostKeyChecking=no -X -p $DOCKERPORT root@$DOCKERIP" >> login.txt
ssh -o StrictHostKeyChecking=no -X -p $DOCKERPORT root@$DOCKERIP

