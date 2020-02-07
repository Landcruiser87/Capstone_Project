#!/usr/bin/env bash
printf '\033[2J'
file=$1
counter=$2
while [ $counter -gt 0 ]
do
	printf '\033[2J'
	tail -n 70 $file
	sleep 1s
	counter=$((counter-1))
done
