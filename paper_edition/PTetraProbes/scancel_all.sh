#!/bin/bash

for j in `seq 4458116 4458151` ; do
  scancel $j
  echo  $j
done
