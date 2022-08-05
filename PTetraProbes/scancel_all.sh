#!/bin/bash

for j in `seq 4458071 4458115` ; do
  scancel $j
  echo  $j
done
