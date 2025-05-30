#!/bin/bash
source ../../../env.sh
echo "Using SPARK_MASTER=$SPARK_MASTER"
/usr/local/spark/bin/spark-submit --master=spark://$SPARK_MASTER:7077 ./heart-disease-prediction.py heart.csv framingham.csv
