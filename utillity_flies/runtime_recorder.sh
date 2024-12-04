#!/bin/bash

# This shell script executes all three binary files generated by the Makefile builder
# and records them in the `benchmark.txt` file in plaintext format - to later be used
# by the `plotter.py` python script to visualize the runtime performance metrics

echo THRUST >> benchmark.txt
echo >> benchmark.txt
for num_vals_a in {10000..1000000..10000}
do
../thrust $num_vals_a 10 0 >> benchmark.txt 2>&1
done
echo >> benchmark.txt
echo >> benchmark.txt

echo SINGLETHREAD >> benchmark.txt
echo >> benchmark.txt
for num_vals_b in {10000..500000..10000}
do
../singlethread $num_vals_b 10 0 >> benchmark.txt 2>&1
done
echo >> benchmark.txt
echo >> benchmark.txt

echo MULTITHREAD >> benchmark.txt
echo >> benchmark.txt
for num_vals_c in {10000..1000000..10000}
do
../multithread $num_vals_c 10 0 >> benchmark.txt 2>&1
done
echo >> benchmark.txt
echo >> benchmark.txt