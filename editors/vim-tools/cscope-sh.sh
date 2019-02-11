#!/bin/bash
dir=`pwd`
#echo $dir
find $dir/ -name '*.c' -o -name '*.h'  > $dir/cscope.files
cscope -b
CSCOPE_DB=$dir/cscope.out
export CSCOPE_DB
