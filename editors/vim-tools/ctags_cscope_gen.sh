#!/bin/bash
rm ./tags
./filelist-gen.sh $1
ctags -L filelist
mv filelist cscope.files
cscope -b
CSCOPE_DB=`pwd`/cscope.out
export CSCOPE_DB


