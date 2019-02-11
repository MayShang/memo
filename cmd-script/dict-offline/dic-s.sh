#!/bin/bash

# usage: ./dic-s.sh filename

if [ -z "$1" ]; then
    echo "input filename"
    exit
else
    [ -f $1.expl ]
    rm $1.expl
fi

tmpfile=.tmp
explfile=.expl
if [ -e ${explfile} ]; then
    rm ${explfile}
fi

if [ -e ${tmpfile} ]; then
    rm ${tmpfile}
fi

grep '\-*>[a-z0-9]*' -o $1 | grep -i '[a-z0-9]*' -o > ${tmpfile}

echo '----------------- vocabulary -------------------------' >> ${explfile}
echo ' ' >> ${explfile}

cat ${tmpfile} >> ${explfile}
echo ' ' >> ${explfile}

cat ${tmpfile} | while read line; do
    echo --------- ${line} --------------- >> ${explfile}
    sdcv ${line} >> ${explfile}
    echo ' ' >> ${explfile}
done

cat $1 ${explfile} > $1.expl

[ -f ${tmpfile} ]
rm ${tmpfile}

[ -f ${explfile} ]
rm ${explfile}
