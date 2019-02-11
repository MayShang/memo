#!/bin/bash
# useful bash base

# --------------------------------
# variables examples
var="hello world"
len=4
echo ${var:0:$len}
echo ${var/hello/A}

echo ${foo:-234}

# local variable
function test()
{
    local var="examples local"

    echo ${var}
}
test
# --------------------------------
# arrary examples
array=(one two three four five)
for i in ${array}; do
    echo ${i}
done

echo {2..5}
echo {8..4}
echo {z..a}

# --------------------------------
# command 
echo $PWD
echo $(pwd)
# clear
echo $(dirname $0) # echo current dir
xxxx >/dev/null 2>&1


# --------------------------------
# read from input

# echo "enter something"
# read something somewhere
# echo "hello ${something} ${somewhere}"

# if [ ${name} != ${somewhere} ]; then
#     echo "different"
# fi

# --------------------------------
# ls -l | ag "\.sh"

# --------------------------------
# loop examples 0
contents=$(cat monitor-network.sh)
# echo "\n$contents\n"
# for line in ${contents}; do
#     echo ${line}
# done

# loop examples 1
filelist=$(ls | grep "\.sh")
echo ${filelist}

for i in ${filelist}; do
    echo ${i}
done

# loop examples 2
for x in {2..4}; do
    echo ${x}
done

# loop example 3
for output in $(ls | grep "\.sh"); do
    # cat ${output}
    echo ${output}
done

# --------------------------------
# case statement
echo "choose:"
read choose
case ${choose} in
    (0) echo "this is 0";;
    (1) echo "this is 1";;
    (2) echo "this is 2";;
esac

# --------------------------------
# how to use `||` `&&` for condition
grep ^root /etc/passwd >/dev/null || echo "root wasn't found"
# cat /etc/passwd
which vi || echo "what will happen?"

grep ^root /etc/passwd >/dev/null && echo "root was found"

# --------------------------------
# show positional parameters

for arg; do
    echo ${arg}
done

while [ ${1+defined} ]; do
    echo $1
    shift
done

# --------------------------------
# why need always use "$@"
export IFS='-'
echo $*
echo $@
echo "$*"
echo "$@"
echo ${@:1:3}
args=${@:1:3}
echo $args

# --------------------------------
# getopts examplels

function display_help()
{
    echo "help"
}

while :
do
    case "$1" in
        -f | --file)
            file="$2" # you may want to check validity of $2
            shift 2
            ;;
        -h | --help)
            display_help
            exit 0
            ;;
        --) # end of all options
            shift
            break;
            ;;
        -*)
            echo "Error: unknown option: $1"
            exit 1
            ;;
        *) # no more options
            break
            ;;
    esac
done


