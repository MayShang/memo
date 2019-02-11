# bash shell

## scripting cheatsheet
[bash cheatsheet](https://devhints.io/bash)

## $#, $@ and $?

`$#`: number of arguments.   
`$@`: all arguments list
`$?`: the last one argument.

### difference between `$@` and `$*`
they list all arguments from `$1`.
difference happens when being quoted,
* `"$@"` => "$1c$2c$3c...${N}"
* `"$*"` => "$1" "$2" "$3" ... "${N}"
* you almost always want a quoted `"$@"`
```
export IFS='-'
echo $*   => 3 4 hello
echo $@   => 3 4 hello
echo "$*" => 3-4-hello
echo "$@" => 3 4 hello
```
    
example:
test.sh
```
echo '$#' $#
echo '$@' $@
echo '$?' $?
```
```
> ./test.sh 1 2 3

you will get output:
$# 3  
$@ 1 2 3
$? 0
```

## 2>&1 and 1>&2    
`2>&1`: redirect `stderr` to a file named `1`. `&` indicates that what follows is a file descriptor and 
        not a filename. so the construct becomes: `2>&1`.
`1>&2`: redirect `stdout` to `stderr`

## cut -d = -f 1
man cut, -d : delimter is `=`. -f: field, `1` indicates the first field.
```
echo "xx=ei" | cut -d = -f 1   ===> xx
```

## wait `$!`
wait for a process to end.

## basic [x in y mins](https://learnxinyminutes.com/docs/bash)
### single/double quote
single quote won't expand the variables.

### string substitution
```
echo ${var/substring/replacedstring}
echo ${var:0:$len}
```
### default value
```
echo ${foo:-"xeq"}
```
### declare array
```
array=(one two three four five)
echo ${array[0]} # => "one"
echo ${array[@]} # => "one two three four five"
echo ${#array[@]} # => "five"

for i in "${array[@]}"; do
    echo $i
done

```

### brace expansion {}
```
echo {1..10}
echo {z..a}
```

### buildin variables
```
echo "last program's return value: $?"
echo "script's PID: $$"
echo "number of arguments passed to script: $#"
$PWD $(pwd)
clear # clear screen
```

### reading values from input
```
read Name where why # note that we don't need to declare a new variable
echo "hello ${Name} ${where} ${why}"
```

### compare between vars
```
if [ $name != ${who} ]; then
    echo xxx
else
    echo xxx
fi
```

#### empty
```
if [ "" != ${name} ]
```

string comparision we can use `==` `!=`, but integer has to use `-eq`

```
if [ ${name} == "heil"] && [ ${age} -eq 15 ]; then
if [ ${name} == "xdj" ] || [${name} == "ieji"]; then
```

### string compare
```
-z <str>         # true if empty
-n <str>         # true if not empty
<str1> = <str2>  # true if equal
<str1> != <str2> # true if not equal
<str1> < <str2>  # true if sorts before
```

### arithmetic compare
```
<int1> -eq <int2  # true if equal
<int1> -nq <int2  # true if not equal
<int1> -le <int2  # true if less than or equal
<int1> -ge <int2  # true if great than or equal
<int1> -lt <int2  # true if less than 
<int1> -gt <int2  # true if great than 
```

### file test
```
-e <file> # true if exist
-f <file> # true if exist
-w
-r
-x 
-s <file> # file size bigger than 0 (not empty)

if [ ! -d file ]; then
```
### command
```
ls -l | grep "\.txt" # .txt file in the current directory
tail -n 10 file.txt # print last 10 lines of file.txt
head -n 10 file.txt 
sort file.txt
cut -d ',' -f 1 file.txt # print only the first column before the ',' character
sed -i 's/okay/great/g' file.txt # replace every occurrence of 'okey' with 'great' in file.txt
grep "^foo.*bar$" file.txt # print to stdout all lines of file.txt which match begin with "foo" and end in "bar"
grep -c "^foo.*bar$" file.txt # 
grep -n "^foo.*bar$" file.txt # give line number
grep -r "^foo.*bar$" someDir # recursively 'grep' 
grep -rI "^foo.*bar$" someDir # recursively but ignore binary files
grep "^foo.*bar$" file.txt | grep -v "baz" # but filter out the lines containing "baz"

trap "rm xx; exit" SIGHUP SIGINT SIGTERM

```

#### read file using `cat`
```
contents=$(cat monitor-network.sh)
echo "\n ${contents}\n"
```
basically get each line from files.

```
filelist=$(ls | grep "\.sh")
for i in ${filelist}; do
    echo ${i}
done
```

cmds can be substituted within other commands using $()
```
echo "there are $(ls | wc -l) items here."
```

### case
```
case ${var} in
    0) echo "this is 0";;
    1) echo "this is 1";;
    2) echo "this is 2";;
esac
```
note: suffix `;;` must be there

## how to use info
```
man info
info info # get help about info
```

## basic [bash wiki](wiki.bash-hackers.org/start)
### `||` and `&&`
```
which vi || echo "if success, echo never show; || means OR, if former fails, something else will show"
grep ^root /etc/passwd && echo "if success, echo will show. && means AND"
```

### shifting 
```
for arg; do
    echo ${arg}
done

while [ ${1+defined} ]; do
    echo $1
    shift
done

```

### range of positional params
`COUNT` can be omitted
```
${@:START:COUNT}
${*:START:COUNT}
"${@:START:COUNT}"
"${*:START:COUNT}"

"${@: -1}" # the last positional param
```

### getopts and cmd-line options process
[cmd line options process](mywiki.wooledge.org/BashFAQ/035)
```
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
```

