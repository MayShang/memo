# gcc compile

## compile a single file app
```
gcc -o test test.c
g++ -o test test.cc
```

### the -Wall option enables more warning

## compile sources files to obj and link the app
```
gcc -c test.c
gcc -c test1.c
gcc -o test test.o test1.o
```

## compile with debug version
```
gcc -o test tst.c -g
```

## gdb 

### gdb 
```
gdb test
gdb test -p PID ( gdb attached to the application while it is running )
```

### gdb usage
* break (b) to set a breakpoint in the code. can be used with a function name or a location in the 
source code, or an absolute memory address.

* print (p) to print the value of a variable. used with a variable name, even if it's a complex one

* c to continue the execution until the next breakpoint.
* next (n) next line of code
* step (s) step into 
* backtrace (bt) to display the function call stack

## make
```
CFLAGS = -Wall

all: test    
test: t1.o t2.o
   gcc -o $@ $^ $(CFLAGS)

clean:
   $(RM) -f test

install:
   $(CP) test /usr/bin
```

### make reference
[http://www.gnu.org/software/make/manual/](http://www.gnu.org/software/make/manual/)
