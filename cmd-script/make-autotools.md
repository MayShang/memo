# what it is and what it for

## reference
search: autotools introduction and autotools tutorial

* make: [make GNU](www.gnu.org/software/make/manual/html_node/index.html)
* Autotools: [autotools](www.lrde.epita.fr/~adl/autotools.html)

### GNU build system
the autotools are tools that will create a GNU Build System for your package.  
Autoconf mostly focuses on `configure` and Automake on `Makefile`.

### auto create Makefile
steps:
1. download `amhello` example
2. `autoscan` generate `configure.scan`
3. change `configure.scan` to `configure.ac`
4. `alocal` generate `alocal.m4`
5. `autoconf` generate `confgure`
6. write `Makefile.am`
7. `automake --add-missin` generate `Makefile.in`
8. `.configure` generate `Makefile`
9. make and make install && make distcheck


### standard dir variables
* prefix      -->   /usr/local
    * exec_prefix   ${prefix}
    * bindir        ${prefix}/bin
    * libdir        ${prefix}/lib
    ...

    * includedir    ${prefix}/include
    * datarootdir   ${prefix}/share
        * datadir   ${datarootdir}
        * mandir    ${datarootdir}/man
        * infodir   ${datarootdir}/info
        * docdir    ${datarootdir}/doc/${PACKAGE}

example:
```
./configure --prefix=/xxx/xx
```
```
make
make install
```
this would install bin/hello and share/doc/README

### Standard Configuration variables
* CC        c compiler cmd
* CFLAGS    c compiler flags
* CXXFLAGS  c++
* LDFLAGS   linker flags
* CPPFLAGS  c/c++ preprocessor flags

below is how one would call `configure` to force it to use `gcc-3` as C compiler, use 
 header files from xxx when compiling, and libraries from xxx when linking.
add `-I` for `CPPFLAGS`, `-L` for `LDFLAGS`.

```
./configure --prefix=/usr/local CC=gcc \
                     CPPFLAGS=-I$HOME/usr/include LDFLAGS=-L$HOME/usr/lib
```

### autoconf summary
autoconf better for big project, for generic project, `make` should be enough.

## Make

