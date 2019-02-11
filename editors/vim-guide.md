# vim usage

# add ctags and cscope 
## add below line to .vimrc
[cscope_map.vim](cscope_map.vim)

```
set tags=tags;
set autochdir

source ~/.vim/autoload/cscope_maps.vim

function! LoadCscope()
  let db = findfile("cscope.out", ".;")
  if (!empty(db))
    let path = strpart(db, 0, match(db, "/cscope.out$"))
    set nocscopeverbose " suppress 'duplicate connection' error
    exe "cs add " . db . " " . path
    set cscopeverbose
  " else add the database pointed to by environment variable
  elseif $CSCOPE_DB != ""
    cs add $CSCOPE_DB
  endif
endfunction
au BufEnter /* call LoadCscope()


```

## how to generate ctags:tags
go to some src directory and execute cmd below
```
ctags -R
```

```
ctags -L filelist
```
## how to generate cscope.file and cscope_db: cscope.out
```
find /proj absolute path/ -name '*.c' -o -name '*.h' > cscope.files
cscope -b
# CSCOPE_DB=cscope.out; export CSCOPE_DB
```

## better to make a shell execute file to auto generate tags and cscope db
[cscope-sh](cscope-sh.sh)

## generate ctags and cscope auto
```
./ctags-cscope-gen.sh vssdk02[VSSDK]
```
[ctags-cscope generator](ctags-cscope-gen.sh)
[filelist generator](filelist-gen.sh)
http://vim.wikia.com/wiki/Creating_your_own_syntax_files


## how to use registers
```
:reg
```
get register
```
"np[n means the nth register, p means paste]

shift +'0p [paste the 0th reg contents]
```

## the difference between buffers, windows and tabs
Summary:

A buffer is the in-memory text of a file.[buffer 是文件在内存中缓存数据]

A window is a viewport on a buffer.[buffer 的观看口]

A tab page is a collection of windows. [tab 是窗口的集合]

in summary, files has memory inside system, these memory is buffers, but how to look or take action on this buffers, 
we use windows. we can open many files at the same time, so this windows casade to tab pages.

如果把 Vim 想象成一个机房的话，Buffer 就是主机，Window 是显示器，而 Tab 是一个个显示器架子。
只不过这个机房里面的显示器可以随意连接到别的主机上面，一个主机可以被多个显示器连接。

## how to switch between tabs
```
ctrl+pagedown
:tabnew file [open file in new tab]
```

## vim youcompleteme plugin proj include path
cp .ycm_extra_conf.py
add path to this file.

## fold function @ vim
```
:set foldmethod=indent
zM, zm, zr, zR
```

# add color scheme

1. cp scheme to ~/.vim/colors/  
2. change `colorscheme` to perfered scheme
```
colorscheme onedark
```

# vim color scheme link
[vim color scheme](github.com/vim-colorschemes)

## how to 'soft' wrap line
```
:set wrap linebreak nolist
```
