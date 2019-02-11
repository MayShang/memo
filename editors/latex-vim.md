# install latex first

# add vim-latex in .vim folder

# install vim-latex-preview plugin
## configure
```
let g:tex_flavor='latex'
let g:livepreview_previewer = 'evince'
let g:livepreview_engine = 'pdflatex'
```

## preview inside vim
```
:LLPStartPreview
```

## compile pdf file
```
:!pdftex %
```
