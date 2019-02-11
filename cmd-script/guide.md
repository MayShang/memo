## git guide

### pushing to a remote 
```
git push origin master
```

### adding an existing project
```
git init  
git add .   
git commit -m "first commit"   
git remote add origin repoURL   
git remote -v   
git push origin master  
```

### config git from scratch
```
git config --global user.name "x"
git config --global user.email "x@x.com"
git config --global core.editor vim

ssh-keygen -t rsa -C "x@x.com"
```

## cmd guide

* delete history
`history -c` then reload session to see effects.

## Packages
* zsh
* oh-my-zsh
* fasd
* vim
* tmux
* awesome
* vim-workshop
* vim-tmux-navigator
* silver searcher ag
    `sudo apt-get install silversearcher-ag`

## zshrc settings
```
eval "$(fasd --init posix-alias zsh-hook)"

alias -g gp='| grep -i'
alias -s md=vim
alias -s c=vim
alias v='f -e vim'
alias o='a -e xdg-open'
```

## man `ag`
`ag foo`: find files containing "foo", and print lines matches in context.  
`ag -l foo`: like above, but only list filename.  
`ag -i -o foo`: case-insensitively.  
`ag -c foo`: print number of matches in each files. this is the number of matches, not the number 
             of matching lines.
`ag -c foo | wc -l`: get the number of matching lines.  
`ag -g foo`: find file, its name is `foo`.   
`ag -g --hidden foo`: find file pattern `foo`, including hidden files.  

## tmux
```
# disable confirmation prompt
bind x kill-pane
bind & kill-window
```
