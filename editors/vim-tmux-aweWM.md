# customize your own beautiful IDE env
1. vim  
2. tmux
3. awesome window manager
4. zsh

## awesome wm   
### reference  
awesome ArchWiki  

### changes
* cp `rc.lua` to be able to customize

```
mkdir -p ~/.config/awesome/  
cp /etc/xdg/awesome/rc.lua ~/.config/awesome/
```
* change themes
* change `mod4` to `Mod1`: map key to left `alt`.

## tmux
### reference
github vim-workshop  
this is guy who is vim fans

### tmux session and window
* A session is a single collection of pseudo terminal under the management of tmux.
  each session has one or more windows linked to it.
* A window occupies the entire screen and may be split into panes.

#### add vim-tmux-navigator plugin
add this plugin to navigate seamlessly between vim and tmux
1. download plugin under .vim/pluged
2. add `Plug vim-tmux-navigator` to `.vimrc` and install plugin to vim
3. read `vim-tmux-navigator` README file to add key map to `.vimrc` and `.tmux.conf`
4. **always remeber** read help file carefully, which help you too much

### changes
* `.tmux.conf` is tmux config file. 
* chagne terminal color to `xterm-256color` to be compatible with vim color scheme
* change prefix key, original is `C-b`, change to `C-n`, which is easily to operate.

## vim 
refer to this folder other files for vim. 

### ctags and cscope
### color scheme

## zsh



