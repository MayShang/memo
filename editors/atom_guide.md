# ATOM GUIDE

## extension Package
```
1. trailing-semicolon,  add semicolon and new line at the end of the line
2. atom-ternjs,
3. php-ide-serenate,
4. atom-autoclose, html tag auto close
5. php-debug, XDebug
6. api-docs,
7. goto definition
8. symbols-tree-view
```

## change keybinding

change file: keymap.cson

```
'atom-text-editor':
  'ctrl-alt-n': 'api-docs:search-under-cursor',
  'alt-;': 'trailing-semicolon:semicolon',
  'alt-,': 'trailing-semicolon:comma',
  'shift-alt-h': 'core:move-left',
  'alt-j': 'core:move-down',
  'alt-k': 'core:move-up',
  'alt-l': 'core:move-right',
```
