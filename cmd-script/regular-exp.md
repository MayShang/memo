# regular expressions

how to say its meaning, normal and we always use this sorts of expressions and because they follow
the same regularity, so we call it the regular expressions.

## match expressions

* .      -- any character, just one character
* [  ]   -- any character listed inside the brackets
* [^ ]   -- any character not listed inside the brackets
* ^      -- matches the beginning of the line
* $      -- matches the end of the line
* \*     -- matches the previous element zero or more times, 
* +      -- matches the previous element one or more times
* ?      -- matches the previous element zero or one time
* \( \)  -- define a sub-expression that can be later recalled by using \n, 

## reference
[http://www.regular-expressions.info](http://www.regular-expressions.info)

# Todo
1. to understand how to extract substring from string   
```
declare v=$(expr $c : "^\s*CONFIG_\([A-Z0-9_]\+=[y|n\"].*\)")
```
