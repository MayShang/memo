# DESIGN IDEAS

## WEBSITE TREE
|~~ private
|     |
|     |~~ shared
|     |     |
|     |     |~~ header.php
|     |     |~~ footer.php
|     |
|     |~~ init.php
|     |~~ func.php
|
|~~ public
|     |
|     |~~ index.php
|     |~~ css
|     |~~ js
|     |~~ images
|     |~~ menu
|     |    |
|     |    |~~ index.php
|     |    |~~ submenu
|     |    |     |
|     |    |     |~~ index.php

##WEB TIPS

### html5 tags DESIGN
![tags design](../images/html5_tags_figure.png)

## How to think design
no matter on the client side or server side, you should think from the total. 
meaning， all files are requested from the server, so all html tags are generaged by server side 
script language, after files downloaded on client side, browser will parse all php elements to readable html tags, including 
 like submit button action function, about this ideas, you need to test. 

but before this, in order to get job done easily, you need to be familiar with bs, this will help you generate html file fast. 
but actually, they are separate.
you can work on them in paralle. 
meaning, you can write processing logic on the raw html tag, and at the same time, work on bs to prepare for better layout. 

when you are familiar with logic, BS should be ready. 

so you can use the best your time for these two areas. 

### how to generate
when get result from sql, php generate html data, and return original request. 
because original page is waiting, it's sync process.
may there is a funtion, pass query result to this funtion, and this funtion generate tags files, and then return. 

request and respond 
ajax is different from the general action requst, 
what is the difference from them?


### server side need to think about thread issue?
this is server hardware consideration. 

## what about JSON response?
this is the same logic as above. 
 
