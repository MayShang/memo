# PHP_IDEAS

## php manual
[manual](http://php.net/manual/en/preface.php)
[string function](http://php.net/manual/en/function.htmlspecialchars.php)

## input type
[Aug.8](#0803)
```
input type="checkbox"  
input type="file"       , upload file to the server
input type="hidden"
input type="email"
input type="password"
input type="radio"
input type="reset"
input type="submit"
input type="text"
option, an option in a SELECT element
select,
textarea, need closing tag, but above all don't need closing tag
```

```
<form action="xx.php" method="get/post">
</form>
```

## php and mysql and webpage ideas
you can begin from a sql table, after you have a static table, you just use it to render data into web browser.

the 2nd step, develop a user interface for someone to manage these data. meaning, some guys can edit, update or delete some data, but they don't need to operate sql language directly.
you need to provide a user interface for admin guys in order that they can enter something.

so first show, second, implement edit/update/create user interface.

### table assignment
* category: id, category_name, position,  
* book: id, category_id, sub_category_id, sub_category_name, owner, book_name, already_read(boolean), introduction, cover, memo,   
* book_attach: id, book_id, contents  

### blueprint
* private
  * shared
    * public_header
    * public_navigator
    * public_footer
    * public_static_homepage
  * funcs
    * init
    * functions
    * database  

* public
  * index
  * imgs
    * xx.pug
    * xx.jpg  
  * css  
    * xx.css  



#### regarding public/index.php
by default, it shows public_static_homepage, on header location, include public_header, and so do footer, navigation.

#### regarding navigation
query category and sub_category_id, add to navigation sidebar list,
if click one li, main area show that category books, 5 books one page, then we have a page choice.

from above this is a project. cost time and energe.

## continous learning
currently I know the basic operation and know the basic logic, but need a practice to get familiar with them.
actually I always have no enough time to do this, I have ideas, but no time because I need to cover job and kid, need to make sure his study fine.

I can get up early each morning, but my health condition is not so good, this summary I need to do exercise to store more energy for myself.

regarding this practice, you can start now.
just like writing, you don't know where to start, but just following yesterday's idea. first, create databases and tables, then develop static pages, then change static page to dynamic pages. then you will know where you can put effort.

## time trace
July and August

| Mon   | Tue   | Wed    | Thu   |  Fri  | Sat   | Sun   |  
| ----- | :---: | :----: | :---: | :---: | :---: | :---: |
|16 [x]   | 17 [x]   | 18 [x]    |  19 [x] | 20 [x]   | 21 [x]   | 22 [x]  |
|23 [x]   | 24 [x]   | 25 [x]    |  26 [x] | 27 [x]   | 28 [x]   | 29 [x]  |
|30 [x]   | 31 [x]   | 1 [x]      |   2 [x] [Aug.2](#0802) | 3 [more](#0803)     | 4     | 5  [more](#0805)     |


### time trace in detail
* Aug.1 :
* <a id="0802">Aug.2 </a> Utilities. and all.
* <a id="0803">Aug.3 </a>
* <a id="0804">Aug.4 </a> new start maybe?
* <a id="0805">Aug.5 </a> maybe more
