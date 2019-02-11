# server setup

## server (LAMP install)
[how to install LAMP](https://howtoubuntu.org/how-to-install-lamp-on-ubuntu)

```
sudo apt install apache2

sudo apt install mysql-server

sudo apt install php-pear php-fpm php-dev php-zip php-curl php-xmlrpc php-gd php-mysql php-mbstring php-xml libapache2-mod-php

sudo service apache2 restart

open browser, http://localhost/

check php, php -r 'echo "\n\nYour PHP installation is working fine.\n\n\n";'
```

## CHANGE APACHE2 ROOT PATH
```
1. get su auth
2. /etc/apache2
3. change /etc/apache2/sites-availale/000-default.conf 
   DocumentRoot /path/to/my/proj
4. change /etc/apche2/apache2.conf
   <Directory /path/to/my/proj>
   ...
   </Directory>
5. sudo service apache2 restart
```

## scp pass passwd
use sshpass
```
sshpass -p "password" scp -r user@example.com:/some/remote/path /some/local/path
```

```
sudo apt-get install sshpass
```

# client side

## jekyll 

### jekyll theme
get below theme from github

* jekyll-theme-next
* minimal-mistake
* mmi
* Nice_Blog
