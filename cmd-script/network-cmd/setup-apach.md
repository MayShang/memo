## apache server setup

### server setup
```
sudo apt-get update
sudo apt-get install apache2

sudo systemctl status apache2
```

### server standard location
```
/var/www/html
```
start apache `systemctl start httpd`

### config
```
/etc/httpd/conf/httpd.conf
```
change `DocumentRoot`

### ffplay http playback
1. put file in `/var/www/html`, including media files
2. console : `ffplay http://xx.xx.xx.xx/v.mpg`


## apache2 && python && mysql
```
sudo apt-get install apache2
sudo mkdir /var/www/test
sudo a2dismod mpm_event
sudo a2enmod mpm_prefork cgi
sudo vim /etc/apache2/sites-enabled/000-default.conf

```
under `<VirtualHost *.80>`, add
```
<VirtualHost *:80>
        <Directory /var/www/test>
                Options +ExecCGI
                DirectoryIndex index.py
        </Directory>
        AddHandler cgi-script .py

        …

        DocumentRoot /var/www/test

        …
```

then  
```
sudo service apache2 restart
```

refer to https://www.digitalocean.com/community/tutorials/how-to-set-up-an-apache-mysql-and-python-lamp-server-without-frameworks-on-ubuntu-14-04

## Django
### install
```
pip3 install Django
```
check if installed
```
python3 -c "import django; print(django.__path__)"
```

start a new proj
```
django-admin startproject mysite
```
then cp to `/var/www/test`

start a new app
```
python3 manage.py startapp appname
```

### make model changes
1. Change your models (in models.py).
2. Run `python manage.py makemigrations` to create migrations for those changes
3. Run `python manage.py migrate` to apply those changes to the database.

### playing with db api
```
python manage.py shell
```

### deploy django project app
uwsgi:  https://docs.djangoproject.com/en/2.2/howto/deployment/wsgi/uwsgi/

### reference
https://docs.djangoproject.com/en/2.2/intro/whatsnext/

## google/ExoPlayer
playback streaming

## deploy flask
```
uwsgi --http xx.xx.xx.xx:5000 --module xtest:app
```
