# add offline dict for ubuntu
## reference
https://askubuntu.com/questions/191125/is-there-an-offline-command-line-dictionary

## step 1: install sdcv
```
sudo apt-get install sdcv
```

## step2: download dictionary files
[dictionary](https://web.archive.org/web/20140917131745/https:)
https://web.archive.org/web/20140917131745/http://abloz.com/huzheng/stardict-dic/dict.org/

## step 3: install downloaded dictionaries
```
sudo mkdir -p /usr/share/stardict/dic/
sudo tar -xvjf xx.bz2 -C /usr/share/stardict/dic
sudo tar -xvzf xx.gz -C /usr/share/stardict/dic
```

## step 4: done
```
sdcv word
```

## comments
perfered dics
* GNU-linux
* wordNet
* collins
* english tresaurus
* oxford dic

## Chinese dict
### git address
https://github.com/jiehua233/youdao-dict.git

###
catch contents from 'http://dict.youdao.com/search?q=word'

### installation and usage
```
pip install -r requirements.txt
chmod +x dict.py
sudo cp dict.py /usr/local/bin/dict

dict world
```

