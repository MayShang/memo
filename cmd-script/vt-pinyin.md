# install kvm
```
sudo egrep '(vmx|svm)' /proc/cpuinfo
apt-get update
sudo apt-get install -y qemu-kvm qemu virt-manager virt-viewer libvirt-bin

kvm-ok
```

BIOS VT set
`F1` enter BIOS to set cup options

# install pinyin
```
sudo apt-get remove ibus-pinyin
sudo apt-get install ibus-libpinyin
```
then 
set `language support` and `Text Entry`
