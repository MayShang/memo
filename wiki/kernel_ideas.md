# Kernel 
[kernel docs](www.kernel.org/doc)

[kernel doc documentation](https://www.kernel.org/doc/Documentation/)

## build and install
* how to build a new version kernel
* how to install it and use it.
* write a driver for a board devices

## more beautiful stuff need to think about
really has no time and energy to waste on others.

## what is eMMC
the term eMMC is short for "embedded multi-media Controller"   
a package consisting of both flash memory and a flash controlller integrated on the same silicon die.
the eMMC solution consists of at least three components:  
* the MMC (multimedia card) interface
* the flash memory
* the flash memory controller

it's offered in an industry-standard BGA package.

inefficient to manage flash memory content from outside the flash memory die. 
hence, eMMC was developed as a standardized method for bundling the controller into the flash die. 

### benefits of eMMC
eliminates the need to develop interface software for all types of NAND memory by integrating
the embeded controller into the memory chip and providing and easy-to-use memory solutions package for 
high-speed data transmissions by devices, such as mobile phones.

### eMMC hardware
eMMC hardware is a flash memory, along with a controller to manage the wear leveling and 
necessary ECC calculations.
A driver is required to use this part, and a basic one is often available from the board manufacturer as part of BSP.
this driver provides a block device interface.

# how to install compiled kernel
```
make xconfig (make config, etc)
make bzImage
make modules
sudo make modules_install
sudu make install

```
comments: command `sudo make insall` will generage /boot/grub/menu.lst automatically.

# compile modules
kernel from which you build your kernel module and to which
you are inserting module should be of same version.

## simple if compile distor's modules.
just add `KERNELDIR=/usr/src/linux-xxx`

method to make modules:
refer to ldd3 any Makefile   

```
modules:
    $(MAKE) -C $(KERNELDIR) M=$(PWD) modules

modules_install:
    $(MAKE) -C $(KERNELDIR) M=$(PWD) modules_install
 
```

*actually* if you have any compiled linux kernel, you can 
compile modle under it. just below things:
```
export PATH=$TOOLCHAIN_PATH/bin:$PATH
export KERNELDIR=$COMILED_KERNEL
export ARCH=arm64
export CROSS_COMPILE=aarch64-linux-gnu-
make
```


