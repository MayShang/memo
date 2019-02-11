# application and module
* application in user space, while module in kernel space
* app link to other library, while module only link to kernel
 
# kernel space module mentions
* small stack, may be as small as a single 4096 bytes page. need to allocate dynamically at call time.
* functions marked double underscore (*__* ) are low level component of the interface, should be used with caution.
* kernel can't do floating point arithmetic.

# kernel build system
* files found in Documentation/kbuild are required reading for anybody wanting to undersand all 
  that is really going on beneath the surface.
* `obj-m := hello.o` working for module build, actually, kernel build system handles the rest.
```
obj-m := module.o
module-objs := file1.o file2.o
```
this module build is invoked within the context of the larger kernel build system.
```
make -C ~/kernel-xxx M='pwd' modules
```
this command starts by changing its directory to the one provided with the `-C` option, which means, your kernel source directory.  
there it finds the kernel's top-level makefile. 
the `-M` option causes that makefile to move back into your module source directory before trying to build the modules targets  
this target, in turn, refer to list of modules found in the `obj-m` variable, which we've set to module.o in our examples.

# modprobe
like insmod, but will do more things than `insmod`: 
it will look at if module reference any symbols that are not currently defined in the kernel. if no, it will probe other modules for
relevant symbols. when found, it will loads them into kernel as well. but `insmod` fails with an "unresolved symbols" message left.

# where is modules
`/proc/modules` and `/sys/modules`

# whatIs

## what is `ELF`
the Executable and linkable Format, also known as `ELF`, is the generic file format for executables in Linux system. 
ELF files are composed of three major components:
* ELF header
* Sections
* segments
they play a different role in the linking /loading process of ELF executables.
sections: for linking a target object, needed on linktime but not on runtime.
Segments: program header, prepare the executable to be loaded into memory. it is not needed on linktime but on runtime.

## what is `GPL`
General Public License

## IDE
Integrated Device Electronics

## DMA
Direct Memory Access

# module code
```
#include <linux/module.h>
#include <linux/init.h>
```
`module.h` contains a great many definitions of symbols and functions needed by loadable modules.
`init.h` to specify your initialization and cleanup functions.

most modules also include `moduleparam.h` to enable the passing of parameters to the module at load time.

## quick points about race condition
the kernel will make calls into your module while your initialization function is still running.
so your code must be prepared to be called as soon as it completes its first registration.
Don't register any facility until all of your internal initialization needed to support that facility has been completed.
first complete your init then register facility.

## module parameters types
* bool, invbool: boolean value. invbool inverts the value, so that true values become false and vice versa.
* charp: a char pointer value, memory is allocated for user-provided strings, and the pointer is set accordingly.
* int, long, short, uint, ulong, ushort: basic integer values of various lengths. 

## module params permission
* definitions found in `<linux/stat.h>`
* 0, means no sysfs entry at all.
* `S_IRUGO`, only can be read
* `S_IRUGO|S_IWUSR`, allow root to change the params.

## why write driver in user-space
when you ar beginning to deal with new and unusual hardware, user space code help you learn to 
manage your hardware without the risk of hanging the whole system.
once you've done that, encapsulating the software in a kernel module should be a painless operation.

# device driver major/minor number 
* you can simple pick a number that appears to be unused, or you can allocate major numberin a 
  dynamic manner.
* but for new drivers, we strongly suggest that you use dynamic allocation to obtain your major device number,
  rather than choosing a number randomly from the ones that ar currently free.
  using `alloc_chrdev_region` rather than `register_chrdev_region`
* dynamic allocation major number can be read after `insmode`, and read '*/proc/devices*', but once
  the number has been assigned, you can read it from /proc/devices.
* most driver fundamental operations involve three important kernel data structures: 
    * file_operation
    * file
    * inode

* regarding `cdev`, the struct `cdev` are interfaces our device to the kernel.

## kernel debugging
* cause all kernel messages to appear at the console by simply entering:    
```
echo 8 > /proc/sys/kernel/printk
echo 4 4 1 6 > /proc/sys/kernel/printk
```

* level
    * 0, KERN_EMERG, emergency messages, usually those that precede a crash
    * 1, KERN_ALERT, a situation requiring immediate action
    * 2, KERN_CRIT, critical conditions, often related to serious hardware or software failures
    * 3, KERN_ERR, device driver often use it to report hardware difficulties
    * 4, KERN_WARNING, 
    * 5, KERN_NOTICE,
    * 6, KERN_INFO,
    * 7, KERN_DEBUG,

* using `printk_ratelimit` to avoid repeated message
```
if (printk_ratelimit()) {
    printk(KERN_NOTICE "the print still on fire!\n");
}
```

* using `proc` file to debug
the proc file is special, software created filesystem that is used by the kernel to export information to the world.
each file under /proc is tied to a kernel function that generates the file's contents on the fly when the file 
is read.
for example, /proc/modules, always returns a list of the currently loaded modules.
/proc file is heavily used in linux system. many utilites such as ps, top, and uptime, get their information from /proc.
some device drivers also export information via /proc.
the proc filesystem is dynamic, so your module can add or remove entires at any time.

## linux memory management
### how to understand memory
* kernel memory
* device memory
* user space memory
    * how user space program access device memory?
      device use hardware phical memory, but user program works under vritual memory system, how this user program access hardware memory.
      user program can have a virtual space larger than system phical memory. virtual memory system allows tricks to map program's memory to 
      device memory.


      
NEXT begin P67
