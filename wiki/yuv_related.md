# yuv 

## sample rate
* 444  
  +---+  
  |YUV|     
  |   |  
  +---+  

* 422
  +---+    +---+
  | Y | UV | Y |
  |   |    |   |
  +---+    +---+
two pixel share one UV

* 420 
  +---+    +---+    
  | Y |    | Y |
  |   |    |   |
  +---+    +---+
        UV
  +---+    +---+
  | Y |    | Y |
  |   |    |   |
  +---+    +---+
four pixel share one UV

## layout
* packed, YUV is packed together.
* planar, yuv component save seperately.
* tile,   block group, group as block, like 6*6 pixel

## bit depth
* 8 bit
* 10 bit
    * compact way, five bytes represent 4 Y pixels, bit endian.
    * lose way,    16bytes represent one 10 bit data, 6 bits zero redutant.
* 12 bit
* 16 bit

## range
* Y is unsigned, 
* UV is signed
  so when UV is 0x80, color is grey. 

## tile
directly layout pixel in memory.better for graphics application.

## yuv file  
http://media.xiph.org/video/derf/

