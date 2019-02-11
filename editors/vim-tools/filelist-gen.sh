#!/bin/bash

echo `pwd`
echo $1


gen_proj02() { 
    find `pwd`/tee/chip -name '*.c' -o -name '*.h' > filelist
    find `pwd`/tee/include  -name '*.c' -o -name '*.h' >> filelist
    find `pwd`/tee/tee -name '*.c' -o -name '*.h' >> filelist
    find `pwd`/tee/tz  -name '*.c' -o -name '*.h' >>filelist
    find `pwd`/tee_dev -name '*.c' -o -name '*.h' >> filelist
    find `pwd`/XXXsdk/osal -name '*.c' -o -name '*.h' >> filelist
    find `pwd`/XXXsdk/ta_XXX -name '*.c' -o -name '*.h'  >> filelist
    find `pwd`/XXXsdk/XXX/gtest/testcases  -name '*.c' -o -name '*.h' -name '*.cpp' >> filelist
    find `pwd`/XXXsdk/XXX/inc  -name '*.c' -o -name '*.h' >> filelist
    find `pwd`/XXXsdk/XXX/src  -name '*.c' -o -name '*.h' >> filelist
    find `pwd`/XXXsdk/XXX/test/client_test  -name '*.c' -o -name '*.h' >> filelist
    find `pwd`/XXXsdk/XXX/tools/XXXdiag  -name '*.c' -o -name '*.h' >> filelist
    find `pwd`/XXXsdk/XXX/tools/XXXdmxutil  -name '*.c' -o -name '*.h' >> filelist
    find `pwd`/XXXsdk/XXX/tools/mdb  -name '*.c' -o -name '*.h' >> filelist
    find `pwd`/XXXsdk/XXX/tools/set_plane  -name '*.c' -o -name '*.h' >> filelist
}

gen_proj() { 
    find `pwd`/tee/tee/chip -name '*.c' -o -name '*.h' > filelist
    find `pwd`/tee/tee/include  -name '*.c' -o -name '*.h' >> filelist
    find `pwd`/tee/tee/tee -name '*.c' -o -name '*.h' >> filelist
    find `pwd`/tee/tee/tz  -name '*.c' -o -name '*.h' >>filelist
    find `pwd`/tee/tee_dev -name '*.c' -o -name '*.h' >> filelist
    find `pwd`/XXXsdk/osal -name '*.c' -o -name '*.h' >> filelist
    find `pwd`/XXXsdk/ta_XXX -name '*.c' -o -name '*.h'  >> filelist
    find `pwd`/XXXsdk/XXX/gtest/testcases  -name '*.c' -o -name '*.h' -name '*.cpp' >> filelist
    find `pwd`/XXXsdk/XXX/inc  -name '*.c' -o -name '*.h' >> filelist
    find `pwd`/XXXsdk/XXX/src  -name '*.c' -o -name '*.h' >> filelist
    find `pwd`/XXXsdk/XXX/test/client_test  -name '*.c' -o -name '*.h' >> filelist
    find `pwd`/XXXsdk/XXX/tools/XXXdiag  -name '*.c' -o -name '*.h' >> filelist
    find `pwd`/XXXsdk/XXX/tools/XXXdmxutil  -name '*.c' -o -name '*.h' >> filelist
    find `pwd`/XXXsdk/XXX/tools/mdb  -name '*.c' -o -name '*.h' >> filelist
    find `pwd`/XXXsdk/XXX/tools/set_plane  -name '*.c' -o -name '*.h' >> filelist
}



if [ $1 = "proj02" ]; then
   echo "hello proj02"
   gen_proj02
   echo "generate proj02 ctags and cscope done!"
fi

if [ $1 = "proj" ]; then
   echo "hello proj"
   gen_proj
   echo "Generage proj ctags and cscope done!"
fi


