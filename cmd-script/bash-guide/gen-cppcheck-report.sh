#!/bin/bash

function generate_report()
{
    cppcheck_report=ampsdk_cppcheck.report
    if [ -e ${cppcheck_report} ]; then
        rm ${cppcheck_report}
    fi

    echo "num   module" >> ${cppcheck_report}
    echo "----------------------" >> ${cppcheck_report}

    tmpfile=cppcheck_tmp_file
    cat ${cppcheck_log} | while read line; do
        # echo $line
        issue_num=$(echo $line | cut -d ' ' -f 1)
        module_cppcheck_report=$(echo $line | cut -d ' ' -f 2)
        module_name=$(echo $module_cppcheck_report | cut -d '/' -f 7)
        if [ ${module_name} = "ta" ]; then
            module_name=$(echo $module_cppcheck_report | cut -d '/' -f 8)
        fi

        if [ ${issue_num} -ne 0 ]; then
            echo "-----------------------------------" >> ${tmpfile}
            echo ${issue_num} "   " ${module_name} | tee -a ${cppcheck_report} ${tmpfile} > /dev/null
            echo " " >> ${tmpfile}
            
            if [ ${module_cppcheck_report} != "total" ]; then
                cat ${module_cppcheck_report} >> ${tmpfile}
            fi
            echo " " >> ${tmpfile}
        fi
    done

    echo "-----------------------------------" >> ${tmpfile}
    echo " " >> ${cppcheck_report}
    cat ${tmpfile} >> ${cppcheck_report}

    [ -e ${tmpfile} ]
    rm ${tmpfile}
}

function main()
{
    cppcheck_log=cppcheck_log_file
    find -name cppcheck_report | xargs wc -l >> ${cppcheck_log}
    if [ -s ${cppcheck_log} ]; then
        generate_report
    else
        echo "oop! cppcheck go wrong!"
    fi

    [ -f ${cppcheck_log} ]
    rm ${cppcheck_log}

    echo "ampsdk cppcheck report generated: ampsdk_cppcheck.report "
}

main
