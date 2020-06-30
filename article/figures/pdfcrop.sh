#!/bin/bash
for file in ./*.pdf; do
    echo $file
    pdfcrop $file $file
done
