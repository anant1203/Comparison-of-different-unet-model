#!/bin/bash
wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=anant1203&password=Helloanant@12&submit=Login' https://www.cityscapes-dataset.com/login/
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=13