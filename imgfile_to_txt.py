# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 11:15:20 2016

@author: wangs
"""

import os

def listFilesToTxt(img_dir,imgfile,wildcard,recursion):
    exts = wildcard.split(" ")
    files = os.listdir(img_dir)
    for name in files:
        fullname=os.path.join(img_dir,name)
        if(os.path.isdir(fullname) & recursion):
            listFilesToTxt(fullname,imgfile,wildcard,recursion)
        else:
            for ext in exts:
                if(name.endswith(ext)):
                    imgfile.write(fullname[fullname.find('/')+1: len(fullname)] + "\n")
                    break

def imageFileToTxt(img_dir, txtfile):
  wildcard = ".jpg" #通配符 
  imgfile = open(txtfile,"w")
  if not imgfile:
    print ("cannot open the file %s for writing" % txtfile)

  listFilesToTxt(img_dir,imgfile,wildcard, 1)
 
  imgfile.close()