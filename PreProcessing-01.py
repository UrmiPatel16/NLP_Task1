# -*- coding: utf-8 -*-
"""NLP_Assign_1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16i2V-CLESWHJCy4BG15ZIL_I8aFx9reK
"""

#from google.colab import drive
#drive.mount('/content/drive')

import pandas as pd
import xml.etree.ElementTree as et
import os
from pathlib import Path
import glob
from xml.etree import ElementTree
import numpy as np
import csv



import xml.etree.ElementTree as et

#paths = ['/Users/urmi/Documents/NLP/Assignment_1/pmc-text-01/28']
paths = []

# traverse all directories - all sub folders within one folder 
for dirpath, dirs, files in os.walk("/Users/urmi/Documents/NLP/Assignment_1/pmc-text-03/"):
  for dir in dirs:
    paths.append(dirpath+dir)
#print(paths)


ds_cols = ["Article_Id","Journal_Id","Publisher_Name","Article_Title","Abstract"]
rows = []

for dir in paths:
  #print(dir)
  for root, dirs, files in os.walk(dir+'/'):
      print(root)
      for file in files:
          #print(root)
          filename, extension = os.path.splitext(file)
          if extension == '.nxml':
            #print(root+file)
            try:
              xtree = et.parse(root+file)
            except:
              print(root+file)

            xroot = xtree.getroot()

            #ds_cols = ["Journal_Id", "Issn","Article_Id"]
            # rows = []
            # read attributes of xml file and store into one dataframe
            for node in xroot.iterfind('front'):
              if node.find('journal-meta') is not None:
                Jid = node.find('journal-meta').find('journal-id').text 
               
    
                publisherName = ""
                if node.find('journal-meta').find('publisher') is not None:
                  publisherName = node.find('journal-meta').find('publisher').find('publisher-name').text

              if node.find('article-meta') is not None:
                articleId = node.find('article-meta').find('.//article-id[@pub-id-type="pmc"]').text 
                
               # articleTitle = node.find('article-meta')
                articleTitle = ""
                if node.find('article-meta').find('title-group') is not None:
                  articleTitle = node.find('article-meta').find('title-group').find('article-title').text

                #abstract = node.find('article-meta').find('abstract').find('sec').find('p').text
                abstract = node.find('article-meta').find('abstract')
                abstract1 = ""
                if abstract is not None:
                  p = abstract.findall('sec/p')
                  for x in p:
                    if x.text is not None:
                      abstract1 += x.text
                  #print(abstract1)
                # print({x.text for x in abstract})
                #print(title)
                rows.append({"Article_Title":articleTitle, 
                              "Journal_Id":Jid, 
                              "Publisher_Name":publisherName,
                              "Article_Id": articleId,
                              "Abstract": abstract1
                              })

ds = pd.DataFrame(rows, columns = ds_cols)
pd.set_option('display.max_columns',None)
print(ds)
#print(ds['Article_Title'])
ds.to_csv('/Users/urmi/Documents/NLP/Assignment_1/second-csv/csv-03.csv') # convert xml data into .csv file data