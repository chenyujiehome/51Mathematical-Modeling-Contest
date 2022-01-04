import csv
import numpy as np

# 读取csv至字典
csvFile = open("source.csv", "r")
reader = csv.reader(csvFile)
a=np.array(reader)
a=1