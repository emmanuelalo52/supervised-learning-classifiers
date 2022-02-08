import os
import pandas as pd
s=os.path.join('https://archive.ics.uci.edu','ml','machine-learning-databases','iris','iris.data')
print('URL:',s)   
df=pd.read_csv(s,header=None,encoding='utf8')
df.tail()