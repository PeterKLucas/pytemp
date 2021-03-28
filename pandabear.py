import pandas as pd
import numpy as np

print(pd.__version__)

df = pd.read_csv('c:\pytemp\data.csv')

print(df)

#print(df.to_string()) 
a = df['BBL']
#print( df['BBL'])
myvar = pd.Series(a)

print(myvar)

data = np.array(df['ProgramName'])
s = pd.Series(data)
print(s)

data2 = {'Name':np.array(df['ProgramName']),'NTA':np.array(df['NTA'])}
df1 = pd.DataFrame(data2)
print(df1)

print('################################')

print(df1.iloc[2])

print(df)
print('now')

print(df.groupby('NTA')['Latitude'].mean())

print('now then')


