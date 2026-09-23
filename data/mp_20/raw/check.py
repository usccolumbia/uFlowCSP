import pandas as pd
df=pd.read_csv('all.csv')
print(df.columns)

print(df.shape,len(list(df['material_id'].unique())))
