from numpy.random import RandomState
import pandas as pd

df=pd.read_csv("myimages/metadata.csv")
print(df.shape)
rng=RandomState()

train=df.sample(frac=0.7, random_state=rng)
test=df.loc[~df.index.isin(train.index)]

train.to_csv("myimages/trainmeta.csv")
test.to_csv("myimages/testmeta.csv")