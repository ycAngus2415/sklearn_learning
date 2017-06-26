import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.rand(5, 3), np.arange(5), {"a", "b", "c"})
print(df)
print(df.loc[0,"a"])
print(df.iloc[0,2])
print(df[df["a"]>0.2]["a"])
print(df[df["a"]>0.2]["a"].size)
