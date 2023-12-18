import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Assuming you have a DataFrame df with columns 'var1', 'var2', 'var3'
df = pd.DataFrame({
    'var1': [1, 2, 3, 4, 5],
    'var2': [2, 3, 4, 5, 6],
    'var3': [3, 4, 5, 6, 7]
})

X = df[['var1', 'var2']]
Y = df['var3']

X = sm.add_constant(X) # adding a constant

model = sm.OLS(Y, X).fit()
predictions = model.predict(X) 

print_model = model.summary()
print(print_model)

# Create scatterplots with fitted curves
sns.regplot(x='var1', y='var3', data=df, line_kws={"color": "red"})
plt.show()

sns.regplot(x='var2', y='var3', data=df, line_kws={"color": "red"})
plt.show()