import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  


df1 = pd.read_csv('table1.csv')
df2 = pd.read_csv('table2.csv')
df3 = pd.read_csv('table3.csv')


df_combined = pd.merge(df1, df2[['SMILES', 'pIC50']], on='SMILES', how='left', suffixes=('', '_from_df2'))
df_combined = pd.merge(df_combined, df3[['SMILES', 'pIC50']], on='SMILES', how='left', suffixes=('', '_from_df3'))


if 'pIC50_from_df2' in df_combined.columns:
    df_combined['pIC50'] = df_combined['pIC50'].fillna(df_combined['pIC50_from_df2'])
if 'pIC50_from_df3' in df_combined.columns:
    df_combined['pIC50'] = df_combined['pIC50'].fillna(df_combined['pIC50_from_df3'])

df_combined.drop(columns=[col for col in df_combined.columns if 'pIC50_from_' in col], inplace=True)


df_combined = df_combined.dropna(subset=['pIC50'])


plt.figure(figsize=(10, 6))
sns.kdeplot(df_combined['pIC50'], fill=True, common_norm=False, palette="crest", linewidth=1.5, alpha=0.7)
plt.title('Smooth Distribution of pIC50 Values')
plt.xlabel('pIC50')
plt.ylabel('Density')
plt.grid(True)
plt.savefig('pIC50_smooth_distribution.png')
plt.show()
