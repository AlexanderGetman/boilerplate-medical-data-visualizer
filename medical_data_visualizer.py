import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv("medical_examination.csv")

# 2
df['overweight'] = ((df['weight']) / ((df['height'] / 100) ** 2)).apply(lambda x: 1 if x >= 25 else 0)

# 3
df[df[['cholesterol', 'gluc']] == 1] = 0
df[df[['cholesterol', 'gluc']] > 1] = 1

# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index = False).size().rename(columns={'size':'total'})

    # 7
    long = sns.catplot(data=df_cat, x='variable', y='total', kind='bar', hue='value', col='cardio', height=5, aspect=2)


    # 8
    fig = long.figure


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df[
       (df['ap_lo'] <= df['ap_hi']) & 
       (df['height'] >= df['height'].quantile(0.025)) & 
       (df['height'] <= df['height'].quantile(0.975)) &
       (df['weight'] >= df['weight'].quantile(0.025)) &
       (df['weight'] <= df['weight'].quantile(0.975))
       ]

    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14
    fig, ax = plt.subplots(figsize=(10, 5))

    # 15
    sns.heatmap(corr, mask=mask, square=True, annot=True, fmt=".01f")

    # 16
    fig.savefig('heatmap.png')
    return fig