import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import scipy.stats as stats


# Multiple hists
def plot_grouped_hists(df: pd.DataFrame,
                       pal: list,
                       nbins: int = 20,
                       ncols: int = 3,
                       alpha: float = 0.5,
                       density: bool = True) -> None:
    """Plot several distributions with mean"""
    df = df.copy()
    df_cols = df.columns
    df_cols_num = len(df_cols)
    col_num = 0
    
    rows = int(np.ceil(df_cols_num / ncols))
    fig, axises = plt.subplots(nrows=rows,
                               ncols=ncols,
                               figsize=(24, 8*rows))
    
    for nrow in range(axises.shape[0]):
        for ncol in range(axises.shape[1]):
            if col_num > df_cols_num - 1:
                break
            mean = df[df_cols[col_num]].mean()
            median = df[df_cols[col_num]].median()
            sigma = df[df_cols[col_num]].std()
            gauss_x = np.linspace(mean - 3*sigma, mean + 3*sigma, 100)
            
            axises[nrow, ncol].hist(
                df[df_cols[col_num]], 
                nbins, 
                alpha=alpha, 
                density=density,
                histtype='bar',
                label='real data')
            axises[nrow, ncol].axvline(x=mean, label=f'Mean', color=pal[0])
            axises[nrow, ncol].axvline(x=median, label=f'Median', color=pal[1])
            axises[nrow, ncol].axvline(x=mean+sigma, label=f'Sigma', color=pal[2])
            axises[nrow, ncol].axvline(x=mean-sigma, label=f'Sigma', color=pal[2])
            axises[nrow, ncol].axvline(x=mean+3*sigma, label=f'3 Sigma', color=pal[3])
            axises[nrow, ncol].axvline(x=mean-3*sigma, label=f'3 Sigma', color=pal[3])
            axises[nrow, ncol].plot(
                gauss_x, stats.norm.pdf(gauss_x, mean, sigma), label='Ideal')
            axises[nrow, ncol].set_title(f'{df_cols[col_num]}')
            col_num += 1
    fig.tight_layout()
    plt.legend(loc="upper right")
    plt.show()
    
    
def plot_corr_matrix(df: pd.DataFrame, 
                     dates: tuple | None = None,
                     figsize=(20, 10)) -> None:
    """Plot correlation matrix"""
    plt.figure(figsize=figsize)
    if dates:
        filtered_rows = (df.index <= dates[1]) & (df.index >= dates[0])
        df = df.loc[filtered_rows]

    # To show one half
    mask = np.triu(np.ones_like(df.corr(), dtype=bool))

    heatmap = sns.heatmap(
        round(df.corr(), 2),
        mask=mask,
        vmin=-1,
        vmax=1,
        annot=True,
        annot_kws={"fontsize":20})
    # Give a title to the heatmap. 
    # Pad defines the distance of the title from the top of the heatmap.
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':16}, pad=12)
