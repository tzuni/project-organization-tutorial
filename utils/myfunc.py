import pandas as pd
import seaborn as sns
from sklearn import decomposition
from sklearn.cluster import OPTICS, cluster_optics_dbscan
import matplotlib.pyplot as plt

columns = [
        'class',
        'alcohol',
        'malic acid',
        'ash',
        'alkalinity of ash',
        'magnesium',
        'total phenols',
        'flavinoids',
        'nonflavinoid phenols',
        'proanthocyanins',
        'color intensity',
        'hue',
        'od280/od315',
        'proline'
    ]

def load_data(
    filename='data/wine.data',
    columns = columns):
    return pd.read_csv(filename, names=columns)

def do_pca(df, components=3):
    pca = decomposition.PCA(n_components=components, whiten=True)
    res = pca.fit_transform(df[df.columns[1:]])
    labels = ['pc{}'.format(i + 1) for i in range(components)]
    pcdf = pd.DataFrame(res, columns=labels)
    pcdf['class'] = df['class']
    return pcdf

def get_colors(ncolors):
    if ncolors < 10:
        return sns.color_palette(palette='tab10', n_colors=ncolors)
    if ncolors < 20:
        return sns.color_palette(palette='tab20', n_colors=ncolors)
    else:
        print("Too many colors requested, only providing 20")
        return sns.color_palette(palette='tab20', n_colors=ncolors)

def cluster_with_my_favorite_settings(df, pcdf):
    clust = OPTICS(min_samples=10, metric='minkowski')
    clust.fit(df)

    ncolors = max(clust.labels_)
    colors = get_colors(ncolors)

    fig, ax = plt.subplots()
    for cat, color in zip(range(0, 5), colors):
        Xk = pcdf[clust.labels_ == cat].copy()
        ax.scatter(Xk['pc1'], Xk['pc2'], color=color, alpha=0.3)
        sel_no_cluster = clust.labels_ == -1
        ax.scatter(pcdf.loc[sel_no_cluster]['pc1'], pcdf.loc[sel_no_cluster]['pc2'], marker="+", alpha=0.1)