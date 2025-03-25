from sklearn.decomposition import PCA
import pandas as pd

def reduce_embeddings(embeddings, texts, n_components=2):
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(embeddings)
    df = pd.DataFrame(reduced, columns=["x", "y"] if n_components == 2 else ["x", "y", "z"])
    df["text"] = texts
    return df
