from __future__ import annotations

import numpy as np


def _project_umap(vectors: np.ndarray) -> np.ndarray:
    import umap

    n_neighbors = int(min(15, max(2, vectors.shape[0] - 1)))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.3,
        metric="cosine",
        random_state=42,
    )
    return reducer.fit_transform(vectors)


def _project_tsne(vectors: np.ndarray) -> np.ndarray:
    from sklearn.manifold import TSNE

    perplexity = float(min(30, max(5, vectors.shape[0] // 4)))
    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=42,
    )
    return reducer.fit_transform(vectors)


def _project_pca(vectors: np.ndarray) -> np.ndarray:
    from sklearn.decomposition import PCA

    return PCA(n_components=2, random_state=42).fit_transform(vectors)


def project_entities(vectors: np.ndarray, method: str) -> np.ndarray:
    try:
        if method == "tsne":
            coords = _project_tsne(vectors)
        else:
            coords = _project_umap(vectors)
    except Exception:
        coords = _project_pca(vectors)
    return np.asarray(coords, dtype=np.float32)
