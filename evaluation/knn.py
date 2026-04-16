import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    from sklearn.neighbors import KNeighborsClassifier
    HAS_FAISS = False

# 🔥 FIX: Recurso persistente para evitar latencia de inicialización
_faiss_res = None

def get_faiss_resources():
    global _faiss_res
    if HAS_FAISS and _faiss_res is None:
        _faiss_res = faiss.StandardGpuResources()
    return _faiss_res

@torch.no_grad()
def extract_features_fast(model, loader, device):
    model.eval()
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        z = model(x)
        feats.append(z.cpu())
        labels.append(y)
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()

def fast_knn(X_train, y_train, X_val, y_val, k=20):
    if not HAS_FAISS:
        knn = KNeighborsClassifier(n_neighbors=k, metric='cosine', n_jobs=-1).fit(X_train, y_train)
        return accuracy_score(y_val, knn.predict(X_val))
        
    faiss.normalize_L2(X_train)
    faiss.normalize_L2(X_val)

    res = get_faiss_resources()
    index = faiss.IndexFlatIP(X_train.shape[1])
    try:
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    except Exception:
        gpu_index = index

    gpu_index.add(X_train)
    _, indices = gpu_index.search(X_val, k)

    votes = np.zeros((indices.shape[0], np.max(y_train)+1), dtype=np.int32)
    for i in range(min(k, indices.shape[1])):
        votes[np.arange(indices.shape[0]), y_train[indices[:, i]]] += 1

    preds = votes.argmax(axis=1)
    if gpu_index != index: del gpu_index
    return accuracy_score(y_val, preds)