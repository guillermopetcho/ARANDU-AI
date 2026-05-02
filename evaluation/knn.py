import atexit
import logging

import numpy as np
import torch

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    from sklearn.neighbors import KNeighborsClassifier
    HAS_FAISS = False

from sklearn.metrics import accuracy_score

_logger = logging.getLogger("AranduSSL")

# ---------------------------------------------------------------------------
# Gestión del ciclo de vida del recurso FAISS GPU
# ---------------------------------------------------------------------------
# El recurso se inicializa de forma perezosa (lazy) la primera vez que se
# necesita y se destruye explícitamente al finalizar el proceso mediante atexit.
# Esto evita fragmentación de memoria GPU en runs largos con muchas evaluaciones.
# ---------------------------------------------------------------------------
_faiss_res = None


def _cleanup_faiss_resources() -> None:
    """Libera el recurso FAISS GPU al terminar el proceso."""
    global _faiss_res
    if _faiss_res is not None:
        del _faiss_res
        _faiss_res = None


if HAS_FAISS:
    atexit.register(_cleanup_faiss_resources)


def get_faiss_resources():
    """Retorna (e inicializa si necesario) el recurso FAISS GPU compartido."""
    global _faiss_res
    if HAS_FAISS and _faiss_res is None:
        _faiss_res = faiss.StandardGpuResources()
    return _faiss_res


@torch.no_grad()
def extract_features_fast(model, loader, device):
    """Extrae features usando el encoder (sin predictor).

    El modelo debe estar en .eval() antes de llamar a esta función.
    Retorna arrays numpy en CPU, listos para KNN o Linear Probe.
    """
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        # F1 FIX: Llamar siempre con use_predictor=False para asegurar
        # que se usan las representaciones del projector, no del predictor.
        # Soporta tanto ModelBase directo como modelos envueltos (DDP/compile).
        try:
            z = model(x, use_predictor=False)
        except TypeError:
            # Fallback: si el modelo no acepta el kwarg (ej. resnet crudo)
            z = model(x)
        feats.append(z.cpu())
        labels.append(y)
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()


def _faiss_search(
    X_train_norm: np.ndarray,
    X_val_norm: np.ndarray,
    k: int,
) -> np.ndarray:
    """Ejecuta la búsqueda KNN sobre GPU (con fallback automático a CPU).

    Separa la lógica de gestión del índice de la lógica de resultados para
    garantizar que `indices` nunca sea indefinido en el llamador.

    Returns:
        indices: array [N_val, k] con los índices de los k vecinos más cercanos.

    Raises:
        RuntimeError: si la búsqueda falla tanto en GPU como en CPU.
    """
    res = get_faiss_resources()
    cpu_index = faiss.IndexFlatIP(X_train_norm.shape[1])
    gpu_index = None

    # Intentar migrar el índice a GPU (no fatal si falla)
    try:
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        active_index = gpu_index
    except Exception as gpu_err:
        _logger.debug(f"FAISS GPU no disponible ({gpu_err}), usando índice CPU.")
        active_index = cpu_index

    indices: np.ndarray | None = None
    try:
        active_index.add(X_train_norm)
        _, indices = active_index.search(X_val_norm, k)
    except Exception as search_err:
        # Si la búsqueda en GPU falló, intentar en CPU antes de rendirse.
        # NOTA: cpu_index siempre está vacío aquí porque solo `active_index`
        # (que era el índice GPU) recibió el .add(). El cpu_index original
        # nunca fue populado, así que el .add() de abajo no es un "doble add".
        if active_index is not cpu_index:
            _logger.warning(
                f"FAISS GPU search falló ({search_err}). Reintentando en CPU..."
            )
            try:
                cpu_index.add(X_train_norm)  # cpu_index estaba vacío — primer y único add
                _, indices = cpu_index.search(X_val_norm, k)
            except Exception as cpu_err:
                raise RuntimeError(
                    f"FAISS search falló en GPU y en CPU. "
                    f"GPU error: {search_err} | CPU error: {cpu_err}"
                ) from cpu_err
        else:
            raise RuntimeError(
                f"FAISS CPU search falló: {search_err}"
            ) from search_err
    finally:
        # Liberar el índice GPU inmediatamente para no retener VRAM entre evaluaciones.
        # Se hace en finally para garantizar la liberación incluso si search() falla.
        if gpu_index is not None:
            del gpu_index

    # Invariante de salida: si llegamos aquí sin excepción, indices debe estar definido.
    assert indices is not None, "Postcondición violada: indices es None tras búsqueda exitosa."
    return indices


def fast_knn(X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray, k: int = 20) -> float:
    """Clasificador KNN rápido usando FAISS (GPU) o sklearn como fallback.

    Garantías:
      - No modifica los arrays de entrada (copias defensivas antes de normalize_L2).
      - Nunca deja `indices` indefinido, incluso ante fallos de hardware FAISS.
      - Fallback automático GPU → CPU → sklearn si FAISS no está instalado.
    """
    if not HAS_FAISS:
        # Fallback CPU: sklearn con métrica coseno
        knn = KNeighborsClassifier(n_neighbors=k, metric='cosine', n_jobs=-1)
        knn.fit(X_train, y_train)
        return float(accuracy_score(y_val, knn.predict(X_val)))

    # Copias defensivas: faiss.normalize_L2 muta los arrays in-place.
    # Sin esto, el caller vería X_train y X_val modificados silenciosamente.
    X_train_norm = np.ascontiguousarray(X_train, dtype=np.float32)
    X_val_norm   = np.ascontiguousarray(X_val,   dtype=np.float32)
    faiss.normalize_L2(X_train_norm)
    faiss.normalize_L2(X_val_norm)

    # Toda la gestión del índice y la búsqueda ocurre en el helper privado.
    # Si falla por completo, propaga RuntimeError con contexto completo.
    indices = _faiss_search(X_train_norm, X_val_norm, k)

    # Voting vectorizado: O(k) operaciones numpy sin loop Python explícito.
    # indices: [N_val, k] → neighbor_labels[i, j] = clase del j-ésimo vecino de i
    num_classes = int(y_train.max()) + 1
    neighbor_labels = y_train[indices]  # [N_val, k]
    votes = np.zeros((len(X_val_norm), num_classes), dtype=np.int32)
    np.add.at(votes, (np.arange(len(X_val_norm))[:, None], neighbor_labels), 1)

    preds = votes.argmax(axis=1)
    return float(accuracy_score(y_val, preds))