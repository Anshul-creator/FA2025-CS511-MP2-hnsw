import faiss
import h5py
import numpy as np
import os

def _pick_dataset(f, candidates):
    "Return the first dataset present from the candidates list."
    for name in candidates:
        if name in f:
            return np.array(f[name])
    return None

def evaluate_hnsw():
    # Path to the dataset placed in the repo root (same dir as this script)
    data_path = os.path.join(os.path.dirname(__file__), "SIFT1M.hdf5")
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Couldn't find SIFT1M.hdf5 at: {data_path}")

    with h5py.File(data_path, "r") as f:
        # Common naming variants across SIFT1M/HDF5 packages
        # Prefer 'base'/'database' for the index; fall back to 'train'
        db = _pick_dataset(f, ["base", "database", "db", "train", "data"])
        queries = _pick_dataset(f, ["query", "test", "queries"])

        if db is None:
            raise KeyError("No database embeddings found. Expected one of: base/database/db/train/data.")
        if queries is None:
            raise KeyError("No query/test embeddings found. Expected one of: query/test/queries.")
    
    db = np.asarray(db, dtype="float32", order="C")
    queries = np.asarray(queries, dtype="float32", order="C")

    d = db.shape[1]
    M = 16
    ef_construction = 200
    ef_search = 200

    index = faiss.IndexHNSWFlat(d, M)  
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    
    index.add(db)
    
    q0 = queries[0:1]  

    # Top-10 ANN
    k = 10
    _, I = index.search(q0, k)  
    
    out_path = os.path.join(os.path.dirname(__file__), "output.txt")
    with open(out_path, "w") as fh:
        for idx in I[0]:
            fh.write(f"{int(idx)}\n")

if __name__ == "__main__":
    evaluate_hnsw()