import sys
import pickle
import numpy as np
from pathlib import Path
from util import get_files
from mne import read_epochs
from mne.channels import find_ch_adjacency
from mne.decoding import (
    SlidingEstimator, GeneralizingEstimator,
    cross_val_multiscore
)
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import StratifiedKFold

try:
    index = int(sys.argv[1])
except:
    print("incorrect arguments")
    sys.exit()


def many_is_in(multiple, target):
    check_ = []
    for i in multiple:
        check_.append(i in target)
    return any(check_)


MEG_PROC = Path("/home/mszul/datasets/act_mis/MEG/processed/")
BEH_PROC = Path("/home/mszul/datasets/act_mis/BEH/")
OUTPUT = Path("/home/mszul/datasets/act_mis/outputs/classification/")

fif_paths = get_files(
    MEG_PROC, "*.fif", 
    strings=["autoreject", "zapline", "observation-epo"]
    )

fif_path = fif_paths[index]

print(fif_path.stem, f"{index+1}/{len(fif_paths)}")

epochs = read_epochs(fif_path, verbose=False)

y = (epochs.events[:,2] == 10).astype(int)

class_weight = {
    0: np.sum(y == 0),
    1: np.sum(y == 1)
}

weights = compute_sample_weight(
    "balanced", y
)

fit_kw = {"model__sample_weight": weights}

ch_adj, ch_names = find_ch_adjacency(epochs.info, ch_type="mag")
ch_map = np.array([many_is_in(epochs.info.ch_names, i) for i in ch_names])
ch_adj = ch_adj.toarray()


results = {}
for targ_ch in epochs.ch_names:
    targ_adj_ch = np.array(ch_names)[ch_adj[np.array(ch_names) == targ_ch].flatten()]
    targ_channel_map = [i == targ_ch for i in epochs.info.ch_names]
    adj_channel_map = [many_is_in(targ_adj_ch, i) for i in epochs.info.ch_names]

    x = epochs.get_data(copy=False)
    x = x[:, adj_channel_map,:]
    k_folds = 10
    kf = StratifiedKFold(n_splits=k_folds, shuffle=True)
    cv_iter = kf.split(np.zeros(x.shape), y)

    pipeline = Pipeline([
        ("scaler", RobustScaler()),
        ("dim_reduction", PCA(n_components=0.99)),
        ("model", LinearSVC(max_iter=10000, dual=False, penalty="l1"))
    ])

    sliding_decoder = GeneralizingEstimator(
        pipeline, n_jobs=1, scoring="roc_auc", verbose=True
    )

    scores = cross_val_multiscore(
        sliding_decoder, x, y, cv=cv_iter, fit_params=fit_kw, 
        n_jobs=30, verbose=False
    )
    
    results[targ_ch] = scores

pickle_file = OUTPUT.joinpath(
    f"searchlight_TG_{fif_path.stem}.pickle"
)

with open(pickle_file, "wb") as file:
    pickle.dump(results, file)