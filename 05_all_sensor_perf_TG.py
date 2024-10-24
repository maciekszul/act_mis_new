import sys
import pickle
import numpy as np
from pathlib import Path
from util import get_files
from mne import read_epochs
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

x = epochs.get_data(copy=False)

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
    n_jobs=10, verbose=False
)

pickle_file = OUTPUT.joinpath(
    f"generalising_estimator_{fif_path.stem}.pickle"
)

with open(pickle_file, "wb") as file:
    pickle.dump(scores, file)