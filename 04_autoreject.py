import sys
import json
import numpy as np
from pathlib import Path
from mne import read_epochs
from extra.util import get_directories, get_files, make_directory, check_many
from autoreject import AutoReject, Ransac, get_rejection_threshold


try:
    index = int(sys.argv[1])
except:
    print("incorrect arguments")
    sys.exit()

processed_path = Path("/home/common/mszul/act_mis/MEG/processed")
fif_paths = get_files(processed_path, "*.fif", strings=["observation", "-epo.fif"])
fif_paths = [i for i in fif_paths if "autoreject" not in i.stem]
fif_paths.sort()

fif_path = fif_paths[index]

print(fif_path.stem, f"{index+1}/{len(fif_paths)}")

epochs = read_epochs(fif_path, verbose=False)
epochs_fit = epochs.pick("mag", verbose=False)
ar = AutoReject(verbose=False)
clean_epochs, reject_log = ar.fit_transform(epochs_fit, return_log=True)

clean_file = fif_path.parent.joinpath("autoreject_" + fif_path.name)
reject_log_file = fif_path.parent.joinpath("autoreject_" + fif_path.stem + ".h5")

clean_epochs.save(clean_file, fmt='single', verbose=False)
ar.save(reject_log_file, overwrite=True)