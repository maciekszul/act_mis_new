import sys
import json
from pathlib import Path
from extra.util import get_directories, get_files, make_directory, check_many
from mne.io import read_raw_fif, RawArray
from mne.preprocessing import EOGRegression, ICA
from meegkit.dss import dss_line_iter
from ecgdetectors import Detectors
import numpy as np
from copy import copy
from extra.util import adjust_QRS_peaks


try:
    index = int(sys.argv[1])
except:
    print("incorrect arguments")
    sys.exit()


processed_path =  "/home/common/mszul/act_mis/MEG/processed"
raw_fif_paths = get_files(processed_path, "*.fif", strings=["act_mis", "-raw.fif"])
raw_fif_paths = [i for i in raw_fif_paths if "zapline" not in i.stem]
raw_fif_paths.sort()

raw_fif_path = raw_fif_paths[index]

raw = read_raw_fif(raw_fif_path, verbose=False, preload=True)
raw_data = raw.get_data()
raw_info = raw.info
first_samp = raw.first_samp
annotations = raw.annotations
mag_ix = np.array([i for i, lab in enumerate(raw.get_channel_types()) if lab == "mag"])

rd = np.array_split(raw_data[mag_ix], 10, axis=1)
rd = [np.moveaxis(i, [0,1], [1,0]) for i in rd]
rd_f = []

for ix, ss in enumerate(rd):
    print(raw_fif_path.name, f"{ix+1}/10")
    data, iters = dss_line_iter(ss, fline=50.0, sfreq=raw_info["sfreq"], spot_sz=5.5, win_sz=10, nfft=1024)
    rd_f.append(data)
del rd
rd_f = [np.moveaxis(i, [0,1], [1,0]) for i in rd_f]
rd_f = np.hstack(rd_f)

new_raw_data = copy(raw_data)
del raw_data
new_raw_data[mag_ix, :] = rd_f

new_raw = RawArray(
    new_raw_data,
    raw_info,
    first_samp=first_samp
)
new_raw = new_raw.set_annotations(annotations)

model_plain = EOGRegression(picks="mag", picks_artifact="eog").fit(new_raw)

new_raw = model_plain.apply(new_raw)
new_raw_path = raw_fif_path.parent.joinpath("zapline_"+raw_fif_path.name)
new_raw.save(new_raw_path, fmt="single", overwrite=True)

# ICA
new_raw = new_raw.filter(5,20, verbose=False)
n_ica = 30
ica = ICA(n_components=n_ica)
ica.fit(new_raw)
ica_data = ica.get_sources(new_raw).get_data()

sfreq = new_raw.info["sfreq"]

results_dict = {}
for ica_comp in range(n_ica):
    detector = Detectors(sfreq)
    r_peaks = detector.pan_tompkins_detector(ica_data[ica_comp])
    pos_r_peaks = adjust_QRS_peaks(ica_data[ica_comp], r_peaks, 100, positive=True)
    neg_r_peaks = adjust_QRS_peaks(ica_data[ica_comp], r_peaks, 100, positive=False)
    pos_med_diff = np.median(np.diff(pos_r_peaks) / sfreq)
    neg_med_diff = np.median(np.diff(neg_r_peaks) / sfreq)
    results_dict[ica_comp] = [pos_r_peaks.tolist(), neg_r_peaks.tolist(), pos_med_diff, neg_med_diff]

ica_filename = new_raw_path.parent.joinpath(new_raw_path.stem[:-4] + "-ica.fif")
ica.save(ica_filename, overwrite=True, verbose=False)

json_filename = new_raw_path.parent.joinpath(new_raw_path.stem[:-4] + "-ecg_score.json")

file = open(json_filename, "w")
json.dump(results_dict, file, indent=4)
file.close()