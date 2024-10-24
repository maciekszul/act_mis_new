import sys
import json
from pathlib import Path
from mne import events_from_annotations, Epochs
from mne.io import read_raw
from mne.preprocessing import read_ica
from extra.util import get_directories, get_files, make_directory, check_many


event_id_dicts = {
    "erf": [{"obs_congr": 5, "obs_incongr": 10}, 5],
    "tf": [{"obs_congr": 5, "obs_incongr": 10}, 2]
}

try:
    index = int(sys.argv[1])
except:
    print("incorrect arguments")
    sys.exit()

try:
    dict_output = event_id_dicts[str(sys.argv[2])]
except:
    print("incorrect arguments")
    sys.exit()

processed_path = Path("/home/common/mszul/act_mis/MEG/processed")
fif_paths = get_files(processed_path, "*.fif", strings=["zapline", "act_mis", "-raw.fif"])

key = fif_paths[index]
misc_files = get_files(key.parent, "*.*", strings=[key.stem[:-4]])
misc_files.sort()

ecg_score = [i for i in misc_files if "ecg_score" in i.stem][0]
ica_path = [i for i in misc_files if "ica" in i.stem][0]
raw_path = [i for i in misc_files if "raw" in i.stem][0]

ica = read_ica(ica_path, verbose=False)
raw = read_raw(raw_path, verbose=False, preload=True)
raw = ica.apply(raw, verbose=False)

events, event_ids = events_from_annotations(
    raw, event_id=dict_output[0]
)

epochs = Epochs(
    raw, events, event_id=event_ids, tmin=-0.1,
    tmax=1, decim=dict_output[1], baseline=None
)

output_file = key.parent.joinpath("" + key.stem[:-3] + "observation" + "-epo.fif")

epochs.save(output_file, fmt="single", overwrite=True)
