import sys
import json
from pathlib import Path
from mne.io import read_raw
from mne.preprocessing import read_ica
from extra.util import get_directories, get_files, make_directory, check_many


try:
    index = int(sys.argv[1])
except:
    print("incorrect arguments")
    sys.exit()


processed_path = Path("/home/common/mszul/act_mis/MEG/processed")
fif_paths = get_files(processed_path, "*.fif", strings=["zapline", "act_mis", "-raw.fif"])

key = fif_paths[index]
misc_files = get_files(key.parent, "*.*", strings=[key.stem[:-4]])
misc_files.sort()

ecg_score, ica_path, raw_path = misc_files

raw = read_raw(raw_path, preload=True)
raw = raw.crop(tmin=0, tmax=100).filter(5, 35)

ica = read_ica(ica_path)
ica.plot_sources(raw, block=True)

print(key.name, "excluded comps: ", ica.exclude)

ica.save(ica_path, overwrite=True)
print("saved:", ica_path)