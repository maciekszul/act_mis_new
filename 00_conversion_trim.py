from zipfile import ZipFile
import shutil as sh
from pathlib import Path
from extra.util import get_directories, get_files, make_directory, check_many
from mne.io import read_raw_ctf


raw_dir = Path("/home/common/mszul/act_mis/MEG/raw")
processed_dir = Path("/home/common/mszul/act_mis/MEG/processed")

all_dirs = get_directories(raw_dir, strings="_04.ds")
all_dirs = [i for i in all_dirs if not i.name == "hz.ds"]
all_dirs.sort()

def clean_raw(ds, output_file):
    raw = read_raw_ctf(
        ds, clean_names=True, verbose=False, preload=True
    )
    set_ch = {"EEG057":"eog", "EEG058": "eog", "UPPT001": "stim"}
    raw = raw.set_channel_types(set_ch)
    first_trigger = raw.annotations.onset[0] - 3
    last_trigger = raw.annotations.onset[-1] + 3
    raw = raw.crop(
        tmin=first_trigger, tmax=last_trigger
    )
    raw = raw.filter(None, 200.0, picks=["meg", "eeg"])
    raw.save(output_file, fmt="single", overwrite=True)

for ix, ds in enumerate(all_dirs):
    file_ix = str(ix%2).zfill(3)
    folder_name = ds.parent.parts[-1]
    output_dir = make_directory(processed_dir, folder_name, check=True)
    output_file = output_dir.joinpath(f"act_mis-{folder_name}-ses-{file_ix}-raw.fif")
    clean_raw(ds, output_file)
    print("SAVED:", output_file)
    
