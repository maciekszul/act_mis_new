from zipfile import ZipFile
import shutil as sh
from pathlib import Path
from extra.util import get_directories, get_files, make_directory, check_many


dataset_dir = Path("/home/common/mszul/act_mis/MEG")
zip_files = get_files(dataset_dir, "*.zip")

for zip_file in zip_files:
    print("START", zip_file.stem)
    archive = ZipFile(zip_file, "r")
    archive_contains = archive.namelist()
    folders = [Path(i) for i in archive_contains if i.endswith(".ds/")]
    folders = [i for i in folders if len(i.parts) == 2]


    all_to_extract = [i for i in archive_contains if any([i.startswith(str(x)) for x in folders])]
    for extract in all_to_extract:
        archive.extract(extract, dataset_dir)
        print("EXTRACTED:", extract)
    print("DONE", zip_file.stem)


dataset_dir = Path("/home/common/mszul/act_mis/ANAT")
zip_files = get_files(dataset_dir, "*.zip")

for zip_file in zip_files:
    print("START", zip_file.stem)
    archive = ZipFile(zip_file, "r")
    archive.extractall(dataset_dir)
    print("DONE", zip_file.stem)