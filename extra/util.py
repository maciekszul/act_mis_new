import numpy as np
from pathlib import Path


def adjust_QRS_peaks(signal, peaks, half_window, positive=True):
    new_peaks = np.zeros(len(peaks))
    for peak_ix, peak in enumerate(peaks):
        start_ix = peak - half_window
        if start_ix < 0:
            start_ix = 0
                    
        end_ix = peak + half_window
        signal_slice = signal[start_ix:end_ix]
        
        if positive == True:
            slice_max_ix = np.argmax(signal_slice)
        else:
            slice_max_ix = np.argmin(signal_slice)
        old_new_dist = slice_max_ix - half_window
        new_peak = peak + old_new_dist
        new_peaks[peak_ix] = new_peak
    new_peaks = np.unique(new_peaks).astype(int)
    return new_peaks


def check_many(multiple, target, func=None):
    """
    Checks for a presence of strings in a target strings.
    
    Parameters:   
    multiple (list): strings to be found in target string
    target (str): target string
    func (str): "all" or "any", use the fuction to search for any or all strings in the filename.
    
    Notes:
    - this function works really well with if statement for list comprehension
    """
    func_dict = {
        "all": all, "any": any
    }
    if func in func_dict.keys():
        use_func = func_dict[func]
    elif func == None:
        raise ValueError("pick function 'all' or 'any'")
    check_ = []
    for i in multiple:
        check_.append(i in target)
    return use_func(check_)


def get_files(target_path, suffix, strings=[""], prefix=None, check="all", depth="all"):
    """
    Returns a list of the files with specific extension, prefix and name containing
    specific strings. Either all files in the directory or in this directory.
    
    Parameters:
    target path (str or pathlib.Path or os.Path): the most shallow searched directory
    suffix (str): file extension in "*.ext" format
    strings (list of str): list of strings searched in the file name
    prefix (str): limit the output list to the file manes starting with prefix
    check (str): "all" or "any", use the fuction to search for any or all strings in the filename.
    depth (str): "all" or "one", depth of search (recurrent or shallow)
    
    Notes:
    - returns a list of pathlib.Path objects
    """
    path = Path(target_path)
    if depth == "all":
        subdirs = [subdir for subdir in path.rglob(suffix) if check_many(strings, str(subdir.name), check)]
        subdirs.sort()
        if isinstance(prefix, str):
            subdirs = [subdir for subdir in subdirs if subdir.name.startswith(prefix)]
        return subdirs
    elif depth == "one":
        subdirs = [subdir for subdir in path.iterdir() if
                   all([subdir.is_file(), subdir.suffix == suffix[1:], check_many(strings, str(subdir.name), check)])]
        if isinstance(prefix, str):
            subdirs = [subdir for subdir in subdirs if subdir.name.startswith(prefix)]
        subdirs.sort()
        return subdirs


def get_directories(target_path, strings=[""], check="all", depth="all"):
    """
    Returns a list of directories in the path (or all subdirectories) containing
    specified strings.
    
    Parameters:
    target path (str or pathlib.Path or os.Path): the most shallow searched directory
    depth (str): "all" or "one", depth of search (recurrent or shallow)
    
    Notes:
    - returns a list of pathlib.Path objects
    """
    path = Path(target_path)
    subdirs = []
    if depth == "all":
        subdirs = [subdir for subdir in path.glob("**/") if check_many(strings, str(subdir), check)]
    elif depth == "one":
        subdirs = [subdir for subdir in path.iterdir() if subdir.is_dir() if check_many(strings, str(subdir), check)]
    subdirs.sort()
    return subdirs


def make_directory(root_path, extended_dir, check=False):
    """
    Creates a directory along with the intermediate directories.
    
    root_path (str or pathlib.Path or os.Path): the root directory
    extended_dir(str or list): directory or directories to create within root_path
    
    Notes:
    - can return a created path or a False if check=True
    """
    root_path = Path(root_path)
    if isinstance(extended_dir, list):
        root_path = root_path.joinpath(*extended_dir)
    else:
        root_path = root_path.joinpath(extended_dir)

    root_path.mkdir(parents=True, exist_ok=True)
    if all([check, root_path.exists()]):
        return root_path
    elif check:
        return root_path.exists()