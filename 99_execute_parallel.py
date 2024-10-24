from joblib import Parallel, delayed
import subprocess as sp


def job_to_do(index):
    sp.call([
        "python",
        "/home/mszul/git/act_mis_new/05_all_sensor_perf_searchlight_sliding.py",
        str(index), "erf"
    ])

Parallel(n_jobs=6)(delayed(job_to_do)(index) for index in range(62))