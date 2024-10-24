import subprocess as sp


for index in range(62):
    sp.call([
        "python",
        "/home/mszul/git/act_mis_new/05_all_sensor_perf_searchlight_TG.py",
        str(index)
    ])