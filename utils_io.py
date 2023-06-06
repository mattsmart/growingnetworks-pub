import datetime
import os
import pickle

from settings import DIR_RUNS


def run_subdir_setup(dir_runs=DIR_RUNS, run_subfolder=None, timedir_override=None, minimal_mode=False):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%I.%M.%S%p")
    experiment_dir = dir_runs

    if timedir_override is not None:
        time_folder = timedir_override
    else:
        time_folder = current_time

    if run_subfolder is None:
        dir_current_run = experiment_dir + os.sep + time_folder
    else:
        if os.path.isabs(run_subfolder):
            dir_current_run = run_subfolder + os.sep + time_folder
        else:
            dir_current_run = experiment_dir + os.sep + run_subfolder + os.sep + time_folder

    # make subfolders in the timestamped run directory:
    dir_plots = os.path.join(dir_current_run, "plots")
    dir_states = os.path.join(dir_current_run, "states")

    if minimal_mode:
        dir_list = [dir_runs, dir_current_run]
    else:
        dir_list = [dir_runs, dir_current_run, dir_plots, dir_states]
    for dirs in dir_list:
        if not os.path.exists(dirs):
            os.makedirs(dirs)
            
    # io path storage to pass around
    io_dict = {'dir_base': dir_current_run,
               'dir_plots': dir_plots,
               'dir_states': dir_states,
               'runinfo': dir_current_run + os.sep + 'run_info.txt'}

    # make minimal run_info settings file with first line as the base output dir
    runinfo_append(io_dict, ('dir_base', dir_current_run))

    return io_dict


def runinfo_append(io_dict, info_list, multi=False):
    # multi: list of list flag
    if multi:
        with open(io_dict['runinfo'], 'a') as runinfo:
            for line in info_list:
                runinfo.write(','.join(str(s) for s in line) + '\n')
    else:
        with open(io_dict['runinfo'], 'a') as runinfo:
            runinfo.write(','.join(str(s) for s in info_list) + '\n')
    return


def pickle_load(fpath):
    with open(fpath, 'rb') as pickle_file:
        loaded_object = pickle.load(pickle_file)
    return loaded_object
