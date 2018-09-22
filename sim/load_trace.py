import os


COOKED_TRACE_FOLDER = './cooked_traces/'


def load_trace(cooked_trace_folder=COOKED_TRACE_FOLDER):
    """Loads network traces from specified `cooked_trace_folder`

    :param cooked_trace_folder:  A string that specifies the path to the directory that contains the trace files

    :returns:
        all_cooked_time: List[List[float]]
        all_cooked_bw: List[List[float]]
        all_file_names: List[str], list of trace file name, one for each list of cooked time and cooked bandwidth
    """

    cooked_files = os.listdir(cooked_trace_folder)

    all_cooked_time = []
    all_cooked_bw = []
    all_file_names = []

    for cooked_file in cooked_files:
        file_path = os.path.join(cooked_trace_folder, cooked_file)
        cooked_time = []
        cooked_bw = []

        # print file_path
        with open(file_path, 'rb') as f:
            for line in f:
                parse = line.split()
                cooked_time.append(float(parse[0]))
                cooked_bw.append(float(parse[1]))
        all_cooked_time.append(cooked_time)
        all_cooked_bw.append(cooked_bw)
        all_file_names.append(cooked_file)

    return all_cooked_time, all_cooked_bw, all_file_names
