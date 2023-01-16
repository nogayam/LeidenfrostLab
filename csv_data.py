import config, csv
import pandas as pd
import string

def get_file_data(file_path):
    try:
        df = pd.read_csv(file_path, skiprows=1, usecols=range(3)) #with x needs to be 3
    except Exception as e:
        try:
            df = pd.read_csv(file_path, sep='\t', skiprows=1, usecols=range(3))
        except Exception as e:
            df = pd.read_csv(file_path, usecols=range(2))

    return df


def get_file_metadata(file_path): #temp-serial-dropletMassPoint.csv - 290-5-A.csv/ lab5-290-5-A.csv
    file_name = file_path.split("/")[-1]
    if " " in file_name:
        file_name.replace(" ", "")

    if "lab" not in file_name:
        temperature = int(file_name.split("-")[0])
        temp_serial = int(file_name.split("-")[1])
        droplet_mass_point = file_name[file_name.find(".") - 1]
    elif "lab" in file_name:
        temperature = int(file_name.split("-")[1])
        temp_serial = int(file_name.split("-")[2])
        droplet_mass_point = file_name[file_name.find(".") - 1]

    return {"file_name": file_name, "temperature": temperature,
            "temperature_serial": temp_serial, "droplet": droplet_mass_point}


def get_all_files(file_list):
    files_dict_list = []
    for file in file_list:
        file_metadata = get_file_metadata(file)
        file_data = get_file_data(file)
        file_dict = {"metadata": file_metadata, "data": file_data}
        files_dict_list.append(file_dict)
    return files_dict_list


def write_to_csv(file_path, data_list):
    with open(file_path, 'w') as f:
        writer = csv.writer(f)
        print(data_list)
        for row in data_list:
            writer.writerow(row)


def get_matlab_file_data(file_path):
    try:
        df = pd.read_csv(file_path).transpose()
        df.columns = ['t', 'y']
    except Exception as e:
        print(e)
    return df


def matlab_split_to_drops(df, zeros_threshold=5):
    zeros = df[df['y'] == 0]
    zero_indices = zeros.index
    sequences = []
    start_index = None
    end_index = None
    for i in range(len(zero_indices)):
        if start_index is None:
            start_index = zero_indices[i]
        elif zero_indices[i] - zero_indices[i-1] != 1:
            end_index = zero_indices[i-1]
            sequences.append((start_index, end_index))
            start_index = zero_indices[i]

        if i == len(zero_indices) - 1:
            end_index = zero_indices[i]
            sequences.append((start_index, end_index))

    for i, (start, end) in enumerate(sequences):
        if end - start >= zeros_threshold:
            new_df = df.iloc[start:end+1]

def get_matlab_file_metadata(file_path): #Data(298 - 3.avi).csv
    file_name = file_path.split("/")[-1]
    if " " in file_name:
        file_name.replace(" ", "")

    if "lab" not in file_name:
        temperature = int(file_name.split("-")[0][5:])
        temp_serial = int(file_name.split("-")[1])
        droplet_mass_point = file_name[file_name.find(".") - 1]
    elif "lab" in file_name:
        temperature = int(file_name.split("-")[1])
        temp_serial = int(file_name.split("-")[2])
        droplet_mass_point = file_name[file_name.find(".") - 1]

    return {"file_name": file_name, "temperature": temperature,
            "temperature_serial": temp_serial, "droplet": droplet_mass_point}






