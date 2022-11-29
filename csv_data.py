import config, csv
import pandas as pd


def get_file_data(file_path):
    try:
        df = pd.read_csv(file_path, skiprows=1, usecols=range(3))
    except Exception as e:
        df = pd.read_csv(file_path, sep='\t', skiprows=1, usecols=range(3))
    return df


def get_file_metadata(file_path): #temp-serial-dropletMassPoint.csv - 290-5-A.csv
    file_name = file_path.split("/")[-1]
    temperature = int(file_name.split("-")[0])
    temp_serial = int(file_name.split("-")[1])
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






