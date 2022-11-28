import csv_data, droplet, config, glob, os

analysis_modules = config.ANALYSIS_MODULES
folder_path = config.CSV_FOLDER
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
output_path = config.OUTPUT_PATH.format(name=config.OUTPUT_FILE_NAME)


def main():
    files = csv_data.get_all_files(csv_files)
    max_heights_data = [["Temp [c]", "Max Height [mm]"]]
    for file in files:
        print(file)
        drop = droplet.Droplet(identifier=file["metadata"]["droplet"],
                                  temp=file["metadata"]["temperature"],
                                  temp_serial=file["metadata"]["temperature_serial"],
                                  raw_data=file["data"])
        drop.get_max_height()
        max_heights_data.append([drop.temperature, drop.max_height])

    csv_data.write_to_csv(output_path, max_heights_data)
    print(f"successfully wrote data to {output_path}")
    return


if __name__ == "__main__":
    main()





