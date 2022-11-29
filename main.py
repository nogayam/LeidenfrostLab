import csv_data, droplet, config, glob, os
import matplotlib.pyplot as plt
import pandas as pd


analysis_modules = config.ANALYSIS_MODULES
folder_path = config.CSV_FOLDER
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
output_path = config.OUTPUT_PATH.format(name=config.OUTPUT_FILE_NAME)


def main():
    files = csv_data.get_all_files(csv_files)
    max_heights_data = [["Temp [c]", "Max Height [mm]"]]
    for file in files[0:1]:
        print(file)
        drop = droplet.Droplet(identifier=file["metadata"]["droplet"],
                                  temp=file["metadata"]["temperature"],
                                  temp_serial=file["metadata"]["temperature_serial"],
                                  raw_data_df=file["data"])
        drop.normalize_data()
        drop.get_max_height()
        max_heights_data.append([drop.temperature, drop.max_height])
        yaxis = drop.raw_data_df["y_norm"]
        xaxis = drop.raw_data_df["t_norm"]
        plt.plot(xaxis, yaxis) #, marker="o", markersize=1, linestyle=" ")
        #plt.plot(xaxis, yaxis, marker="o", markerfacecolor="red", linestyle=" ")
        plt.show()

    #csv_data.write_to_csv(output_path, max_heights_data)


    print(f"successfully wrote data to {output_path}")
    return


if __name__ == "__main__":
    main()





