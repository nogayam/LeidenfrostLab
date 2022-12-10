import csv_data, droplet, config, glob, os
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

import pandas as pd


analysis_modules = config.ANALYSIS_MODULES
folder_path = config.CSV_FOLDER
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
output_path = config.OUTPUT_PATH.format(name=config.OUTPUT_FILE_NAME)


def main():
    files = csv_data.get_all_files(csv_files)
    max_heights_data = [["Temp [c]", "Max Height [mm]"]]
    df_temps = {temp: pd.DataFrame() for temp in config.TEMP}

    for file in files[:]:
        drop = droplet.Droplet(identifier=file["metadata"]["droplet"],
                                  temp=file["metadata"]["temperature"],
                                  temp_serial=file["metadata"]["temperature_serial"],
                                  raw_data_df=file["data"])
        drop.normalize_data()
        drop.get_max_height()
        if drop.temperature not in ["295", "305"]:
            max_heights_data.append([drop.temperature, drop.max_height])
        drop.get_peaks(peak_width=3) #3 to avoid jitter
        #drop.get_no_peaks()
        #drop.plot_y_with_peaks()
        #drop.remove_peaks()
        #drop.plot_no_peaks()

        #print(drop.peaks_df["peaks_y"])
        peaksy = drop.peaks_df.iloc[drop.peaks_df["peaks_y"].nlargest(1).index]["peaks_y"]
        peakst = drop.peaks_df.iloc[drop.peaks_df["peaks_y"].nlargest(1).index]["peaks_t"]

        df_temps[str(drop.temperature)] = pd.concat([df_temps[str(drop.temperature)], drop.peaks_df.iloc[drop.peaks_df["peaks_y"].index]])
    #max_heights = [lis[1] for lis in max_heights_data[1:]]
    #max_temp = [int(lis[0]) for lis in max_heights_data[1:]]
    #max_heights_df = pd.DataFrame(data={"temp": max_temp, "max_height": max_heights})
    #plt.hist2d(max_heights_df["temp"], max_heights_df["max_height"], bins=10)
    #plt.show()

    fig, axs = plt.subplots(4, 4)
    #plt.suptitle('Rolling Avg')
    for temp in df_temps.keys():
        if temp in ["295", "305"]:
            continue
        t_index = config.TEMP.index(temp)
        current_axs = axs[int(t_index / 4), t_index % 4]
        current_axs.set_title(f"T {temp} [C]")
        current_axs.hist(df_temps[temp]["peaks_y"], bins=20, density=False, range=(0, 20))
        current_axs.set_xlabel('time [s]')
        current_axs.set_ylabel('height [mm]')
        current_axs.set_ylim([0, 40])
        #current_axs.hist2d(df_temps[temp]["peaks_t"], df_temps[temp]["peaks_y"], bins=10)
        df_temps[temp] = df_temps[temp].sort_values(by=["peaks_t"])
        #current_axs.plot(df_temps[temp]["peaks_t"], df_temps[temp]["peaks_y"], marker="o", markersize=1.5, linestyle=" ")
        #line, = plt.plot(df_temps[temp]["peaks_t"], df_temps[temp]["peaks_y"].rolling(5).mean(), marker="o", markersize=1, markerfacecolor="red", linestyle="-")
        #line.set_label(f"T {temp} [C] - Mean {round(df_temps[temp]['peaks_y'].mean(),2)} [mm]")

    plt.legend()
    plt.show()

    #csv_data.write_to_csv(output_path, max_heights_data)


    #print(f"successfully wrote data to {output_path}")
    return


if __name__ == "__main__":
    main()





