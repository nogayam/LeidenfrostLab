import csv_data, droplet, config, glob, os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

pd.options.mode.chained_assignment = None  # default='warn'

analysis_modules = config.ANALYSIS_MODULES
folder_path = config.CSV_FOLDER
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
output_path = config.OUTPUT_PATH.format(name=config.OUTPUT_FILE_NAME)


def main():
    files = csv_data.get_all_files(csv_files)
    max_heights_data = [["Temp [c]", "Max Height [mm]"]]
    df_temps = {temp: pd.DataFrame() for temp in config.TEMP}
    df_temps_oscillations = {temp: pd.DataFrame() for temp in config.TEMP}
    df_linear_fit = {temp: pd.DataFrame() for temp in config.TEMP}
    df_temp_stdv = {temp: [] for temp in config.TEMP}
    df_peaks_amount = {temp: [] for temp in config.TEMP}
    normalization_dict = {temp: 0 for temp in config.TEMP}

    for file in files[:]:
        drop = droplet.Droplet(identifier=file["metadata"]["droplet"],
                                  temp=file["metadata"]["temperature"],
                                  temp_serial=file["metadata"]["temperature_serial"],
                                  raw_data_df=file["data"])
        drop.normalize_data()

        drop.validate()
        if not drop.validated:
            continue

        drop.get_max_height()
        if drop.temperature not in ['305', '306', '308']: #["295", "305"]:
            max_heights_data.append([drop.temperature, drop.max_height])

        drop.get_peaks(peak_width=1) #3 to avoid jitter
        drop.get_peak_times()
        drop.get_std()
        drop.get_peaks_amount()
        #drop.get_no_peaks()

        drop.get_oscillations_filter(2)
        drop.linear_fit_osc()
        drop.stdv_from_linear_fit()
        #drop.plot_y_with_osc_threshold()

        #drop.normalize_by_osc_linear_fit()
        #drop.plot_y_with_peaks()

        normalization_dict[str(drop.temperature)] += len(drop.raw_data_df)
        peaksy = drop.peaks_df.iloc[drop.peaks_df["peaks_y"].nlargest(1).index]["peaks_y"]
        peakst = drop.peaks_df.iloc[drop.peaks_df["peaks_y"].nlargest(1).index]["peaks_t"]
        df_peaks_amount[str(drop.temperature)] = df_peaks_amount[str(drop.temperature)] + [drop.peaks_amount]

        df_height_linear_fit = pd.DataFrame(data={"height": drop.peaks_df["peaks_y"], "radius": drop.linear_osc_fit_fun(drop.peaks_df["peaks_t"])})

        df_linear_fit[str(drop.temperature)] = pd.concat([df_linear_fit[str(drop.temperature)], df_height_linear_fit])
        df_temps[str(drop.temperature)] = pd.concat([df_temps[str(drop.temperature)], drop.peaks_df.iloc[drop.peaks_df["peaks_y"].index]])
        df_temps_oscillations[str(drop.temperature)] = pd.concat([df_temps_oscillations[str(drop.temperature)],
                                                                  drop.osc_threshold_df.iloc[drop.osc_threshold_df["osc_y"].index]])
        df_temp_stdv[str(drop.temperature)] = df_temp_stdv[str(drop.temperature)] + [drop.osc_stdv_linear]
        #df_temps[str(drop.temperature)] = pd.concat([df_temps[str(drop.temperature)], drop.raw_data_df.iloc[drop.raw_data_df["y_norm"].index]])

#Plot: temp vs stdv (oscillation radius)
    temps = []
    temps_stdv = []
    for temp in df_temp_stdv.keys():
        temps += [int(temp)]
        temps_stdv += [np.average(df_temp_stdv[temp])]

    plt.plot(temps, temps_stdv)
    plt.suptitle(f'Temperature vs. Oscillation STDV')
    plt.ylabel('Oscillation STDV [mm]')
    plt.xlabel('Temperature [c]')
    plt.show()

        #df_temps[str(drop.temperature)] = pd.concat([df_temps[str(drop.temperature)], drop.peaks_df.iloc[drop.peaks_df["dt"].index]])
    #max_heights = [lis[1] for lis in max_heights_data[1:]]
    #max_temp = [int(lis[0]) for lis in max_heights_data[1:]]
    #max_heights_df = pd.DataFrame(data={"temp": max_temp, "max_height": max_heights})
    #plt.hist2d(max_heights_df["temp"], max_heights_df["max_height"], bins=10)
    #plt.show()

    print(df_peaks_amount)
# Plot: peaks amount vs temp
    temps = []
    temps_average_peaks_amount = []
    for temp in df_peaks_amount.keys():
        temps += [int(temp)]
        temps_average_peaks_amount += [np.average(df_peaks_amount[temp])]

    plt.plot(temps, temps_average_peaks_amount)
    plt.suptitle(f'Temperature vs. Average Peaks Amount')
    plt.ylabel('Average Peaks Amount')
    plt.xlabel('Temperature [c]')
    plt.show()


#Plot: height vs. radius
    fig, axs = plt.subplots(4, 5)
    avg_weighted_x = []
    avg_weighted_y = []
    for temp in df_temps.keys():
        if temp in ['305', '306', '308']:  # "295", "305"]:
            continue
        t_index = config.TEMP_2.index(temp)
        #if [int(t_index / 5), t_index % 5] == [4, 2]:
            #current_axs = axs[3, 3]
        #else:
        current_axs = axs[int(t_index / 5), t_index % 5]
        current_axs.set_title(f"T {temp} [C]")
        if df_temps[temp].empty:
            print(f'{temp} has no peaks!')
            continue

        current_axs.set_ylabel('Height [mm]')
        current_axs.set_xlabel('Radius [mm]')

        height = df_linear_fit[temp]["height"]
        radius = df_linear_fit[temp]["radius"]

        avg_weighted_radius = np.sum(height * radius) / np.sum(height)

        avg_weighted_y += [avg_weighted_radius]
        avg_weighted_x += [int(temp)]

        current_axs.plot(radius, height, marker='o', markersize=1.5, linestyle=' ')
        current_axs.annotate(f"Avg Radius: {round(avg_weighted_radius, 4)}", xy=(1, 1), xycoords='axes fraction',
                     xytext=(-5, -5), textcoords='offset points',
                     ha='right', va='top')
        current_axs.plot([avg_weighted_radius, avg_weighted_radius], [0, 30])

    plt.legend()
    plt.show()

    plt.plot(avg_weighted_x, avg_weighted_y)
    plt.suptitle('Height-Radius weighted average')
    plt.show()


# Plot: time vs. peaks
    fig, axs = plt.subplots(4, 5)
    avg_weighted_y = []
    avg_weighted_x = []
    for temp in df_temps.keys():
        if temp in ['305', '306', '308']: #"295", "305"]:
            continue
        t_index = config.TEMP_2.index(temp)
        #if [int(t_index / 5), t_index % 5] == [4, 2]:
         #   current_axs = axs[3, 3]
        #else:
        current_axs = axs[int(t_index / 5), t_index % 5]


        current_axs.set_title(f"T {temp} [C]")
        if df_temps[temp].empty:
            print(f'{temp} has no peaks!')
            continue

        current_axs.set_xlabel('Time [s]')
        current_axs.set_ylabel('Height [mm]')

        t = df_temps[temp]['peaks_t']
        y = df_temps[temp]['peaks_y']

        avg_weighted_t = np.sum(t * y) / np.sum(y)

        avg_weighted_y += [avg_weighted_t]
        avg_weighted_x += [int(temp)]

        current_axs.plot(t, y, marker='o', markersize=0.7, linestyle=' ')
        current_axs.annotate(f"Avg Time: {round(avg_weighted_t, 4)}", xy=(1, 1), xycoords='axes fraction',
                     xytext=(-5, -5), textcoords='offset points',
                     ha='right', va='top')
        current_axs.plot([avg_weighted_t, avg_weighted_t], [0, 30])
    plt.legend()
    plt.show()

    plt.plot(avg_weighted_x, avg_weighted_y)
    plt.suptitle('Peaks-Time weighted average')
    plt.show()

#Plot: Peaks Height Histogram
    fig, axs = plt.subplots(4, 5)
    for temp in df_temps.keys():
        if temp in ['305', '306', '308']: #"295", "305"]:
            continue
        t_index = config.TEMP_2.index(temp)
        #if [int(t_index / 5), t_index % 5] == [4, 2]:
         #   current_axs = axs[3, 3]
        #else:0
        current_axs = axs[int(t_index / 5), t_index % 5]


        current_axs.set_title(f"T {temp} [C]")
        if df_temps[temp].empty:
            print(f'{temp} has no peaks!')
            continue

        current_axs.hist(df_temps[temp]["peaks_y"], bins=20, density=True, range=(0, 20)) #1
        total_events = np.sum(np.histogram(df_temps[temp]["peaks_y"], bins=20)[0])
        current_axs.annotate("Total events: {}".format(total_events), xy=(1, 1), xycoords='axes fraction',
                     xytext=(-5, -5), textcoords='offset points',
                     ha='right', va='top')


        #current_axs.hist(df_temps[temp]["y_norm"], bins=60, density=False, range=(0, 20))
        #current_axs.hist(df_temps[temp]["dt"], bins=40, density=False, range=(0, 0.4)) #0.02
        current_axs.set_xlabel('Height [mm]')
        current_axs.set_ylabel('Events')
        #current_axs.set_ylim([0, 80])
        #current_axs.hist2d(df_temps[temp]["peaks_t"], df_temps[temp]["peaks_y"], bins=10)

        #df_temps[temp] = df_temps[temp].sort_values(by=["peaks_t"])
        #df_temps[temp] = df_temps[temp].sort_values(by=["dt"])
        #current_axs.plot(df_temps[temp]["peaks_t"], df_temps[temp]["peaks_y"].rolling(20).mean(), marker="o", markersize=1.5, linestyle="-")
        #current_axs.set_label(f"T {temp} [C] - Mean {round(df_temps[temp]['peaks_y'].mean(), 2)} [mm]")
        #current_axs.set_xlabel('Time [s]')
        #current_axs.set_ylabel('Height [mm]')

        #line, = plt.plot(df_temps[temp]["peaks_t"], df_temps[temp]["peaks_y"].rolling(20).mean(), marker="o", markersize=1, markerfacecolor="red", linestyle="-")
        #line.set_label(f"T {temp} [C] - Mean {round(df_temps[temp]['peaks_y'].mean(),2)} [mm]")

    plt.legend()
    plt.show()

# Plot: Oscillation Height Histogram
    fig, axs = plt.subplots(4, 5)
    for temp in df_temps_oscillations.keys():
        if temp in ['305', '306', '308']:  # "295", "305"]:
            continue
        t_index = config.TEMP_2.index(temp)
        # if [int(t_index / 5), t_index % 5] == [4, 2]:
        #   current_axs = axs[3, 3]
        # else:0
        current_axs = axs[int(t_index / 5), t_index % 5]

        current_axs.set_title(f"T {temp} [C]")
        if df_temps_oscillations[temp].empty:
            print(f'{temp} has no peaks!')
            continue

        current_axs.hist(df_temps_oscillations[temp]["osc_y"], bins=30, density=True, range=(0, 2))  # 1
        total_events = np.sum(np.histogram(df_temps_oscillations[temp]["osc_y"], bins=20)[0])
        current_axs.annotate("Total events: {}".format(total_events), xy=(1, 1), xycoords='axes fraction',
                             xytext=(-5, -5), textcoords='offset points',
                             ha='right', va='top')

        current_axs.set_xlabel('Height [mm]')
        current_axs.set_ylabel('Events')

    plt.legend()
    #plt.show()

# Plot: Peaks Time Histogram
    fig, axs = plt.subplots(4, 5)
    for temp in df_temps.keys():
        if temp in ['305', '306', '308']:  # "295", "305"]:
            continue
        t_index = config.TEMP_2.index(temp)
        current_axs = axs[int(t_index / 5), t_index % 5]

        current_axs.set_title(f"T {temp} [C]")
        if df_temps[temp].empty:
            print(f'{temp} has no peaks!')
            continue

        current_axs.hist(df_temps[temp]["dt"], bins=30, density=False, range=(0, 0.4)) #0.02
        total_events = np.sum(np.histogram(df_temps[temp]["dt"], bins=30)[0])
        current_axs.annotate("Total events: {}".format(total_events), xy=(1, 1), xycoords='axes fraction',
                             xytext=(-5, -5), textcoords='offset points',
                             ha='right', va='top')
        current_axs.set_xlabel('Time [s]')
        current_axs.set_ylabel('Events')
    plt.legend()
    plt.show()
    #csv_data.write_to_csv(output_path, max_heights_data)


    #print(f"successfully wrote data to {output_path}")
    return


if __name__ == "__main__":
    main()





