import pandas as pd
import matplotlib.pyplot as plt
import scipy
import numpy as np


class Droplet:
    def __init__(self, identifier, temp, temp_serial, raw_data_df):
        self.identifier = identifier
        self.temperature = temp
        self.temperature_serial = temp_serial
        self.raw_data_df = raw_data_df

    def normalize_to_mm(self):
        if self.temperature == "305":
            if self.temperature_serial in list(range(2,7)):
                y = self.raw_data_df["y"]
                self.raw_data_df["y"] = y * 1000
                #x = self.raw_data_df["x"]
                #self.raw_data_df["x"] = y * 1000

    def normalize_data(self):
        y = self.raw_data_df["y"]
        self.raw_data_df["y_norm"] = y - y.min()
        #x = self.raw_data_df["x"]
        #x0 = x[0]
        # To set first x value as x0=0.
        #self.raw_data_df["x_norm"] = -1 * (x - x0)
        #self.raw_data_df["x_norm"][0] = 0 #it puts 0 with minus
        #The general direction of movement is to the negative size of the x axis thats why I multiplied by -1
        t = self.raw_data_df["t"]
        t0 = t[0]
        self.raw_data_df["t_norm"] = t - t0
        return

    def get_max_height(self):
        y = self.raw_data_df["y_norm"]
        self.max_height = y.max()
        return

    def get_peaks(self, peak_width=1, height_threshold=3):
        peaks_y = []
        peaks_t = []
        peaks_original_index = []
        last_y = None
        counter = 0
        potential_peak = None
        descending = False
        for index, row in self.raw_data_df.iterrows():
            t = row["t_norm"]
            y = row["y_norm"]
            original_index = index

            if last_y is None:
                last_y = y
                continue

            if y == last_y:
                counter = 0
            elif y > last_y:
                if descending:
                    descending = False
                    counter = 1

                else:
                    if counter < peak_width:
                        counter += 1

                    if counter == peak_width:
                        if y >= height_threshold:
                            potential_peak = {"y": y, "t": t, "peaks_original_index": original_index}
            else:
                if not descending:
                    descending = True
                    counter = peak_width - 1
                else:
                    if counter > 0:
                        counter -= 1

                    if counter == 0 and potential_peak:
                        peaks_y.append(potential_peak["y"])
                        peaks_t.append(potential_peak["t"])
                        peaks_original_index.append((potential_peak["peaks_original_index"]))
                        potential_peak = None

            last_y = y

        self.peaks_df = pd.DataFrame(data={"peaks_y": peaks_y, "peaks_t": peaks_t, "peaks_original_index": peaks_original_index})
        return

    def get_oscillations_filter(self, threshold=3):
        osc_y, osc_t = [], []
        for index, row in self.raw_data_df.iterrows():
            t = row["t_norm"]
            y = row["y_norm"]
            if y < threshold:
                osc_y.append(y)
                osc_t.append(t)
        self.osc_threshold_df = pd.DataFrame(data={"osc_y": osc_y, "osc_t": osc_t})
        self.osc_std = self.osc_threshold_df.std()["osc_y"]

    def linear_fit_osc(self):
        oscy = self.osc_threshold_df["osc_y"]
        osct = self.osc_threshold_df["osc_t"]
        coef = np.polyfit(osct, oscy, 1)
        self.linear_osc_fit_fun = np.poly1d(coef)
        self.osc_linear_fit = pd.DataFrame(data={"t": osct, "y": self.linear_osc_fit_fun(osct)})

    def stdv_from_linear_fit(self):
        oscy = self.osc_threshold_df["osc_y"]
        osct = self.osc_threshold_df["osc_t"]
        self.osc_stdv_linear = np.sqrt(np.sum((oscy - self.linear_osc_fit_fun(osct)) ** 2) / len(osct))

    def normalize_by_osc_linear_fit(self):
        y_norm = self.raw_data_df["y_norm"]
        t_norm = self.raw_data_df["t_norm"]
        self.raw_data_df["y_linear_norm"] = y_norm - self.linear_osc_fit_fun(t_norm);
        self.raw_data_df["y_linear_norm"] = self.raw_data_df["y_linear_norm"] - self.raw_data_df["y_linear_norm"].min()

    # Plots
    def plot_y_with_peaks(self):
        yaxis = self.raw_data_df["y_norm"]
        xaxis = self.raw_data_df["t_norm"]
        plt.plot(xaxis, yaxis, marker="x", markersize=2)
        plt.xlabel('Time [s]')
        plt.ylabel('Height [mm]')
        peaksy = self.peaks_df["peaks_y"]
        peakst = self.peaks_df["peaks_t"]
        plt.plot(peakst, peaksy, marker="o", markerfacecolor="red", linestyle=" ")
        plt.suptitle(f'{self.temperature}-{self.temperature_serial}-{self.identifier}')
        plt.show()
        return

    def plot_y_with_osc_threshold(self):
        yaxis = self.raw_data_df["y_norm"]
        xaxis = self.raw_data_df["t_norm"]
        plt.plot(xaxis, yaxis, marker="x", markersize=2)
        plt.xlabel('Time [s]')
        plt.ylabel('Height [mm]')
        ocsy = self.osc_threshold_df["osc_y"]
        ocst = self.osc_threshold_df["osc_t"]
        plt.plot(ocst, ocsy, marker="o", markerfacecolor="red", linestyle=" ")
        linear_fit_y = self.osc_linear_fit["y"]
        linear_fit_t = self.osc_linear_fit["t"]
        plt.plot(linear_fit_t, linear_fit_y, markerfacecolor="red", linestyle="-", color="purple")
        plt.suptitle(f'{self.temperature}-{self.temperature_serial}-{self.identifier}')
        plt.show()
        return

    def get_peak_times(self):
        if self.peaks_df.empty:
            self.peaks_df["dt"] = None
        else:
            self.peaks_df["dt"] = self.peaks_df["peaks_t"].diff(1)
            self.peaks_df["dt"][0] = self.peaks_df["peaks_t"][0]

        return

    def get_no_peaks(self):
        df = self.raw_data_df
        threshold = df['y_norm'].mean() + 3 * df['y_norm'].std()
        peaks = df[df['y_norm'] > threshold]
        for index, row in peaks.iterrows():
            df.at[index, 'y_norm'] = (df.at[index - 3, 'y_norm'] + df.at[index + 3, 'y_norm']) / 2

        plt.plot(df['t_norm'], df['y_norm'], label='Data with Peaks Removed')
        plt.legend()
        plt.show()

    def plot_no_peaks(self):
        yaxis = self.no_peaks_df["y_norm"]
        xaxis = self.no_peaks_df["t_norm"]
        plt.plot(xaxis, yaxis) #, marker="x", markersize=2)
        plt.show()
        return

    def get_std(self):
        self.std = self.raw_data_df["y_norm"].std()
        return

    def get_peaks_amount(self):
        self.peaks_amount = len(self.peaks_df)

    def validate(self):
        y = self.raw_data_df["y_norm"]
        t = self.raw_data_df["t_norm"]
        validated = False
        if t.max() >= 0.18 and y.max() <= 35:
            validated = True

        self.validated = validated




