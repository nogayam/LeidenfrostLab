import pandas as pd
import matplotlib.pyplot as plt
import scipy


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
                x = self.raw_data_df["x"]
                self.raw_data_df["x"] = y * 1000

    def normalize_data(self):
        y = self.raw_data_df["y"]
        self.raw_data_df["y_norm"] = y - y.min()
        x = self.raw_data_df["x"]
        x0 = x[0]
        # To set first x value as x0=0.
        self.raw_data_df["x_norm"] = -1 * (x - x0)
        self.raw_data_df["x_norm"][0] = 0 #it puts 0 with minus
        #The general direction of movement is to the negative size of the x axis thats why I multiplied by -1
        t = self.raw_data_df["t"]
        t0 = t[0]
        self.raw_data_df["t_norm"] = t - t0
        return

    def get_max_height(self):
        y = self.raw_data_df["y_norm"]
        self.max_height = y.max()
        return

    def get_peaks(self, peak_width=3):
        peaks_y = []
        peaks_t = []
        last_y = None
        counter = 0
        potential_peak = None
        descending = False
        for index, row in self.raw_data_df.iterrows():
            t = row["t_norm"]
            y = row["y_norm"]

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
                        potential_peak = {"y": y, "t": t}
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
                        potential_peak = None

            last_y = y

        self.peaks_df = pd.DataFrame(data={"peaks_y": peaks_y, "peaks_t": peaks_t})
        return

    def plot_y_with_peaks(self):
        yaxis = self.raw_data_df["y_norm"]
        xaxis = self.raw_data_df["t_norm"]
        plt.plot(xaxis, yaxis) #, marker="x", markersize=2)
        peaksy = self.peaks_df["peaks_y"]
        peakst = self.peaks_df["peaks_t"]
        plt.plot(peakst, peaksy, marker="o", markerfacecolor="red", linestyle=" ")
        plt.show()
        return

    def get_no_peaks(self):
        df_all = self.raw_data_df.merge(self.peaks_df.drop_duplicates(), left_on=['y_norm', 't_norm'],
                                        right_on=['peaks_y', 'peaks_t'], how='left', indicator=True)
        no_peaks_df = df_all[df_all['_merge'] == 'left_only']
        self.no_peaks_df = no_peaks_df
        return

    def plot_no_peaks(self):
        yaxis = self.no_peaks_df["y_norm"]
        xaxis = self.no_peaks_df["t_norm"]
        plt.plot(xaxis, yaxis) #, marker="x", markersize=2)
        plt.show()
        return

    def get_peaks_histogram(self, maxvals=10):
        hist_df = {self.temperature: pd.DataFrame()}
        hist_df = [hist_df[str(self.temperature)], self.peaks_df.iloc[self.peaks_df["peaks_y"].nlargest(maxvals).index]]
        self.hist_df = hist_df
        return

    #def get_rolling_avg(self, rolling=20):
     #   line, = plt.plot(df_temps[temp]["peaks_t"], df_temps[temp]["peaks_y"].rolling(5).mean(), marker="o", markersize=1, markerfacecolor="red", linestyle="-")

    def remove_peaks(self, peaks_width=3):
        indexes_to_remove = []
        cond = self.raw_data_df["y_norm"].isin(self.peaks_df[["peaks_y"]])
        for ind in self.raw_data_df[cond].index:
            if ind in list(range(peaks_width)):
                first = 0
            else:
                first = ind - peaks_width

            if ind in range(len(self.raw_data_df) - peaks_width, len(self.raw_data_df)):
                last = len(self.raw_data_df)
            else:
                last = ind + peaks_width

            indexes_to_remove.append(range(first, last))

        self.no_peaks = self.raw_data_df[["y_norm", "t_norm"]]
        for ind_range in indexes_to_remove:
            self.no_peaks.drop([self.no_peaks.index[ind_range]])

        return

    def new_plot_no_peaks(self):
        yaxis = self.no_peaks["y_norm"]
        xaxis = self.no_peaks["t_norm"]
        plt.plot(xaxis, yaxis) #, marker="x", markersize=2)
        plt.plot(xaxis, yaxis)
        plt.show()
        return





