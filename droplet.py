import pandas as pd
import matplotlib.pyplot as plt


class Droplet:
    def __init__(self, identifier, temp, temp_serial, raw_data_df):
        self.identifier = identifier
        self.temperature = temp
        self.temperature_serial = temp_serial
        self.raw_data_df = raw_data_df

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

    def get_peaks(self, peak_width=2):
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
        plt.plot(xaxis, yaxis)
        peaksy = self.peaks_df["peaks_y"]
        peakst = self.peaks_df["peaks_t"]
        plt.plot(peakst, peaksy, marker="o", markerfacecolor="red", linestyle=" ")
        plt.show()
        return





