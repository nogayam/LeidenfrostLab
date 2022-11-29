
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

    def get_peaks(self):
        pass




