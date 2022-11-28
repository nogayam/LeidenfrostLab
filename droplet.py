
class Droplet:
    def __init__(self, identifier, temp, temp_serial, raw_data):
        self.identifier = identifier
        self.temperature = temp
        self.temperature_serial = temp_serial
        self.raw_data = raw_data

    def get_max_height(self):
        heights = [record["y"] for record in self.raw_data]
        self.max_height = max(heights)
        return