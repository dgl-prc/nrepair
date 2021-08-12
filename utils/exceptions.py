class UnsupportedDataType(Exception):
    def __init__(self, dataset):
        echo_info = f"Unsupported Dataset:{dataset}"
        super(UnsupportedDataType, self).__init__(echo_info)
