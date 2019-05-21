class GymException(Exception):
    def __init__(self, *args, **kwargs):
        super(GymException, self).__init__(args, kwargs)