class UniqueDict(dict):
    """A dictionary with unique keys

    Duplicate keys will be ignored
    """

    def __setitem__(self, key, value):
        if key not in self.keys():
            dict.__setitem__(self, key, value)
        ## Ignore a duplicate key silently