from offworld_gym.utils.data_structures import UniqueDict

class Singleton(type):
    """
    A meta class to define another class as a singleton class
    """
    
    _instances = UniqueDict()
    
    def __call__(singleton_class, *args, **kwargs):
        singleton_class._instances[singleton_class] = super(Singleton, singleton_class).__call__(*args, **kwargs)
        return singleton_class._instances[singleton_class]