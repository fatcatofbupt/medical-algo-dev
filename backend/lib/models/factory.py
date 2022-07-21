
import logging
from typing import Callable

logger = logging.getLogger(__name__)

class ModelFactory:
    """ The factory class for creating Type"""

    registry = {}
    params = {}
    """ Internal registry for available Type """
    @classmethod
    def show_models(cls):
        return cls.params

    @classmethod
    def register(cls, name: str) -> Callable:
        """ Class method to register Executor class to the internal registry.
        Args:
            name (str): The name of the Type.
        Returns:
            The Type class itself.
        """

        def inner_wrapper(wrapped_class) -> Callable:
            if name in cls.registry:
                logger.warning('%s already exists. Will replace it', name)
            cls.registry[name] = wrapped_class
            cls.params[name] = wrapped_class.params
            return wrapped_class

        return inner_wrapper

    # end register()

    @classmethod
    def create(cls, name: str, **kwargs):
        """ Factory command to create the Type.
        This method gets the appropriate Type class from the registry
        and creates an instance of it, while passing in the parameters
        given in ``kwargs``.
        Args:
            name (str): The name of the Type to create.
        Returns:
            An instance of the Type that is created.
        """

        if name not in cls.registry:
            logger.warning('%s does not exist in the registry', name)
            return None

        class_type = cls.registry[name]
        ins = class_type(**kwargs)
        return ins
