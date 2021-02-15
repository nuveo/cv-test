import logging, os, sys, time
import logging.config
import functools, yaml

@functools.lru_cache()
def setup_logger(
    output=None, color=True, name="spam_filter" ):
    """
    Inicializa logger
    Args:
        output ([type], optional): [description]. Defaults to None.
        color (bool, optional): [description]. Defaults to True.
    Returns:
        loggin.Logger: a logger
    """

    with open('logger_config.yaml', 'r') as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    
    # create a custom logger
    logger = logging.getLogger(name)
    
    return logger