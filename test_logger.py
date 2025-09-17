from obs_system.utils.logger import get_logger 

logger = get_logger(__name__)

print(logger)
print(logger.level)
logger.info("THEEEE MOY OLYMPIAKE MOY") 
logger.info(f"{logger.level}")
logger.debug("NewINFOFNO")
