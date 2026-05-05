from obs_system.application_module.dummy_application.dummy_app import Application
from obs_system.application_module.dummy_application.intermediary import gui_connector
from obs_system.application_module.dummy_application.pipeline_config import PipelineConfig, build_arg_parser
from obs_system.utils.logger import get_logger, remove_logger

logger = get_logger(name="obs_system."+__name__)
remove_logger("matplotlib") 
remove_logger("matplotlib.font_manager") 


def main():

    logger.debug("--- Initializing Application ---")
    argparser = build_arg_parser()
    
    config = PipelineConfig.from_namespace(argparser.parse_args())
 
    logger.warning("If you change the input video source, adjust the background subtractor image. Otherwise, it will classify all frames without movement")

    if config.gui:
        gui_connector(config.host_address, config.port_address)
        return

    config = config.validate()

    app = Application()
    app.run_application(config)

   
if __name__ == "__main__":
    main()
    

