import logging 
import datetime 
import os, sys


def remove_logger(name=None): 
    logr = logging.getLogger(name) 
    logr.propagate = True 
    logr.handlers.clear()
    logr.setLevel(logging.WARNING)


def jupyter_logger(level=logging.INFO)->logging.StreamHandler: 
    jupyter_handler = logging.StreamHandler(sys.stdout)
    jupyter_handler.setLevel(level)

    jupyter_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ) 
    jupyter_handler.setFormatter(jupyter_formatter)

    return jupyter_handler


def setup_logging(level=logging.INFO, log_dir="assets/logs"):
    root = logging.getLogger("obs_system")
    root.setLevel(level)
    # if root.handlers:  # idempotent
    #     root.setLevel(level)
    #     for h in root.handlers:
    #         h.setLevel(level)
    #     

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log_{datetime.datetime.now():%Y%m%d_%H%M%S}.log")

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console (stdout)
    # sh = logging.StreamHandler(sys.stdout)
    # sh.setLevel(level)
    # sh.setFormatter(fmt)

    # File
    fh = logging.FileHandler(log_file, encoding="utf-8", mode="w")
    fh.setLevel(level)
    fh.setFormatter(fmt)

    root.setLevel(level)
    # root.addHandler(sh)
    root.addHandler(fh)

    jupyter_handler = jupyter_logger(level=level)

    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root.handlers):
        root.addHandler(jupyter_handler) 

    root.propagate = True


def get_logger(name=None):
    return logging.getLogger(name)


def list_loggers(show_handlers=False):
    reg = logging.Logger.manager.loggerDict  # name -> Logger or PlaceHolder
    rows = []
    for name, obj in reg.items():
        if not isinstance(obj, logging.Logger):
            continue
        level = logging.getLevelName(obj.level) if obj.level else "NOTSET"
        eff   = logging.getLevelName(obj.getEffectiveLevel())
        hs    = [type(h).__name__ for h in obj.handlers]
        rows.append((name, level, eff, obj.propagate, hs))
    rows.sort()
    for name, level, eff, prop, hs in rows:
        line = f"{name:40} level={level:7} effective={eff:7} propagate={prop}"
        if show_handlers:
            line += f" handlers={hs}"
        print(line)

    for n in logging.Logger.manager.loggerDict:
        if n.startswith(("matplotlib", "urllib3", "botocore")):
            print(n)

# usage

# remove_logger("matplotlib")
# remove_logger("matplotlib.font_manager")
setup_logging(logging.INFO)

