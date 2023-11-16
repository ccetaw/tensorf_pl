import os
import torch
import imageio 
import json
from icecream import ic

import numpy as np

class Logger:
    """
    Class Logger. An integration of tensorboard writer and icecream debug print, as well as other ostream utilities. 
    This class is to be declared globally. You must call set_logdir() method to set the log directory.
    """

    def __init__(self, debug) -> None:
        """
        Input:
        - debug: bool. 
        """
        # Init parameters
        self.logdir = '.'
        self.debug = debug
        
        # ic debug_print
        self.debug_print = ic
        self.debug_print.configureOutput(prefix="Debug | ", includeContext=True)
        if debug:
            self.debug_print.enable()
        else:
            self.debug_print.disable()
        
    def info_print(self, info):
        print(f"- {info}")

    def set_logdir(self, basedir):
        """
        Set the log directory.
        ----
        Input:
        - basedir: str. Path to directory. 
        """
        self.logdir = basedir
        os.makedirs(self.logdir, exist_ok=True)

    def set_mode(self, debug=True):
        """
        Enable printing debug information. 
        """
        if debug:
            self.debug_print.enable()
        else:
            self.debug_print.disable()

    def _prepare_dir(self, path):
        """
        Make directory if path contains a dir.
        """
        prefix = os.path.dirname(path)
        if prefix != '':
            os.makedirs(os.path.join(self.logdir, prefix), exist_ok=True)
        write_path = os.path.join(self.logdir, path)
        
        return write_path

    def save_ckpt(self, path, ckpt):
        write_path = self._prepare_dir(path)
        torch.save(ckpt, write_path)

        if self.debug:
            info = f"Save checkpoint to {write_path}"
            self.debug_print(info)

    def write_image(self, path, image):
        """
        Write a numpy array image to designated path. Convert it to uint8 in the first place.
        """
        write_path = self._prepare_dir(path)
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)
        imageio.imwrite(write_path, image)

        if self.debug:
            info = f"Write image to {write_path}"
            self.debug_print(info)

    def write_dict2txt(self, path, dict_):
        write_path = self._prepare_dir(path)
        try:
            dict_.pop('config')
        except:
            pass

        try:
            dict_.pop('train')
        except:
            pass

        with open(write_path, 'w') as f:
            for key, value in dict_.items(): 
                f.write('%s = %s\n' % (key, value))

        if self.debug:
            info = f"Write txt file to {write_path}"
            self.debug_print(info)

    def write_dict2json(self, path, dict_):
        write_path = self._prepare_dir(path)
        with open(write_path, 'w') as f:
            f.write(json.dumps(dict_, indent=4))

        if self.debug:
            info = f"Write json file to {write_path}"
            self.debug_print(info)

logger = Logger(debug=True)

if __name__ == "__main__":
    import numpy as np
    test_image = (np.random.random(size=(800, 800, 3)) * 255).astype('uint8')
    test_dict = {'Name' : "Alice",
                 'Age' : 21,
                 'Degree' : "Bachelor Cse",
                 'University' : "Northeastern Univ"}

    logdir = 'log'
    logger = Logger(debug=True)

    logger.debug_print(test_dict)
    logger.debug_print(test_image.shape)

    logger.write_dict2txt('test_dict.txt', test_dict)
    logger.write_dict2json('test_dict.json', test_dict)
    logger.write_image('imgs/test_image.png', test_image)

    logger.set_mode(debug=False)
    
    logger.debug_print(test_dict)
    logger.debug_print(test_image.shape)
    logger.write_image('imgs/test_image.png', test_image)
