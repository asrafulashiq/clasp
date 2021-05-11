""" Class to communicate with the wrapper sacript """

import yaml
import os
from loguru import logger
import time


class YAMLCommunicator(object):
    def __init__(self, runtime_file, rpi_flagfile, neu_flagfile,
                 mu_flagfile) -> None:
        super().__init__()
        self.runfile = os.path.expanduser(runtime_file)
        self.rpi_flagfile = os.path.expanduser(rpi_flagfile)
        self.neu_flagfile = os.path.expanduser(neu_flagfile)
        self.mu_flagfile = os.path.expanduser(mu_flagfile)

        assert os.path.exists(self.runfile)
        data = {"Batch_Processed": "FALSE", "Bin_Processed": "FALSE"}
        self.set_flag_init(data, self.rpi_flagfile)

    def is_association_ready(self):
        return self._get_flag('Association_Ready', self.neu_flagfile) == 'TRUE'

    def is_batch_ready(self):
        # check whether frame is ready for RPI
        return self._get_flag('Frames_Ready_RPI', self.runfile) == 'TRUE'

    def set_bin_processed(self, value='TRUE'):
        # tell the wrapper that batch is processed from RPI side
        self._set_flag("Bin_Processed", value, self.rpi_flagfile)

    # def set_batch_read(self):
    #     # tell the wrapper that batch is processed from RPI side
    #     self._set_flag('Frames_Ready_RPI', 'FALSE', self.runfile)

    def check_next_batch_ready(self):
        return self._get_flag("Next_Batch_Requested", self.runfile) == 'TRUE'

    def set_batch_processed(self, value='TRUE'):
        # This will be set to TRUE by Ashraful once
        # the combined output for current batch has been created.
        self._set_flag('Batch_Processed', value, self.rpi_flagfile)

    def is_end_of_frames(self):
        # return False
        # # check whether there are any frame left
        return self._get_flag('No_More_Frames', self.runfile) == 'TRUE'

    def check_people_processed_ready(self):
        return self._get_flag('People_Processed', self.mu_flagfile) == 'TRUE'

    def _get_flag(self, flag_name, flagfile):
        value = None
        with open(flagfile, 'r') as fp:
            try:
                data = yaml.full_load(fp)
                if data is None:
                    logger.warning("flag is None")
                    return self._get_flag(flag_name, flagfile)
                value = data[flag_name]
            except KeyError:
                logger.warning(f"No flag {flag_name}!!")
        return value

    def _set_flag(self, flag_name, value, filename):
        try:
            with open(filename, 'r') as fp:
                data = yaml.full_load(fp)
            data[flag_name] = value
            with open(filename, 'w') as fp:
                yaml.dump(data, fp)
            return value
        except:
            # catch any error here
            time.sleep(0.01)  # wait for 0.5 sec
            logger.error("Reading error while setting file!!")
            self._set_flag(flag_name, value, filename)

    def set_flag_init(self, data, filename):
        try:
            with open(filename, 'w') as fp:
                yaml.dump(data, fp)
        except:
            self.set_flag_init(data, filename)