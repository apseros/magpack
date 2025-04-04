import numpy as np
import re
import struct
import logging


def _get_binary(flag, binary: int):
    """Reads binary information from a file."""
    if binary == 4:
        if struct.unpack('>f', flag)[0] == 1234567.0:
            dtype = '>f4'
            dtype_str = '>f'
        elif struct.unpack('<f', flag)[0] == 1234567.0:
            dtype = '<f4'
            dtype_str = '<f'
        else:
            raise ValueError('Cannot determine binary 4 data dtype')

    elif binary == 8:
        if struct.unpack('>d', flag)[0] == 123456789012345.0:
            dtype = '>f8'
            dtype_str = '>d'
        elif struct.unpack('<d', flag)[0] == 123456789012345.0:
            dtype = '<f8'
            dtype_str = '<d'
        else:
            raise ValueError('Cannot determine binary 8 data dtype')

    else:
        raise ValueError('Unsupported binary data size')

    return dtype, dtype_str


class OVF:
    """Class for reading OVF files.

    Attributes
    ----------
    filename : str
        The filename of the OVF file.
    properties : dict
        The properties found in the metadata of the OVF file.
    magnetization : np.ndarray
        The magnetization vector field.
    """
    @classmethod
    def _validate(cls, filename):
        try:
            open(filename, 'rb')
        except FileNotFoundError:
            logging.warning("No such file or directory.")
            return False
        return True

    def __new__(cls, filename):
        if cls._validate(filename):
            return super().__new__(cls)

    def __init__(self, filename):
        """
        Parameters
        ----------
        filename : str
            The filename of the OVF file.
        """
        # Path to the ovf file.
        self.filename: str = filename
        # Dictionary with OVF metadata.
        self.properties: dict = {}
        # Magnetization read from the OVF file provided as a numpy array with shape (3, nx, ny, nz):
        # (magnetization component, spatial x dimension, spatial y dimension, spatial z dimension).
        self.magnetization: np.ndarray
        self._get_data()

    def _get_data(self):
        """Loads metadata and magnetization binary from OVF file."""
        pattern = r'#\s(?!Begin|End)([\w]+): (\d.?\d*[eE][+-]\d+]?|\d+)'  # matches header metadata
        try:
            file = open(self.filename, 'rb')
        except FileNotFoundError:
            logging.warning("No such file or directory.")
        else:
            with file:
                for line in file:
                    try:
                        decoded_line = line.decode()[:-1]  # remove newline character
                    except UnicodeDecodeError:
                        logging.warning("Can't read line")
                    logging.debug(decoded_line)
                    content = re.findall(pattern, decoded_line)
                    if content:
                        name, value = content[0]
                        self.properties[name] = float(value)
                    elif decoded_line.startswith('# Begin: Data Binary '):
                        binary = int(decoded_line[-1])
                        break

                flag = file.read(binary)
                dtype, dtype_str = _get_binary(flag, binary)

                nx, ny, nz = map(int, [self.properties['xnodes'], self.properties['ynodes'], self.properties['znodes']])
                n_sites = nx * ny * nz
                data = np.fromfile(file, dtype=dtype)
                self.magnetization = np.array(
                    [data[i:3 * n_sites:3].reshape((nx, ny, nz), order='F') for i in range(3)])
