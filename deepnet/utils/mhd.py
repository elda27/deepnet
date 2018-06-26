import re
import numpy as np
import zlib
import os

_type_table = {
    'MET_CHAR':'i1',
    'MET_UCHAR':'u1',
    'MET_SHORT':'i2',
    'MET_USHORT':'u2',
    'MET_INT':'i4',
    'MET_UINT':'u4',
    'MET_LONG':'i8',
    'MET_ULONG':'u8',
    'MET_FLOAT':'f4',
    'MET_DOUBLE':'f8'
    }
_dtype_table = {
    'int8':'MET_CHAR',
    'uint8':'MET_UCHAR',
    'int16':'MET_SHORT',
    'uint16':'MET_USHORT',
    'int32':'MET_INT',
    'uint32':'MET_UINT',
    'int64':'MET_LONG',
    'uint64':'MET_ULONG',
    'float32':'MET_FLOAT',
    'float64':'MET_DOUBLE'
    }
def _str2array(string):
    try:
        return list(map(int, string.split()))
    except:
        try:
            return list(map(float, string.split()))
        except:
            return string
def _array2str(array):
    if isinstance(array, str):
        return array
    elif isinstance(array, int):
        return str(array)
    else:
        return ' '.join(list(map(str, array)))

def read(filename):
    header, f = read_header(filename)
    image = read_data(filename, header, f)
    return image, header

def read_header(filename):
    header = {}
    with open(filename, 'rb') as f:
        meta_regex = re.compile('(\w+) = (.+)')
        for line in f:
            line = line.decode('ascii')
            match = meta_regex.match(line)
            if match:
                header[match.group(1)] = match.group(2).rstrip('\r')
                if match.group(1) == 'ElementDataFile':
                    break;
            else:
                raise RuntimeError('Bad meta header')

    header = {key:_str2array(value) for (key, value) in header.items()} #convert string into array if possible
    return header, f

def read_data(filename, header, f):
    data_filename = header['ElementDataFile']
    if data_filename == 'LOCAL': #mha
        data = f.read()
    else: #mhd
        with open(os.path.join(os.path.dirname(filename),data_filename), 'rb') as fimage:
            data = fimage.read()
    if ('CompressedData' in header) and header['CompressedData'] == 'True': #Decompress data
        data = zlib.decompressobj().decompress(data)
    data = np.frombuffer(data,dtype=np.dtype(_type_table[header['ElementType']]))
    dim = header['DimSize']
    image = np.reshape(data,list(reversed(dim)),order='C')
    return image



_default_header = {
    'ObjectType':'Image',
    'NDims':'3',
    'BinaryData':'True',
    'BinaryDataByteOrderMSB':'False',
    #'TransformMatrix':'1 0 0 0 1 0 0 0 1',
    #'Offset':'0 0 0',
    #'CenterOfRotation':'0 0 0',
    #'AnatomicalOrientation':'???',
    'CompressedData':'True'
        }

def write(filename, image, header={}):
    h = _default_header
    h['ElementSpacing'] = np.ones(image.ndim) #default spacing
    h.update(header)
    h['NDims'] = len(image.shape)
    h['ElementType'] = _dtype_table[image.dtype.name]
    h['DimSize'] = reversed(image.shape)
    h = {key:_array2str(value) for (key, value) in h.items()} #convert array into string if possible
    filename_base, file_extension = os.path.splitext(os.path.basename(filename))
    dirname = os.path.dirname(filename)
    compress_data = h['CompressedData']=='True'
    h.pop('ElementDataFile',None) #delete
    if not compress_data and 'CompressedData' in h:
        del h['CompressedData']
    if (file_extension == '.mhd'):
        if (compress_data):
            data_filename = filename_base + '.zraw'
        else:
            data_filename = filename_base + '.raw'
    else:
        data_filename = 'LOCAL'
    data = image.tobytes()
    with open(filename, 'w') as f:
        f.write('ObjectType = '+h.pop('ObjectType')+'\n')
        f.write('NDims = '+h.pop('NDims')+'\n')
        if (compress_data):
            data = zlib.compress(data)
            h['CompressedDataSize'] = str(len(data))
        for key, value in h.items():
            f.write(key+' = '+value+'\n')
        f.write('ElementDataFile = '+data_filename+'\n')
        if data_filename == 'LOCAL':
            # reopen file in binary mode
            f.close()
            f = open(filename, 'ab')
            f.write(data)
        else:
            with open(os.path.join(dirname, data_filename), 'wb') as fdata:
                fdata.write(data)
