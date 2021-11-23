from typing import Tuple, Optional
import numpy as np
from numpy import genfromtxt
from PIL import Image
from scipy.ndimage import sobel
import time
import os
from io import StringIO
import errno
from memory_profiler import profile
from sys import argv
from struct import *
import os
from struct import *



#########################################################################################################################
#CODIGO SEAM CARVING

WIDTH_FIRST = 'width-first'
HEIGHT_FIRST = 'height-first'
VALID_ORDERS = (WIDTH_FIRST, HEIGHT_FIRST)

FORWARD_ENERGY = 'forward'
BACKWARD_ENERGY = 'backward'
VALID_ENERGY_MODES = (FORWARD_ENERGY, BACKWARD_ENERGY)

DROP_MASK_ENERGY = 1e5
KEEP_MASK_ENERGY = 1e3


def _rgb2gray(rgb: np.ndarray) -> np.ndarray:
    coeffs = np.array([0.2125, 0.7154, 0.0721], dtype=np.float32)
    return (rgb @ coeffs).astype(rgb.dtype)


def _get_seam_mask(src: np.ndarray, seam: np.ndarray) -> np.ndarray:
    return ~np.eye(src.shape[1], dtype=np.bool_)[seam]


def _remove_seam_mask(src: np.ndarray, seam_mask: np.ndarray) -> np.ndarray:
    if src.ndim == 3:
        h, w, c = src.shape
        seam_mask = np.dstack([seam_mask] * c)
        dst = src[seam_mask].reshape((h, w - 1, c))
    else:
        h, w = src.shape
        dst = src[seam_mask].reshape((h, w - 1))
    return dst


def _remove_seam(src: np.ndarray, seam: np.ndarray) -> np.ndarray:
    seam_mask = _get_seam_mask(src, seam)
    dst = _remove_seam_mask(src, seam_mask)
    return dst


def _get_energy(gray: np.ndarray) -> np.ndarray:
    assert gray.ndim == 2

    gray = gray.astype(np.float32)
    grad_x = sobel(gray, axis=1)
    grad_y = sobel(gray, axis=0)
    energy = np.abs(grad_x) + np.abs(grad_y)
    return energy


def _get_backward_seam(energy: np.ndarray) -> np.ndarray:
    assert energy.size > 0 and energy.ndim == 2
    h, w = energy.shape
    cost = energy[0]
    parent = np.empty((h, w), dtype=np.int32)
    base_idx = np.arange(-1, w - 1, dtype=np.int32)

    for r in range(1, h):
        left_shift = np.hstack((cost[1:], np.inf))
        right_shift = np.hstack((np.inf, cost[:-1]))
        min_idx = np.argmin([right_shift, cost, left_shift],
                            axis=0) + base_idx
        parent[r] = min_idx
        cost = cost[min_idx] + energy[r]

    c = np.argmin(cost)
    seam = np.empty(h, dtype=np.int32)

    for r in range(h - 1, -1, -1):
        seam[r] = c
        c = parent[r, c]

    return seam


def _get_backward_seams(gray: np.ndarray, num_seams: int,
                        keep_mask: Optional[np.ndarray]) -> np.ndarray:
    h, w = gray.shape
    seams_mask = np.zeros((h, w), dtype=np.bool_)
    rows = np.arange(0, h, dtype=np.int32)
    idx_map = np.tile(np.arange(0, w, dtype=np.int32), h).reshape((h, w))
    energy = _get_energy(gray)
    for _ in range(num_seams):
        if keep_mask is not None:
            energy[keep_mask] += KEEP_MASK_ENERGY
        seam = _get_backward_seam(energy)
        seams_mask[rows, idx_map[rows, seam]] = True

        seam_mask = _get_seam_mask(gray, seam)
        gray = _remove_seam_mask(gray, seam_mask)
        idx_map = _remove_seam_mask(idx_map, seam_mask)
        if keep_mask is not None:
            keep_mask = _remove_seam_mask(keep_mask, seam_mask)

        # Only need to re-compute the energy in the bounding box of the seam
        _, cur_w = energy.shape
        lo = max(0, np.min(seam) - 1)
        hi = min(cur_w, np.max(seam) + 1)
        pad_lo = 1 if lo > 0 else 0
        pad_hi = 1 if hi < cur_w - 1 else 0
        mid_block = gray[:, lo - pad_lo:hi + pad_hi]
        _, mid_w = mid_block.shape
        mid_energy = _get_energy(mid_block)[:, pad_lo:mid_w - pad_hi]
        energy = np.hstack((energy[:, :lo], mid_energy, energy[:, hi + 1:]))

    return seams_mask


def _get_forward_seam(gray: np.ndarray,
                      keep_mask: Optional[np.ndarray]) -> np.ndarray:
    assert gray.size > 0 and gray.ndim == 2
    gray = gray.astype(np.float32)
    h, w = gray.shape

    top_row = gray[0]
    top_row_lshift = np.hstack((top_row[1:], top_row[-1]))
    top_row_rshift = np.hstack((top_row[0], top_row[:-1]))
    dp = np.abs(top_row_lshift - top_row_rshift)

    parent = np.zeros(gray.shape, dtype=np.int32)
    base_idx = np.arange(-1, w - 1, dtype=np.int32)

    for r in range(1, h):
        curr_row = gray[r]
        curr_lshift = np.hstack((curr_row[1:], curr_row[-1]))
        curr_rshift = np.hstack((curr_row[0], curr_row[:-1]))
        cost_top = np.abs(curr_lshift - curr_rshift)
        if keep_mask is not None:
            cost_top[keep_mask[r]] += KEEP_MASK_ENERGY

        prev_row = gray[r - 1]
        cost_left = cost_top + np.abs(prev_row - curr_rshift)
        cost_right = cost_top + np.abs(prev_row - curr_lshift)

        dp_left = np.hstack((np.inf, dp[:-1]))
        dp_right = np.hstack((dp[1:], np.inf))

        choices = np.vstack([cost_left + dp_left, cost_top + dp,
                             cost_right + dp_right])
        dp = np.min(choices, axis=0)
        parent[r] = np.argmin(choices, axis=0) + base_idx

    c = np.argmin(dp)

    seam = np.empty(h, dtype=np.int32)
    for r in range(h - 1, -1, -1):
        seam[r] = c
        c = parent[r, c]

    return seam


def _get_forward_seams(gray: np.ndarray, num_seams: int,
                       keep_mask: Optional[np.ndarray]) -> np.ndarray:
    h, w = gray.shape
    seams_mask = np.zeros((h, w), dtype=np.bool)
    rows = np.arange(0, h, dtype=np.int32)
    idx_map = np.tile(np.arange(0, w, dtype=np.int32), h).reshape((h, w))
    for _ in range(num_seams):
        seam = _get_forward_seam(gray, keep_mask)
        seams_mask[rows, idx_map[rows, seam]] = True
        seam_mask = _get_seam_mask(gray, seam)
        gray = _remove_seam_mask(gray, seam_mask)
        idx_map = _remove_seam_mask(idx_map, seam_mask)
        if keep_mask is not None:
            keep_mask = _remove_seam_mask(keep_mask, seam_mask)

    return seams_mask


def _get_seams(gray: np.ndarray, num_seams: int, energy_mode: str,
               keep_mask: Optional[np.ndarray]) -> np.ndarray:
    assert energy_mode in VALID_ENERGY_MODES
    if energy_mode == BACKWARD_ENERGY:
        return _get_backward_seams(gray, num_seams, keep_mask)
    else:
        return _get_forward_seams(gray, num_seams, keep_mask)


def _reduce_width(src: np.ndarray, delta_width: int, energy_mode: str,
                  keep_mask: Optional[np.ndarray]) -> np.ndarray:
    assert src.ndim in (2, 3) and delta_width >= 0
    if src.ndim == 2:
        gray = src
        src_h, src_w = src.shape
        dst_shape = (src_h, src_w - delta_width)
    else:
        gray = _rgb2gray(src)
        src_h, src_w, src_c = src.shape
        dst_shape = (src_h, src_w - delta_width, src_c)

    seams_mask = _get_seams(gray, delta_width, energy_mode, keep_mask)
    dst = src[~seams_mask].reshape(dst_shape)
    return dst


def _expand_width(src: np.ndarray, delta_width: int, energy_mode: str,
                  keep_mask: Optional[np.ndarray]) -> np.ndarray:
    assert src.ndim in (2, 3) and delta_width >= 0
    if src.ndim == 2:
        gray = src
        src_h, src_w = src.shape
        dst_shape = (src_h, src_w + delta_width)
    else:
        gray = _rgb2gray(src)
        src_h, src_w, src_c = src.shape
        dst_shape = (src_h, src_w + delta_width, src_c)

    seams_mask = _get_seams(gray, delta_width, energy_mode, keep_mask)
    dst = np.empty(dst_shape, dtype=np.uint8)

    for row in range(src_h):
        dst_col = 0
        for src_col in range(src_w):
            if seams_mask[row, src_col]:
                lo = max(0, src_col - 1)
                hi = src_col + 1
                dst[row, dst_col] = src[row, lo:hi].mean(axis=0)
                dst_col += 1
            dst[row, dst_col] = src[row, src_col]
            dst_col += 1
        assert dst_col == src_w + delta_width

    return dst


def _resize_width(src: np.ndarray, width: int, energy_mode: str,
                  keep_mask: Optional[np.ndarray]) -> np.ndarray:
    assert src.size > 0 and src.ndim in (2, 3)
    assert width > 0
    assert energy_mode in VALID_ENERGY_MODES

    src_w = src.shape[1]
    if src_w < width:
        dst = _expand_width(src, width - src_w, energy_mode, keep_mask)
    else:
        dst = _reduce_width(src, src_w - width, energy_mode, keep_mask)
    return dst


def _resize_height(src: np.ndarray, height: int, energy_mode: str,
                   keep_mask: Optional[np.ndarray]) -> np.ndarray:
    assert src.ndim in (2, 3) and height > 0
    if src.ndim == 3:
        src = _resize_width(src.transpose((1, 0, 2)), height, energy_mode,
                            keep_mask).transpose((1, 0, 2))
    else:
        src = _resize_width(src.T, height, energy_mode, keep_mask).T
    return src


def _check_mask(mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.bool)
    if mask.ndim != 2:
        raise ValueError('Invalid mask of shape {}: expected to be a 2D '
                         'binary map'.format(mask.shape))
    if mask.shape != shape:
        raise ValueError('The shape of mask must match the image: expected {}, '
                         'got {}'.format(shape, mask.shape))
    return mask


def _check_src(src: np.ndarray) -> np.ndarray:
    src = np.asarray(src, dtype=np.uint8)
    if src.size == 0 or src.ndim not in (2, 3):
        raise ValueError('Invalid src of shape {}: expected an 3D RGB image or '
                         'a 2D grayscale image'.format(src.shape))
    return src


def resize(src: np.ndarray, size: Tuple[int, int],
           energy_mode: str = 'backward', order: str = 'width-first',
           keep_mask: Optional[np.ndarray] = None) -> np.ndarray:
        
    src = _check_src(src)
    src_h, src_w = src.shape[:2]

    width, height = size
    width = int(round(width))
    height = int(round(height))
    if width <= 0 or height <= 0:
        raise ValueError('Invalid size {}: expected > 0'.format(size))
    if width >= 2 * src_w:
        raise ValueError('Invalid target width {}: expected less than twice '
                         'the source width (< {})'.format(width, 2 * src_w))
    if height >= 2 * src_h:
        raise ValueError('Invalid target height {}: expected less than twice '
                         'the source height (< {})'.format(height, 2 * src_h))

    if order not in VALID_ORDERS:
        raise ValueError('Invalid order {}: expected {}'.format(
            order, VALID_ORDERS))

    if energy_mode not in VALID_ENERGY_MODES:
        raise ValueError('Invalid energy mode {}: expected {}'.format(
            energy_mode, VALID_ENERGY_MODES))

    if keep_mask is not None:
        keep_mask = _check_mask(keep_mask, (src_h, src_w))

    if order == WIDTH_FIRST:
        src = _resize_width(src, width, energy_mode, keep_mask)
        src = _resize_height(src, height, energy_mode, keep_mask)
    else:
        src = _resize_height(src, height, energy_mode, keep_mask)
        src = _resize_width(src, width, energy_mode, keep_mask)

    return src


def remove_object(src: np.ndarray, drop_mask: np.ndarray,keep_mask: Optional[np.ndarray] = None) -> np.ndarray:

    src = _check_src(src)

    drop_mask = _check_mask(drop_mask, src.shape[:2])

    if keep_mask is not None:
        keep_mask = _check_mask(keep_mask, src.shape[:2])

    gray = src if src.ndim == 2 else _rgb2gray(src)

    while drop_mask.any():
        energy = _get_energy(gray)
        energy[drop_mask] -= DROP_MASK_ENERGY
        if keep_mask is not None:
            energy[keep_mask] += KEEP_MASK_ENERGY
        seam = _get_backward_seam(energy)
        seam_mask = _get_seam_mask(src, seam)
        gray = _remove_seam_mask(gray, seam_mask)
        drop_mask = _remove_seam_mask(drop_mask, seam_mask)
        src = _remove_seam_mask(src, seam_mask)
        if keep_mask is not None:
            keep_mask = _remove_seam_mask(keep_mask, seam_mask)

    return src


##############################################################################################################################
#CODIGO LZW


def compressed(input_file, n, rutaGuardado):          
    maximum_table_size = pow(2,int(n))   
    file = open(input_file)                 
    data = file.read()            

    # Building and initializing the dictionary.
    dictionary_size = 256                   
    dictionary = {chr(i): i for i in range(dictionary_size)}    
    string = ""             # String is null.
    compressed_data = []    # variable to store the compressed data.

    # iterating through the input symbols.
    # LZW Compression algorithm
    for symbol in data:                     
        string_plus_symbol = string + symbol # get input symbol.
        if string_plus_symbol in dictionary: 
            string = string_plus_symbol
        else:
            compressed_data.append(dictionary[string])
            if(len(dictionary) <= maximum_table_size):
                dictionary[string_plus_symbol] = dictionary_size
                dictionary_size += 1
            string = symbol

    if string in dictionary:
        compressed_data.append(dictionary[string])

    # storing the compressed string into a file (byte-wise).
    
            
    out = input_file.split(".")[0]
    out=input_file.split("/")[-1]

    output_file = open(rutaGuardado+out+ ".lzw", "wb")
    for data in compressed_data:
        output_file.write(pack('>H',int(data)))
        
    output_file.close()
    file.close()


def decompressed(input_file,ruta,rutaGuardado,n):
    maximum_table_size = pow(2,int(n))
    a = ruta+"/"+input_file
    file = open(a, "rb")
    compressed_data = []
    next_code = 256
    decompressed_data = ""
    string = ""

    # Reading the compressed file.
    while True:
        rec = file.read(2)
        if len(rec) != 2:
            break
        (data, ) = unpack('>H', rec)
        compressed_data.append(data)

    # Building and initializing the dictionary.
    dictionary_size = 256
    dictionary = dict([(x, chr(x)) for x in range(dictionary_size)])

    # iterating through the codes.
    # LZW Decompression algorithm
    for code in compressed_data:
        if not (code in dictionary):
            dictionary[code] = string + (string[0])
        decompressed_data += dictionary[code]
        if not(len(string) == 0):
            dictionary[next_code] = string + (dictionary[code][0])
            next_code += 1
        string = dictionary[code]

    # storing the decompressed string into a file.
    out = input_file.split(".")[0]
    output_file = open(rutaGuardado+out + "_decoded.csv", "w")
    for data in decompressed_data:
        output_file.write(data)
        
    output_file.close()
    file.close()
    

################################################################################################################################
def getFiles(ruta): 
    all_files = os.listdir(ruta)
    return all_files #Devuelve en una lista los archivos

def crearCarpetas(rutaScript):
    try:
        os.mkdir(rutaScript+"/csvComprimidos")
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    
    try:
        os.mkdir(rutaScript+"/csvComprimidos/enfermos_csv")
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    
    try:
        os.mkdir(rutaScript+"/csvComprimidos/sanos_csv")
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    
    try:
        os.mkdir(rutaScript+"/csvComprimidos/enfermosSeamCarving_csv")
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    try:
        os.mkdir(rutaScript+"/csvComprimidos/sanosSeamCarving_csv")
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    try:
        os.mkdir(rutaScript+"/csvDescomprimidos")
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    try:
        os.mkdir(rutaScript+"/csvDescomprimidos/enfermos_descomprimidosCSV")
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    
    try:
        os.mkdir(rutaScript+"/csvDescomprimidos/sanos_descomprimidosCSV")
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
#@profile
def ProcesoDeCompresion(archivo,ruta,rutaGuardado2,rutaGuardado1):# 1 Para Seam Carving y 2 para LZW
    inicio = time.time()
    for i in range(2): # para correr todo el codigo escribo len(archivo)
        #PASAR CSV A NUMPY
        src = genfromtxt(ruta+"/"+archivo[i], delimiter=',') #Aqui el archivo CSV sin comprimir se guarda en una matriz numpy
        #IMPLEMENTAR SEAM CARVING
        src_h, src_w = src.shape
        dst = resize( #Se comprime y se guarda en formato matriz numpy
                src, (src_w - 25, src_h-25),
                energy_mode='backward',   # Choose from {backward, forward}
                order='width-first',  # Choose from {width-first, height-first}
                keep_mask=None
                )

        #IMPLEMENTAR LZW
        # Paso el array de numpy a csv
        np.savetxt(rutaGuardado1+archivo[i], dst, delimiter=',' , fmt='%1.0f')
        compressed(rutaGuardado1+archivo[i],12,rutaGuardado2)
    final = time.time()
    print(final-inicio,"segundo de compresion para 30 archivos")


#@profile  
def procesoDeDescompresion(archivo,ruta,rutaGuardado): #Archivo es la lista de archivos a descomprimir, Ruta es donde se encuentra el archivo a descomprimir, rutaGuardado es donde se va a guardar
    inicio = time.time()
    for i in range(2):
        decompressed(archivo[i],ruta,rutaGuardado,12)
    final = time.time()
    print(final-inicio,"segundo de compresion para 30 archivos")

#@profile
def main():
    rutaScript = os.getcwd()
    crearCarpetas(rutaScript)

    # estos son los originales que es de donde inicia el proceso de compresion
    rutaScriptEnfermos = rutaScript+"/csv/enfermo_csv"
    rutaScriptSanos = rutaScript+"/csv/sano_csv"
    # aqui se van a guardar los archivos comprimidos en esta direccion

    rutaGuardadoEnfermos1 = "csvComprimidos/enfermosSeamCarving_csv/" #Para guardar Seam Carving
    rutaGuardadoSanos1 = "csvComprimidos/sanosSeamCarving_csv/"
    rutaGuardadoEnfermos2 = "csvComprimidos/enfermos_csv/" #Para guardar LZW
    rutaGuardadoSanos2 = "csvComprimidos/sanos_csv/"
    
    archivosEnfermos = getFiles(rutaScriptEnfermos) #Lista con todos los archivos del ganado enfermo
    archivosSanos = getFiles(rutaScriptSanos) #Lista con todos los archivos del ganado sano

    #Proceso con los archivos del ganado sano
    ProcesoDeCompresion(archivosSanos,rutaScriptSanos,rutaGuardadoSanos2,rutaGuardadoSanos1)
    #Proceso con los archivos del ganado enfermo
    ProcesoDeCompresion(archivosEnfermos,rutaScriptEnfermos,rutaGuardadoEnfermos2,rutaGuardadoEnfermos1)
    
    #########################################################################################################
    #DESCOMPRESION

    #Rutas donde estan los archivos ya comprimidos
    rutaScriptEnfermosCompr = rutaScript+"/csvComprimidos/enfermos_csv"
    rutaScriptSanosCompr = rutaScript+"/csvComprimidos/sanos_csv"
    #Ruta donde se van a guardar los archivos descomprimidos
    rutaGuardadoDescomEnfermos = "csvDescomprimidos/enfermos_descomprimidosCSV/"
    rutaGuardadoDescomSanos ="csvDescomprimidos/sanos_descomprimidosCSV/"
    #Sacar los archivos comprimidos
    archivosEnfermosComprimidos = getFiles(rutaGuardadoEnfermos2) #Busca los de LZW
    archivosSanosComprimidos = getFiles(rutaGuardadoSanos2)

    #Proceso descompresio con los archivos comprimidos del ganado sano
    procesoDeDescompresion(archivosSanosComprimidos,rutaScriptSanosCompr,rutaGuardadoDescomSanos) #archivosSanosComprimidos Lista de archivos, rutaScriptSanosCompr ruta para encontrar dichos archivos, rutaGuardadoDescomSanos a donde se va a aguardar
    #Proceso descompresio con los archivos comprimidos del ganado enfermo
    procesoDeDescompresion(archivosEnfermosComprimidos,rutaScriptEnfermosCompr,rutaGuardadoDescomEnfermos)

main()


"""
LICENCIAS DE USO DE CÓDIGO


CODIGO SEAM CARVING TOMADO DE: https://github.com/li-plus/seam-carving/blob/master/seam_carving/carve.py

MIT License

Copyright (c) 2020 Jiahao Li

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.





"""

