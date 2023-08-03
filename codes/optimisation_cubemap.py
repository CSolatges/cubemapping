from math import pi, sin, cos, tan, atan2, hypot, floor
from numpy import clip
from PIL import Image
import numpy as np
import sys
import time
from pathlib import Path
import cProfile


def main():
    img_equi = Image.open(sys.argv[1])
    number_level, height = get_closest_2_pow_i_multiple_of_1024_from_height(
        img_equi.size[0])
    inSize = (2*height, height)
    img_equi = img_equi.resize(inSize)
    img_cubemap = Image.new("RGB", (inSize[0], int(inSize[0]*3/4)), "black")
    cube_mapping(img_equi, img_cubemap)


def get_closest_2_pow_i_multiple_of_1024_from_height(height: int) -> tuple[int, int]:
    """trouve le multiple de 1024 en fomat 2^i le plus proche de la hauteur; retourne, la valeur de i et la nouvelle hauteur """
    i = 0
    while height >= 1024*(2**i):
        i += 1
    return (i, 1024*2**i)


def out_img_to_xyz(face_matrix: np.array, edge_size: int, out_size: tuple(int, int)) -> np.array:
    I = np.arange(out_size[0])
    J = np.arange(3*edge_size)
    A = 2.0*float(I)/edge_size
    B = 2.0*float(J)/edge_size
    X, Y, Z = np.zeros(out_size[0], edge_size)


faces_names = ["b", "l", "f", "r", "u", "d"]


def create_faces_array(cubemap_width: int, edge_size: int) -> np.array:
    i_indices = np.arange(cubemap_width)
    face_number = i_indices//edge_size
    face = np.array(faces_names)[face_number]
    front_bool = face == "f"
    rng1 = np.arange(3*edge_size)
    rng2 = np.arange(edge_size, 2*edge_size)
    Mat = -np.ones((cubemap_width, 3*edge_size))
    Mat[front_bool] = rng1
    Mat[:, edge_size:2*edge_size][~front_bool] = rng2
    face_matrix = np.tile(face, (3*edge_size, 1))
    face_matrix[Mat > 2 * edge_size] = 'd'
    face_matrix[Mat < edge_size] = 'u'
    face_matrix[:, :edge_size][~front_bool] = 'n'  # notaface
    face_matrix[:, 2*edge_size:][~front_bool] = 'n'  # notaface
    return face_matrix


def interpolater(X, Y, Z, edge, inPix):
    THETA = atan2(Y, X)  # range -pi to pi
    R = hypot(X, Y)
    PHI = atan2(Z, R)  # range -pi/2 to pi/2
    # source img coords
    UF = (2.0*edge*(THETA + pi)/pi)
    VF = (2.0*edge * (pi/2 - PHI)/pi)
    # Use bilinear interpolation between the four surrounding pixels
    UI = floor(UF)  # coord of pixel to bottom left
    VI = floor(VF)
    U2 = UI+1       # coords of pixel to top right
    V2 = VI+1
    mu = UF-UI      # fraction of way across pixel
    nu = VF-VI
    # Pixel values of four corners
    A = inPix[UI % inSize[0], clip(VI, 0, inSize[1]-1)]
    B = inPix[U2 % inSize[0], clip(VI, 0, inSize[1]-1)]
    C = inPix[UI % inSize[0], clip(V2, 0, inSize[1]-1)]
    D = inPix[U2 % inSize[0], clip(V2, 0, inSize[1]-1)]
    # interpolate
    (r, g, b) = (
        A[0]*(1-mu)*(1-nu) + B[0]*(mu)*(1-nu) + C[0]*(1-mu)*nu+D[0]*mu*nu,
        A[1]*(1-mu)*(1-nu) + B[1]*(mu)*(1-nu) + C[1]*(1-mu)*nu+D[1]*mu*nu,
        A[2]*(1-mu)*(1-nu) + B[2]*(mu)*(1-nu) + C[2]*(1-mu)*nu+D[2]*mu*nu)
    return ((r, g, b))

# convert using an inverse transformation


def cube_mapping(img_equi, img_cubemap):
    out_size = img_cubemap.size
    equi_pixels = img_equi.load()
    cubemap_pixels = img_cubemap.load()
    edge = inSize[0]//4   # the length of each edge in pixels
    face_matrix = create_faces_array(out_size, edge)
    X, Y, Z = out_img_to_xyz(face_matrix, edge)
    (R, G, B) = interpolater(X, Y, Z, edge, equi_pixels)
