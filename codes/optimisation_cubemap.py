from math import pi,sin,cos,tan,atan2,hypot,floor
from numpy import clip
from PIL import Image
import numpy as np
import sys
import time
from pathlib import Path
import cProfile 


def get_closest_2_pow_i_multiple_of_1024_from_height(eqrec):
    """trouve le multiple de 1024 en fomat 2^i le plus proche de la hauteur; retourne, la valeur de i et la nouvelle hauteur """
    hauteur=eqrec.size[1]
    i=0
    while hauteur >= 1024*(2**i):
        i+=1
    return(i,1024*2**i)

imgIn = Image.open(sys.argv[1])
number_level, height=get_closest_2_pow_i_multiple_of_1024_from_height(imgIn)
inSize=(2*height,height)
imgIn.resize(inSize)

def outImgToXYZ(i,j,face,edge):
    a = 2.0*float(i)/edge
    b = 2.0*float(j)/edge
    if face=="b":
        (x,y,z) = (-1.0, 1.0-a, 3.0 - b)
    elif face=="l":
        (x,y,z) = (a-3.0, -1.0, 3.0 - b)
    elif face=="f":
        (x,y,z) = (1.0, a - 5.0, 3.0 - b)
    elif face=="r":
        (x,y,z) = (7.0-a, 1.0, 3.0 - b)
    elif face=="u":
        (x,y,z) = (b-1.0, a -5.0, 1.0)
    elif face=="d":
        (x,y,z) = (5.0-b, a-5.0, -1.0)
    return (x,y,z)

faces_names=["b", "l", "f", "r","u", "d"]

def create_faces_array(outSize, edge):
    i_indices = np.arange(outSize[0])
    face_number=i_indices//edge
    face=np.array(faces_names)[face_number]
    front_bool=face=="f"
    rng=np.arange(3*edge)
    face2=np.tile(face, (3*edge, 1))
    face2lowerthanedge= rng<edge
    face2greaterthan2edges = rng>=2*edge
    face3=np.where(face2lowerthanedge, "u", np.where(face2greaterthan2edges, "d", face2))
    return face3


# convert using an inverse transformation
def cube_mapping (imgIn,imgOut):
    outSize = imgOut.size
    inPix = imgIn.load()
    outPix = imgOut.load()
    edge = inSize[0]//4   # the length of each edge in pixels
    mat= create_faces_array(outSize, edge)
    print(mat)

    """faire une fonction annexe qui s'occupe de créer la matrice avec les bonnes faces 
    et lui exécuter la fonction outImgTXYZ de manière à """


imgOut = Image.new("RGB",(inSize[0],int(inSize[0]*3/4)),"black")
cube_mapping(imgIn,imgOut)