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
#inSize=(2*height,height)
#imgIn.resize(inSize)

faces_names=["b", "l", "f", "r","u", "d"]

def outImgToXYZ(i,j,face,edge):
    a = 2.0*float(i)/edge
    b = 2.0*float(j)/edge
    if face=='b':
        (x,y,z) = (-1.0, 1.0-a, 3.0 - b)
    elif face=='l':
        (x,y,z) = (a-3.0, -1.0, 3.0 - b)
    elif face=='f': 
        (x,y,z) = (1.0, a - 5.0, 3.0 - b)
    elif face=='r': 
        (x,y,z) = (7.0-a, 1.0, 3.0 - b)
    elif face=='u': 
        (x,y,z) = (b-1.0, a -5.0, 1.0)
    elif face=='d':
        (x,y,z) = (5.0-b, a-5.0, -1.0)
    return (x,y,z)

# convert using an inverse transformation
def convertBack(imgIn,imgOut):
    inSize=imgIn.size
    outSize = imgOut.size
    inPix = imgIn.load()
    outPix = imgOut.load()
    edge = inSize[0]//4   # the length of each edge in pixels
    for i in range(outSize[0]):
        face_number = i//edge 
        face=faces_names[face_number]
        if face_number==2:
            rng = range(0,edge*3)
        else:
            rng = range(edge,edge*2)

        for j in rng:
            if j<edge:
                face2 = 'u' 
            elif j>=2*edge:
                face2 = 'd'
            else:
                face2 = face
            (x,y,z) = outImgToXYZ(i,j,face2,edge)
            theta = atan2(y,x) # range -pi to pi
            r = hypot(x,y)
            phi = atan2(z,r) # range -pi/2 to pi/2
            # source img coords
            uf = ( 2.0*edge*(theta + pi)/pi )
            vf = ( 2.0*edge * (pi/2 - phi)/pi)
            # Use bilinear interpolation between the four surrounding pixels
            ui = floor(uf)  # coord of pixel to bottom left
            vi = floor(vf)
            u2 = ui+1       # coords of pixel to top right
            v2 = vi+1
            mu = uf-ui      # fraction of way across pixel
            nu = vf-vi
            # Pixel values of four corners
            A = inPix[ui % inSize[0],clip(vi,0,inSize[1]-1)]
            B = inPix[u2 % inSize[0],clip(vi,0,inSize[1]-1)]
            C = inPix[ui % inSize[0],clip(v2,0,inSize[1]-1)]
            D = inPix[u2 % inSize[0],clip(v2,0,inSize[1]-1)]
            # interpolate
            (r,g,b) = (
              A[0]*(1-mu)*(1-nu) + B[0]*(mu)*(1-nu) + C[0]*(1-mu)*nu+D[0]*mu*nu,
              A[1]*(1-mu)*(1-nu) + B[1]*(mu)*(1-nu) + C[1]*(1-mu)*nu+D[1]*mu*nu,
              A[2]*(1-mu)*(1-nu) + B[2]*(mu)*(1-nu) + C[2]*(1-mu)*nu+D[2]*mu*nu )

            outPix[i,j] = (int(round(r)),int(round(g)),int(round(b)))

inSize=imgIn.size
imgOut = Image.new("RGB",(inSize[0],int(inSize[0]*3/4)),"black")
convertBack(imgIn,imgOut)
imgOut.save(sys.argv[1].split('.')[0]+"cubemap.png")
