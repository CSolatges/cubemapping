from math import pi,atan2,hypot,floor
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
def cube_mapping(imgIn,imgOut):
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
imgOut = Image.new("RGB",(inSize[0],inSize[0]*3//4),"black")
cube_mapping(imgIn,imgOut)
imgOut.save(sys.argv[1].split('.')[0]+"_cubemap.png")


def create_faces(imgOut):
    OutPix=imgOut.load()
    OutSize=imgOut.size

    length=OutSize[0]//4
    height= OutSize[1]//3
    #création des images des 6 faces
    up=Image.new("RGB",(length,height),"black")
    down=Image.new("RGB",(length,height),"black")
    front=Image.new("RGB",(length,height),"black")
    back=Image.new("RGB",(length,height),"black")
    left=Image.new("RGB",(length,height),"black")
    right=Image.new("RGB",(length,height),"black")
    #remplissage de Up
    UpPix=up.load()
    for i in range(2*length,3*length):
         for j in range(height):
              UpPix[i-2*length-1,j]=OutPix[i,j]
    up=up.rotate(90) #mise de la tuile dans le bon sens
    #remplissage de down
    DownPix=down.load()
    for i in range(2*length, 3*length):
         for j in range(2*height,3*height):
              DownPix[i-2*length-1,j-2*height-1]=OutPix[i,j]
    down=down.rotate(270) #mise de la tuile dans le bon sens
    #remplissage de front
    FrontPix=front.load()
    for i in range(length, 2*length):
         for j in range(height, 2*height):
              FrontPix[i-length-1,j-height-1]=OutPix[i,j]

    #remplissage de back
    BackPix=back.load()
    for i in range(3*length, 4*length):
         for j in range(height, 2*height):
              BackPix[i-3*length-1,j-height-1]=OutPix[i,j]
    #remplissage de left
    LeftPix=left.load()
    for i in range(length):
         for j in range(height, 2*height):
              LeftPix[i,j-height-1]=OutPix[i,j]
    #remplissage de right
    RightPix=right.load()
    for i in range(2*length, 3*length):
         for j in range(height, 2*height):
              RightPix[i-2*length-1,j-height-1]=OutPix[i,j]
    return [back, left, front, right, up, down]

FACES= create_faces(imgOut)


def create_tiles(FACES):
    """fonction qui prend en argument le niveau de découpe et qui retourne les tuiles correspondantes"""
    """level n = (n+1)² tuiles / face"""
    dossier_parent = Path("/home/constance/Documents/GM4/Stage/codes/")
    img=str(sys.argv[1].split('.')[0])
    dossier_img=img
    for m in range(len(FACES)):
        dossier_face = faces_names[m]
        face=FACES[m]
        niv=str(0)
        dossier_niveau=niva=str(0)
        dossier_ligne=niv
        chemin_dossier_ligne = dossier_parent / dossier_img/ dossier_niveau / dossier_face / dossier_ligne
        chemin_dossier_ligne.mkdir(parents=True, exist_ok=True)
        face256=face.copy()
        face256.resize((256,256))
        file_name="_"+dossier_face+"_256.jpg"
        face256.save(chemin_dossier_ligne.resolve() / file_name)

        for level in range(number_level):
            largeur_image, hauteur_image = face.size
            largeur_sous_image = largeur_image / (level + 1)
            hauteur_sous_image = hauteur_image / (level + 1)
            niv=str(level+1)
            dossier_niveau = niv

            for i in range(2**level):
                a = str(i)
                dossier_ligne = a
                chemin_dossier_ligne = dossier_parent / dossier_img/ dossier_niveau / dossier_face / dossier_ligne
                chemin_dossier_ligne.mkdir(parents=True, exist_ok=True)

                for j in range(2**level):
                    x = i * largeur_sous_image
                    y = j * hauteur_sous_image
                    sous_image = face.crop((x, y, x + largeur_sous_image, y + hauteur_sous_image))
                    sous_image = sous_image.resize((512, 512))
                    sous_image.save(chemin_dossier_ligne.resolve() / "_{}.jpg".format(j))


#create_tiles(FACES)