from math import pi, atan2, hypot, floor
import sys
from pathlib import Path
from numpy import clip
from PIL import Image
# import numpy as np

# import time

# import cProfile


def get_closest_2_pow_i_multiple_of_1024_from_height(hauteur: int) -> tuple[int, int]:
    """trouve le multiple de 1024 en fomat 2^i le plus proche de la hauteur; retourne, la valeur de i et la nouvelle hauteur """
    i = 0
    while hauteur >= 1024*(2**i):
        i += 1
    return (i, 1024*(2**i))


def main():
    img_equi_path = sys.argv[1]
    img_equi = Image.open(img_equi_path)
    number_level, height = get_closest_2_pow_i_multiple_of_1024_from_height(
        img_equi.height)
    (equi_width, equi_height) = (2*height, height)
    (cubemap_width, cubemap_height) = (equi_width, equi_width*3//4)
    img_equi = img_equi.resize((equi_width, equi_height))
    img_cubemap = Image.new(
        "RGB", (cubemap_width, cubemap_height), "black")
    cube_mapping(img_equi, img_cubemap)
    img_cubemap_path = img_equi_path.split('.', maxsplit=1)[0]+"_cubemap.png"
    img_cubemap.save(img_cubemap_path)
    faces = create_faces(img_cubemap)
    create_tiles(faces, number_level)


faces_names = ["b", "l", "f", "r", "u", "d"]


def from_equi_xy_to_cube_xyz(pixel_x_360: int, pixel_y_360: int, face: str, edge_size: int) -> tuple[float, float, float]:
    a = 2.0*float(pixel_x_360)/edge_size
    b = 2.0*float(pixel_y_360)/edge_size
    if face == 'b':
        return (-1.0, 1.0-a, 3.0 - b)
    if face == 'l':
        return (a-3.0, -1.0, 3.0 - b)
    if face == 'f':
        return (1.0, a - 5.0, 3.0 - b)
    if face == 'r':
        return (7.0-a, 1.0, 3.0 - b)
    if face == 'u':
        return (b-1.0, a - 5.0, 1.0)
    if face == 'd':
        return (5.0-b, a-5.0, -1.0)

# convert using an inverse transformation


def cube_mapping(img_equi: Image.Image, img_cubemap: Image.Image) -> None:
    inPix = img_equi.load()
    outPix = img_cubemap.load()
    edge_size = img_equi.width//4   # the length of each edge in pixels
    for i in range(img_cubemap.width):
        face_number = i//edge_size
        face = faces_names[face_number]
        if face_number == 2:
            rng = range(0, edge_size*3)
        else:
            rng = range(edge_size, edge_size*2)

        for j in rng:
            if j < edge_size:
                face2 = 'u'
            elif j >= 2*edge_size:
                face2 = 'd'
            else:
                face2 = face
            (x, y, z) = from_equi_xy_to_cube_xyz(i, j, face2, edge_size)
            theta = atan2(y, x)  # range -pi to pi
            r = hypot(x, y)
            phi = atan2(z, r)  # range -pi/2 to pi/2
            # source img coords
            uf = 2.0*edge_size*(theta + pi)/pi
            vf = 2.0*edge_size * (pi/2 - phi)/pi
            # Use bilinear interpolation between the four surrounding pixels
            ui = floor(uf)  # coord of pixel to bottom left
            vi = floor(vf)
            u2 = ui+1       # coords of pixel to top right
            v2 = vi+1
            mu = uf-ui      # fraction of way across pixel
            nu = vf-vi
            # Pixel values of four corners
            A = inPix[ui % img_equi.width, clip(vi, 0, img_equi.height-1)]
            B = inPix[u2 % img_equi.width, clip(vi, 0, img_equi.height-1)]
            C = inPix[ui % img_equi.width, clip(v2, 0, img_equi.height-1)]
            D = inPix[u2 % img_equi.width, clip(v2, 0, img_equi.height-1)]
            # interpolate
            (r, g, b) = (
                A[0]*(1-mu)*(1-nu) + B[0]*(mu) *
                (1-nu) + C[0]*(1-mu)*nu+D[0]*mu*nu,
                A[1]*(1-mu)*(1-nu) + B[1]*(mu) *
                (1-nu) + C[1]*(1-mu)*nu+D[1]*mu*nu,
                A[2]*(1-mu)*(1-nu) + B[2]*(mu)*(1-nu) + C[2]*(1-mu)*nu+D[2]*mu*nu)

            outPix[i, j] = (int(round(r)), int(round(g)), int(round(b)))


Faces = tuple[Image.Image, Image.Image, Image.Image,
              Image.Image, Image.Image, Image.Image]


def create_faces(img_cubemap: Image.Image) -> Faces:
    cubemap_pixels = img_cubemap.load()

    length = img_cubemap.width//4
    height = img_cubemap.height//3
    # création des images des 6 faces
    up = Image.new("RGB", (length, height), "black")
    down = Image.new("RGB", (length, height), "black")
    front = Image.new("RGB", (length, height), "black")
    back = Image.new("RGB", (length, height), "black")
    left = Image.new("RGB", (length, height), "black")
    right = Image.new("RGB", (length, height), "black")
    # remplissage de Up
    up_pixels = up.load()
    for i in range(2*length, 3*length):
        for j in range(height):
            up_pixels[i-2*length-1, j] = cubemap_pixels[i, j]
    up = up.rotate(90)  # mise de la tuile dans le bon sens
    # remplissage de down
    down_pixels = down.load()
    for i in range(2*length, 3*length):
        for j in range(2*height, 3*height):
            down_pixels[i-2*length-1, j-2*height-1] = cubemap_pixels[i, j]
    down = down.rotate(270)  # mise de la tuile dans le bon sens
    # remplissage de front
    front_pixels = front.load()
    for i in range(length, 2*length):
        for j in range(height, 2*height):
            front_pixels[i-length-1, j-height-1] = cubemap_pixels[i, j]

    # remplissage de back
    back_pixels = back.load()
    for i in range(3*length, 4*length):
        for j in range(height, 2*height):
            back_pixels[i-3*length-1, j-height-1] = cubemap_pixels[i, j]
    # remplissage de left
    left_pixels = left.load()
    for i in range(length):
        for j in range(height, 2*height):
            left_pixels[i, j-height-1] = cubemap_pixels[i, j]
    # remplissage de right
    right_pixels = right.load()
    for i in range(2*length, 3*length):
        for j in range(height, 2*height):
            right_pixels[i-2*length-1, j-height-1] = cubemap_pixels[i, j]
    return (back, left, front, right, up, down)


def create_tiles(faces: Faces, number_level: int):
    # mettre le chemin absolu du dossier ou se situe le code
    dossier_parent = Path("./cubemapping/codes")
    img = str(sys.argv[1].split('.', maxsplit=1)[0])
    dossier_img = img
    for (face, dossier_face) in zip(faces, faces_names):
        niv = '0'
        dossier_niveau = '0'
        dossier_ligne = '0'
        chemin_dossier_ligne = Path(
            dossier_parent, dossier_img, dossier_niveau, dossier_face, dossier_ligne)
        chemin_dossier_ligne.mkdir(parents=True, exist_ok=True)
        face256 = face.copy()
        face256 = face256.resize((256, 256))
        file_name = "0.jpg"
        face256.save(chemin_dossier_ligne.resolve() / file_name)

        for level in range(number_level):
            largeur_image, hauteur_image = face.size
            largeur_sous_image = largeur_image / (level + 1)
            hauteur_sous_image = hauteur_image / (level + 1)
            niv = str(level+1)
            dossier_niveau = niv

            for col in range(2**level):
                a = str(col)
                dossier_ligne = a
                chemin_dossier_ligne = dossier_parent / dossier_img / \
                    dossier_niveau / dossier_face / dossier_ligne
                chemin_dossier_ligne.mkdir(parents=True, exist_ok=True)

                for line in range(2**level):
                    x = line * largeur_sous_image
                    y = col * hauteur_sous_image
                    sous_image = face.crop(
                        (x, y, x + largeur_sous_image, y + hauteur_sous_image))
                    sous_image = sous_image.resize((512, 512))
                    sous_image.save(
                        chemin_dossier_ligne.resolve() / "{}.jpg".format(line))


if __name__ == '__main__':
    main()
