from math import pi, sin, cos, tan, atan2, hypot, floor
from pathlib import Path
import sys
from numpy import clip
from PIL import Image
import numpy as np


def main():
    img_equi = Image.open(sys.argv[1])
    number_level, height = get_closest_2_pow_i_multiple_of_1024_from_height(
        img_equi.size[0])
    (equi_width, equi_height) = (2*height, height)
    img_equi = img_equi.resize((equi_width, equi_height))
    img_cubemap = Image.new("RGB", (equi_width, int(equi_width*3/4)), "black")
    cube_mapping(img_equi, img_cubemap)
    faces = create_faces(img_cubemap)
    create_tiles(faces, number_level)


def get_closest_2_pow_i_multiple_of_1024_from_height(height: int) -> tuple[int, int]:
    """trouve le multiple de 1024 en fomat 2^i le plus proche de la hauteur; retourne, la valeur de i et la nouvelle hauteur """
    i = 0
    while height >= 1024*(2**i):
        i += 1
    return (i, 1024*2**i)


def from_equi_xy_to_cube_xyz_array(pixel_x_360: np.ndarray, pixel_y_360: np.ndarray, face: np.ndarray, edge_size: int) -> np.ndarray:
    a_coef = 2.0 * pixel_x_360 / edge_size
    b_coef = 2.0 * pixel_y_360 / edge_size

    x_coord = np.where(face == 'b', -1.0, np.where(face == 'l', a_coef - 3.0, np.where(face == 'f',
                                                                                       1.0, np.where(face == 'r', 7.0 - a_coef, np.where(face == 'u', b_coef - 1.0, 5.0 - b_coef)))))
    y_coord = np.where((face == 'b') | (face == 'r'), 1.0 - a_coef, np.where((face == 'l')
                                                                             | (face == 'f'), -1.0, np.where(face == 'u', a_coef - 5.0, a_coef - 5.0)))
    z_coord = np.where(face == 'b', 3.0 - b_coef, np.where(face == 'l', 3.0 - b_coef, np.where(face ==
                                                                                               'f', 3.0 - b_coef, np.where(face == 'r', 3.0 - b_coef, np.where(face == 'u', 1.0, -1.0)))))

    return np.column_stack((x_coord, y_coord, z_coord))


faces_names = ["b", "l", "f", "r", "u", "d"]


def create_faces_array(cubemap_width: int, edge_size: int) -> np.array:
    i_indices = np.arange(cubemap_width)
    face_number = i_indices//edge_size
    face = np.array(faces_names)[face_number]
    front_bool = face == "f"
    rng1 = np.arange(3*edge_size)
    rng2 = np.arange(edge_size, 2*edge_size)
    face_number_matrix = -np.ones((cubemap_width, 3*edge_size))
    face_number_matrix[front_bool] = rng1
    face_number_matrix[:, edge_size:2*edge_size][~front_bool] = rng2
    face_matrix = np.tile(face, (3*edge_size, 1))
    face_matrix[face_number_matrix > 2 * edge_size] = 'd'
    face_matrix[face_number_matrix < edge_size] = 'u'
    # face_matrix[:, :edge_size][~front_bool] = 'n'  # notaface
    # face_matrix[:, 2*edge_size:][~front_bool] = 'n'  # notaface
    return face_matrix


def interpolate_pixels(cube_coords: np.ndarray, edge_size: int, img_equi: Image.Image) -> tuple[np.array, np.array, np.array]:
    x_mat = cube_coords[:, 0]  # Extracting the first column (x coordinates)
    y_mat = cube_coords[:, 1]  # Extracting the second column (y coordinates)
    z_mat = cube_coords[:, 2]  # Extracting the third column (z coordinates)

    theta_coefs = np.arctan2(y_mat, x_mat)  # Applying arctan2
    r_coefs = np.hypot(x_mat, y_mat)  # Applying hypot (Euclidean norm)
    phi_coefs = np.arctan2(z_mat, r_coefs)  # Applying arctan2 to compute phi

    uf_mat = 2.0*edge_size*(theta_coefs+pi)/pi
    vf_mat = 2.0*edge_size*(pi/2-phi_coefs)/pi

    ui_mat = floor(uf_mat)
    vi_mat = floor(vf_mat)
    u_2 = ui_mat+1
    v_2 = vi_mat+1
    mu_coef = uf_mat-ui_mat
    nu_coef = vf_mat-vi_mat

    width = img_equi.width
    height = img_equi.height
    equi_pixels = img_equi.load()
    top_left_corners = equi_pixels[ui_mat % width, clip(vi_mat, 0, height-1)]
    top_right_corners = equi_pixels[u_2 % width, clip(vi_mat, 0, height-1)]
    bottom_left_corners = equi_pixels[ui_mat % width, clip(v_2, 0, height-1)]
    bottom_right_corners = equi_pixels[u_2 % width, clip(v_2, 0, height-1)]

    (r_mat, g_mat, b_mat) = (top_left_corners[:, 0]*(1-mu_coef)*(1-nu_coef) + top_right_corners[:, 0]*mu_coef*(1-nu_coef) + bottom_left_corners[:, 0]*(1-mu_coef)*nu_coef + bottom_right_corners[:, 0]*mu_coef*nu_coef,
                             top_left_corners[:, 1]*(1-mu_coef)*(1-nu_coef) + top_right_corners[:, 1]*mu_coef*(
                                 1-nu_coef) + bottom_left_corners[:, 1]*(1-mu_coef)*nu_coef + bottom_right_corners[:, 1]*mu_coef*nu_coef,
                             top_left_corners[:, 2]*(1-mu_coef)*(1-nu_coef) + top_right_corners[:, 2]*mu_coef*(1-nu_coef) + bottom_left_corners[:, 2]*(1-mu_coef)*nu_coef + bottom_right_corners[:, 2]*mu_coef*nu_coef)
    return (r_mat, g_mat, b_mat)


def cube_mapping(img_equi: Image.Image, img_cubemap: Image.Image) -> None:
    out_size = img_cubemap.size
    cubemap_pixels = img_cubemap.load()
    edge_size = img_equi.size[0]//4   # the length of each edge in pixels
    face_matrix = create_faces_array(out_size, edge_size)
    pixel_x_360 = np.arange(img_equi.size[0])
    pixel_y_360 = np.arange(img_equi.size[1])
    cube_coords = from_equi_xy_to_cube_xyz_array(
        pixel_x_360, pixel_y_360, face_matrix, edge_size)
    (r_mat, g_mat, b_mat) = interpolate_pixels(
        cube_coords, edge_size, img_equi)
    cubemap_pixels = (int(round(r_mat)), int(round(g_mat)), int(round(b_mat)))


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
    img = str(sys.argv[1].split('.')[0])
    dossier_img = img
    for (face, dossier_face) in zip(faces, faces_names):
        niv = '0'
        dossier_niveau = '0'
        dossier_ligne = '0'
        chemin_dossier_ligne = Path(
            dossier_parent, dossier_img, dossier_niveau, dossier_face, dossier_ligne)
        chemin_dossier_ligne.mkdir(parents=True, exist_ok=True)
        face256 = face.copy()
        face256.resize((256, 256))
        file_name = "_"+dossier_face+"_256.jpg"
        face256.save(chemin_dossier_ligne.resolve() / file_name)

        for level in range(number_level):
            largeur_image, hauteur_image = face.size
            largeur_sous_image = largeur_image / (level + 1)
            hauteur_sous_image = hauteur_image / (level + 1)
            niv = str(level+1)
            dossier_niveau = niv

            for i in range(2**level):
                a = str(i)
                dossier_ligne = a
                chemin_dossier_ligne = dossier_parent / dossier_img / \
                    dossier_niveau / dossier_face / dossier_ligne
                chemin_dossier_ligne.mkdir(parents=True, exist_ok=True)

                for j in range(2**level):
                    x = i * largeur_sous_image
                    y = j * hauteur_sous_image
                    sous_image = face.crop(
                        (x, y, x + largeur_sous_image, y + hauteur_sous_image))
                    sous_image = sous_image.resize((512, 512))
                    sous_image.save(
                        chemin_dossier_ligne.resolve() / "_{}.jpg".format(j))


# if __name__ == '__main__':
#    main()
