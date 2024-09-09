from PIL import Image
import cv2
from skimage import io, color
from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine, euclidean
import dlib
from sklearn.decomposition import PCA
from scipy.spatial import procrustes
from numpy.linalg import norm
from scipy.spatial.distance import cosine
import os
import csv

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("data/models/shape_predictor_68_face_landmarks.dat")


def get_landmarks(image_path):
    try:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        for face in faces:
            landmarks = predictor(gray, face)
            matrix = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]
            return np.asarray(matrix), faces
    except Exception as e:
        print("Fatal:", e)


def helmert_transformation(A, B):
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    A_centered = A - centroid_A
    B_centered = B - centroid_B

    H = np.dot(A_centered.T, B_centered)
    # Singular Value Decomposition.
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    scale = np.sum(S) / np.sum(A_centered ** 2)
    t = centroid_B - scale * np.dot(centroid_A, R)

    # scale:Ölçek, t:öteleme parametreleri, R:Dönüşüm matrisi
    return scale, R, t


def calculate_rmse(A_transformed, B):
    """
    Dönüştürülmüş nokta kümesi ile hedef nokta kümesi arasındaki RMSE'yi hesaplar.

    Args:
    - A_transformed: Dönüştürülmüş nokta kümesi (n x m) boyutunda numpy array.
    - B: Hedef nokta kümesi (n x m) boyutunda numpy array.

    Returns:
    - rmse: Dönüşümün standart sapması (RMSE).
    """
    residuals = A_transformed - B
    # print(np.max(residuals))
    squared_residuals = np.square(residuals)
    mse = np.mean(squared_residuals)
    rmse = np.sqrt(mse)
    return rmse


def plot_transformation(A, A_transformed, B):
    fig, ax = plt.subplots()
    plt.scatter(A[:, 0], A[:, 1], c='blue', label='Orijinal A', marker='o')
    plt.scatter(A_transformed[:, 0], A_transformed[:, 1], c='green', label='Dönüştürülmüş A', marker='^')
    plt.scatter(B[:, 0], B[:, 1], c='red', label='Hedef B', marker='x')

    for i in range(A.shape[0]):
        plt.plot([A[i, 0], A_transformed[i, 0]], [A[i, 1], A_transformed[i, 1]], 'k--')
        plt.plot([A_transformed[i, 0], B[i, 0]], [A_transformed[i, 1], B[i, 1]], 'g--')

    # Resim koordinatlarından gelen değerleri düzgün resmetmek için
    ax.invert_yaxis()

    plt.title('Helmert Dönüşümü')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()


def start():
    # Burada resimlerin içinde eşleştirme yaparak başlıcak buradakı

    directory = os.path.dirname(os.path.abspath(__file__)) + "/data/faces"
    files = os.listdir(directory)

    for f in files:
        result = []
        a, faces1a = get_landmarks(directory + "/" + f)
        files2 = os.listdir(directory)
        print(f)
        for ff in files2:
            if f != ff:
                b, face1b = get_landmarks(directory + "/" + ff)
                pca = PCA(n_components=2)
                pca.fit(a)
                a_transformed = pca.transform(a)

                pca.fit(b)
                b_transformed = pca.transform(b)

                # Procrustes
                matrix1, matrix2, disparity = procrustes(a_transformed, b_transformed)
                distances = np.linalg.norm(matrix1 - matrix2, axis=1)
                std_dev = np.std(distances)

                # Frobenius
                frobenius_norm = norm(matrix1 - matrix2)

                # Cosine Similarity
                cosine_similarity = 1 - cosine(matrix1.flatten(), matrix2.flatten())

                # Helmert
                scale, R, t = helmert_transformation(a, b)
                # A nokta kümesini dönüştür
                a_transformed = scale * np.dot(a, R) + t

                # RMSE'yi hesapla
                rmse = calculate_rmse(a_transformed, b)
                # plot_transformation(a, a_transformed, b)
                result.append([f, ff, disparity, std_dev, frobenius_norm, cosine_similarity, rmse])
        result_path = os.path.dirname(os.path.abspath(__file__)) + "/results/" + f + ".csv"
        with open(result_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(result)


start()
