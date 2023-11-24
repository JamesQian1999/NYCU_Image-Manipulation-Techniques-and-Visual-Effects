import numpy as np
import sklearn.neighbors
import scipy.sparse
import warnings
import cv2
import argparse


def knn_matting(image, trimap, my_lambda = 100, n_neighbors = 10, feature="rgb"):
    [h, w, c] = image.shape
    image, trimap = image / 255.0, trimap / 255.0
    foreground = (trimap == 1.0).astype(int)
    background = (trimap == 0.0).astype(int)

    ####################################################
    # TODO: find KNN for the given image
    ####################################################
    x, y = np.unravel_index(np.arange(h*w), (h, w))
    if feature=="rgb":
        FeatureVector = np.transpose(image.reshape(h*w,c)).T
        C = 3
    elif feature=="rgbxy":
        FeatureVector = np.append(image.reshape(h*w, c).T, [x, y], axis=0).T
        C = 5
    else:
        raise NotImplementedError("Unknown feature type")

    knns = sklearn.neighbors.NearestNeighbors(n_neighbors=n_neighbors, n_jobs=8).fit(FeatureVector).kneighbors(FeatureVector)[1]

    ####################################################
    # TODO: compute the affinity matrix A
    #       and all other stuff needed
    ####################################################
    Row = np.repeat(np.arange(h*w), n_neighbors)
    Col = knns.reshape(h*w*n_neighbors)
    k = 1 - np.linalg.norm(FeatureVector[Row] - FeatureVector[Col], axis=1)/C
    A = scipy.sparse.coo_matrix((k, (Row, Col)),shape=(h*w, h*w))
    
    ####################################################
    # TODO: solve for the linear system,
    #       note that you may encounter en error
    #       if no exact solution exists
    ####################################################
    D = scipy.sparse.diags(np.ravel(A.sum(axis=1)))
    M = scipy.sparse.diags(np.ravel(trimap))
    L = D-A
    v = np.ravel(foreground)

    c = my_lambda*v.T
    H = (L + my_lambda*M)

    warnings.filterwarnings('error')
    alpha = []
    try:
        alpha = np.minimum(np.maximum(scipy.sparse.linalg.spsolve(H, c), 0), 1).reshape(h, w)
    except Warning:
        alpha = np.minimum(np.maximum(scipy.sparse.linalg.lsqr(H, c)[0], 0), 1).reshape(h, w)

    mask = foreground + background

    alpha = alpha * (1-mask) + foreground
    return alpha

def init():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=str
    )
    parser.add_argument(
        "--trimap",
        type=str
    )
    parser.add_argument(
        "--bg",
        type=str
    )
    parser.add_argument(
        "--output",
        type=str
    )
    parser.add_argument(
        "--my_lambda",
        type=int
    )
    parser.add_argument(
        "--n_neighbors",
        type=int
    )
    parser.add_argument(
        "--feature",
        type=str
    )
    args = parser.parse_args()
    return args

def main():

    args = init()
    image = cv2.imread(args.image)
    trimap = cv2.imread(args.trimap if args.trimap is not None else f"trimap/{args.image.split('/')[1]}", cv2.IMREAD_GRAYSCALE)

    alpha = knn_matting(image, trimap, args.my_lambda, args.n_neighbors, args.feature)
    alpha = alpha[:, :, np.newaxis]

    ####################################################
    # TODO: pick up your own background image, 
    #       and merge it with the foreground
    ####################################################
    bg = cv2.imread(args.bg)
    [h, w, c] = bg.shape
    bg_w = cv2.resize(bg, (image.shape[1], int(image.shape[1]/w*h)))
    bg_h = cv2.resize(bg, (int(image.shape[0]/h*w), image.shape[0]))
    bg = bg_w if(bg_w.shape[0] >= bg.shape[0] and bg_w.shape[1] >= bg.shape[1]) else bg_h
    bg = bg[:image.shape[0], :image.shape[1], :]
    bg = cv2.GaussianBlur(bg, (5, 5), 0)
    result = image * alpha + bg * (1 - alpha)

    outimg = args.output if args.output is not None else f"result/{args.image.split('/')[1].split('.')[0]}_lambda[{args.my_lambda}]_knn[{args.n_neighbors}]_featureVec[{args.feature}].png"
    cv2.imwrite(outimg, result)

if __name__ == "__main__":
    main()
    # python3 -u release.py --image image/gandalf.png --trimap trimap/gandalf.png --bg background/bg.jpg --output result/gandalf.png --my_lambda 20 --n_neighbors 10 --feature rgbxy
