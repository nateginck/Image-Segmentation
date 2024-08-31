from kmeans_multiple import kmeans_multiple
import numpy as np

def Segment_kmeans(im_in, K, iters, R):
    # find shape of object
    H, W, C = im_in.shape

    # reshape image
    X = im_in.reshape(H * W, C)

    # call kmeans function
    ids, means, ssd = kmeans_multiple(X, K, iters, R)

    # recolor image
    im_out = means[ids].reshape(H, W, C)
    return im_out
