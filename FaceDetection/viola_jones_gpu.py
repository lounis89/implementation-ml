"""
A Python implementation of the Viola-Jones ensemble classification method described in 
Viola, Paul, and Michael Jones. "Rapid object detection using a boosted cascade of simple features." Computer Vision and Pattern Recognition, 2001. CVPR 2001. Proceedings of the 2001 IEEE Computer Society Conference on. Vol. 1. IEEE, 2001.
Works in both Python2 and Python3
"""
import numpy as np
import math
import pickle
from sklearn.feature_selection import SelectPercentile, f_classif
from numba import cuda
from numba.cuda.cudadrv.driver import driver
from numba.np import numpy_support as nps
import numba as nb
from math import *

class ViolaJones:
    def __init__(self, T = 10):
        """
          Args:
            T: The number of weak classifiers which should be used
        """
        self.T = T
        self.alphas = []
        self.clfs = []

    def train(self, training, pos_num, neg_num):
        """
        Trains the Viola Jones classifier on a set of images (numpy arrays of shape (m, n))
          Args:
            training: An array of tuples. The first element is the numpy array of shape (m, n) representing the image. The second element is its classification (1 or 0)
            pos_num: the number of positive samples
            neg_num: the number of negative samples
        """
        weights = np.zeros(len(training))
        training_data = []
        print("Computing integral images")
        for x in range(len(training)):
            training_data.append((integral_image(training[x][0]), training[x][1]))
            if training[x][1] == 1:
                weights[x] = 1.0 / (2 * pos_num)
            else:
                weights[x] = 1.0 / (2 * neg_num)
        """
        print("Building features")
        features = self.build_features(training_data[0][0].shape)
        print("Applying features to training examples")
        X, y = self.apply_features(features, training_data)
        print("Selecting best features")
        indices = SelectPercentile(f_classif, percentile=10).fit(X.T, y).get_support(indices=True)
        X = X[indices]
        features = features[indices]
        print("Selected %d potential features" % len(X))

        for t in range(self.T):
            weights = weights / np.linalg.norm(weights)
            weak_classifiers = self.train_weak(X, y, features, weights)
            clf, error, accuracy = self.select_best(weak_classifiers, weights, training_data)
            beta = error / (1.0 - error)
            for i in range(len(accuracy)):
                weights[i] = weights[i] * (beta ** (1 - accuracy[i]))
            alpha = math.log(1.0/beta)
            self.alphas.append(alpha)
            self.clfs.append(clf)
            print("Chose classifier: %s with accuracy: %f and alpha: %f" % (str(clf), len(accuracy) - sum(accuracy), alpha))
        """
    def train_weak(self, X, y, features, weights):
        """
        Finds the optimal thresholds for each weak classifier given the current weights
          Args:
            X: A numpy array of shape (len(features), len(training_data)). Each row represents the value of a single feature for each training example
            y: A numpy array of shape len(training_data). The ith element is the classification of the ith training example
            features: an array of tuples. Each tuple's first element is an array of the rectangle regions which positively contribute to the feature. The second element is an array of rectangle regions negatively contributing to the feature
            weights: A numpy array of shape len(training_data). The ith element is the weight assigned to the ith training example
          Returns:
            An array of weak classifiers
        """
        total_pos, total_neg = 0, 0
        for w, label in zip(weights, y):
            if label == 1:
                total_pos += w
            else:
                total_neg += w

        classifiers = []
        total_features = X.shape[0]
        for index, feature in enumerate(X):
            if len(classifiers) % 1000 == 0 and len(classifiers) != 0:
                print("Trained %d classifiers out of %d" % (len(classifiers), total_features))

            applied_feature = sorted(zip(weights, feature, y), key=lambda x: x[1])

            pos_seen, neg_seen = 0, 0
            pos_weights, neg_weights = 0, 0
            min_error, best_feature, best_threshold, best_polarity = float('inf'), None, None, None
            for w, f, label in applied_feature:
                error = min(neg_weights + total_pos - pos_weights, pos_weights + total_neg - neg_weights)
                if error < min_error:
                    min_error = error
                    best_feature = features[index]
                    best_threshold = f
                    best_polarity = 1 if pos_seen > neg_seen else -1

                if label == 1:
                    pos_seen += 1
                    pos_weights += w
                else:
                    neg_seen += 1
                    neg_weights += w
            
            clf = WeakClassifier(best_feature[0], best_feature[1], best_threshold, best_polarity)
            classifiers.append(clf)
        return classifiers
                
    def build_features(self, image_shape):
        """
        Builds the possible features given an image shape
          Args:
            image_shape: a tuple of form (height, width)
          Returns:
            an array of tuples. Each tuple's first element is an array of the rectangle regions which positively contribute to the feature. The second element is an array of rectangle regions negatively contributing to the feature
        """
        height, width = image_shape
        features = []
        for w in range(1, width+1):
            for h in range(1, height+1):
                i = 0
                while i + w < width:
                    j = 0
                    while j + h < height:
                        #2 rectangle features
                        immediate = RectangleRegion(i, j, w, h)
                        right = RectangleRegion(i+w, j, w, h)
                        if i + 2 * w < width: #Horizontally Adjacent
                            features.append(([right], [immediate]))

                        bottom = RectangleRegion(i, j+h, w, h)
                        if j + 2 * h < height: #Vertically Adjacent
                            features.append(([immediate], [bottom]))
                        
                        right_2 = RectangleRegion(i+2*w, j, w, h)
                        #3 rectangle features
                        if i + 3 * w < width: #Horizontally Adjacent
                            features.append(([right], [right_2, immediate]))

                        bottom_2 = RectangleRegion(i, j+2*h, w, h)
                        if j + 3 * h < height: #Vertically Adjacent
                            features.append(([bottom], [bottom_2, immediate]))

                        #4 rectangle features
                        bottom_right = RectangleRegion(i+w, j+h, w, h)
                        if i + 2 * w < width and j + 2 * h < height:
                            features.append(([right, bottom], [immediate, bottom_right]))

                        j += 1
                    i += 1
        return np.array(features)

    def select_best(self, classifiers, weights, training_data):
        """
        Selects the best weak classifier for the given weights
          Args:
            classifiers: An array of weak classifiers
            weights: An array of weights corresponding to each training example
            training_data: An array of tuples. The first element is the numpy array of shape (m, n) representing the integral image. The second element is its classification (1 or 0)
          Returns:
            A tuple containing the best classifier, its error, and an array of its accuracy
        """
        best_clf, best_error, best_accuracy = None, float('inf'), None
        for clf in classifiers:
            error, accuracy = 0, []
            for data, w in zip(training_data, weights):
                correctness = abs(clf.classify(data[0]) - data[1])
                accuracy.append(correctness)
                error += w * correctness
            error = error / len(training_data)
            if error < best_error:
                best_clf, best_error, best_accuracy = clf, error, accuracy
        return best_clf, best_error, best_accuracy
    
    def apply_features(self, features, training_data):
        """
        Maps features onto the training dataset
          Args:
            features: an array of tuples. Each tuple's first element is an array of the rectangle regions which positively contribute to the feature. The second element is an array of rectangle regions negatively contributing to the feature
            training_data: An array of tuples. The first element is the numpy array of shape (m, n) representing the integral image. The second element is its classification (1 or 0)
          Returns:
            X: A numpy array of shape (len(features), len(training_data)). Each row represents the value of a single feature for each training example
            y: A numpy array of shape len(training_data). The ith element is the classification of the ith training example
        """
        X = np.zeros((len(features), len(training_data)))
        y = np.array(list(map(lambda data: data[1], training_data)))
        i = 0
        for positive_regions, negative_regions in features:
            feature = lambda ii: sum([pos.compute_feature(ii) for pos in positive_regions]) - sum([neg.compute_feature(ii) for neg in negative_regions])
            X[i] = list(map(lambda data: feature(data[0]), training_data))
            i += 1
        return X, y

    def classify(self, image):
        """
        Classifies an image
          Args:
            image: A numpy 2D array of shape (m, n) representing the image
          Returns:
            1 if the image is positively classified and 0 otherwise
        """
        total = 0
        ii = integral_image(image)
        for alpha, clf in zip(self.alphas, self.clfs):
            total += alpha * clf.classify(ii)
        return 1 if total >= 0.5 * sum(self.alphas) else 0

    def save(self, filename):
        """
        Saves the classifier to a pickle
          Args:
            filename: The name of the file (no file extension necessary)
        """
        with open(filename+".pkl", 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        A static method which loads the classifier from a pickle
          Args:
            filename: The name of the file (no file extension necessary)
        """
        with open(filename+".pkl", 'rb') as f:
            return pickle.load(f)

class WeakClassifier:
    def __init__(self, positive_regions, negative_regions, threshold, polarity):
        """
          Args:
            positive_regions: An array of RectangleRegions which positively contribute to a feature
            negative_regions: An array of RectangleRegions which negatively contribute to a feature
            threshold: The threshold for the weak classifier
            polarity: The polarity of the weak classifier
        """
        self.positive_regions = positive_regions
        self.negative_regions = negative_regions
        self.threshold = threshold
        self.polarity = polarity
    
    def classify(self, x):
        """
        Classifies an integral image based on a feature f and the classifiers threshold and polarity
          Args:
            x: A 2D numpy array of shape (m, n) representing the integral image
          Returns:
            1 if polarity * feature(x) < polarity * threshold
            0 otherwise
        """
        feature = lambda ii: sum([pos.compute_feature(ii) for pos in self.positive_regions]) - sum([neg.compute_feature(ii) for neg in self.negative_regions])
        return 1 if self.polarity * feature(x) < self.polarity * self.threshold else 0
    
    def __str__(self):
        return "Weak Clf (threshold=%d, polarity=%d, %s, %s" % (self.threshold, self.polarity, str(self.positive_regions), str(self.negative_regions))

class RectangleRegion:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    
    def compute_feature(self, ii):
        """
        Computes the value of the Rectangle Region given the integral image
        Args:
            integral image : numpy array, shape (m, n)
            x: x coordinate of the upper left corner of the rectangle
            y: y coordinate of the upper left corner of the rectangle
            width: width of the rectangle
            height: height of the rectangle
        """
        return ii[self.y+self.height][self.x+self.width] + ii[self.y][self.x] - (ii[self.y+self.height][self.x]+ii[self.y][self.x+self.width])

    def __str__(self):
        return "(x= %d, y= %d, width= %d, height= %d)" % (self.x, self.y, self.width, self.height)
    def __repr__(self):
        return "RectangleRegion(%d, %d, %d, %d)" % (self.x, self.y, self.width, self.height)
     
# Source: https://github.com/numba/numba/blob/main/numba/cuda/kernels/transpose.py
def transpose(a, b=None):
    """Compute the transpose of 'a' and store it into 'b', if given,
    and return it. If 'b' is not given, allocate a new array
    and return that.
    This implements the algorithm documented in
    http://devblogs.nvidia.com/parallelforall/efficient-matrix-transpose-cuda-cc/
    :param a: an `np.ndarray` or a `DeviceNDArrayBase` subclass. If already on
        the device its stream will be used to perform the transpose (and to copy
        `b` to the device if necessary).
    """

    # prefer `a`'s stream if
    stream = getattr(a, 'stream', 0)

    if not b:
        cols, rows = a.shape
        strides = a.dtype.itemsize * cols, a.dtype.itemsize
        b = cuda.cudadrv.devicearray.DeviceNDArray(
            (rows, cols),
            strides,
            dtype=a.dtype,
            stream=stream)

    dt = nps.from_dtype(a.dtype)

    tpb = driver.get_device().MAX_THREADS_PER_BLOCK
    # we need to factor available threads into x and y axis
    tile_width = int(math.pow(2, math.log(tpb, 2) / 2))
    tile_height = int(tpb / tile_width)

    tile_shape = (tile_height, tile_width + 1)

    @cuda.jit
    def kernel(input, output):

        tile = cuda.shared.array(shape=tile_shape, dtype=dt)

        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        bx = cuda.blockIdx.x * cuda.blockDim.x
        by = cuda.blockIdx.y * cuda.blockDim.y
        x = by + tx
        y = bx + ty

        if by + ty < input.shape[0] and bx + tx < input.shape[1]:
            tile[ty, tx] = input[by + ty, bx + tx]
        cuda.syncthreads()
        if y < output.shape[0] and x < output.shape[1]:
            output[y, x] = tile[tx, ty]

    # one block per tile, plus one for remainders
    blocks = int(b.shape[0] / tile_height + 1), int(b.shape[1] / tile_width + 1)
    # one thread per tile element
    threads = tile_height, tile_width
    kernel[blocks, threads, stream](a, b)

    return b

@cuda.jit
def kernel(s_tab,N,m,sum_tab):
      
      local_id = cuda.threadIdx.x
      blockIdx_x = cuda.blockIdx.x
      blockDimx_x = cuda.blockDim.x
      #global_id = cuda.grid(1)

      # Montée
      for d in range(0,m):
        cuda.syncthreads()
        k = local_id * 2**(d+1) + (blockIdx_x) * (blockDimx_x)
        if (local_id * 2**(d+1)<= N-1):
          s_tab[k+(2**(d+1)-1)]+= s_tab[k+(2**(d)-1)]
  
      cuda.syncthreads()
      if local_id == 0:
        sum_tab[blockIdx_x] = s_tab[((blockIdx_x) * (blockDimx_x))+blockDimx_x-1]
        s_tab[((blockIdx_x) * (blockDimx_x))+blockDimx_x-1]=0
     
      # Descente
      for d in range (m-1,-1,-1) :
          cuda.syncthreads()
          k = local_id * 2**(d+1) + (blockIdx_x) * (blockDimx_x)
          if (local_id * 2**(d+1) <= N-1):
            t = s_tab[k+(2**d)-1]
            s_tab[k+(2**d)-1] = s_tab[k+(2**(d+1))-1]
            s_tab[k+(2**(d+1))-1]+= t
  

def ScanCPU_Special(array,m,N):
    #m=int(math.log2(len(array))
    for d in range(0,m):
        for k in range (0,N-1,(2**(d+1))):
            array[k+(2**(d+1))-1]+= array[k+(2**d)-1]

    array[N-1]=0
    for d in range (m-1,-1,-1) :
        for k in range (0,N-1,2**(d+1)) :
            t = array[k+(2**d)-1]
            array[k+(2**d)-1]= array[k+(2**(d+1))-1]
            array[k+(2**(d+1))-1]+= t
    return array


def scanGPU(p):
  #print("SCAN GPU",p)
  N=len(p) # taille du tableau
  #print(N)
  m= int(log2(N)) # 2^m = N
  threadsPerBlock= 16 #1 seule block par grilles pour les tableau de petites tailles
  blocksPerGrid= math.ceil(N/threadsPerBlock)
  #print("TPB",threadsPerBlock,"BPG",blocksPerGrid)
   
  if(log2(N)%2 != 0):  
    for i in range((N-threadsPerBlock),threadsPerBlock):
       p = np.append(p,0)
  #print(p)
  s_tab = cuda.to_device(p)

  array2 = np.arange(0, blocksPerGrid, 1)
  sum_tab = cuda.to_device(array2)
  kernel[blocksPerGrid,threadsPerBlock](s_tab,threadsPerBlock,int(log2(threadsPerBlock)),sum_tab)
  cuda.synchronize()
  result = s_tab.copy_to_host()
  res_sum_tab = sum_tab.copy_to_host()
  res_copy = result.copy()
  scan_res_sum = ScanCPU_Special(res_sum_tab.copy(),int(log2(len(res_sum_tab))),len(res_sum_tab))

  #print("RES_COPY",res_copy)    
  #print("scan_res_sum",scan_res_sum)

  cpt = 0
  for i in range(0,len(res_copy)):
    if i>0 and i%threadsPerBlock==0:
      #print(i)
      cpt = cpt+1
    temp = res_copy[i] + scan_res_sum[cpt]
    res_copy[i] = temp
  res_copy = res_copy[0:N]
  return res_copy

def integral_image(image):
    """
    Computes the integral image representation of a picture. The integral image is defined as following:
    1. s(x, y) = s(x, y-1) + i(x, y), s(x, -1) = 0
    2. ii(x, y) = ii(x-1, y) + s(x, y), ii(-1, y) = 0
    Where s(x, y) is a cumulative row-sum, ii(x, y) is the integral image, and i(x, y) is the original image.
    The integral image is the sum of all pixels above and left of the current pixel
      Args:
        image : an numpy array with shape (m, n)
    """
    """
    ii = np.zeros(image.shape)
    s = np.zeros(image.shape)
    for y in range(len(image)):
        for x in range(len(image[y])):
            s[y][x] = s[y-1][x] + image[y][x] if y-1 >= 0 else image[y][x]
            ii[y][x] = ii[y][x-1]+s[y][x] if x-1 >= 0 else s[y][x]
    return ii
    """
    new_array = []
    #Pour chaque ligne de la matrice
    for p in image:
      new_array.append(scanGPU(p))
      
      # Si la taille du tableau est inférieur à 32
      """
      if (N<=32):
        result = ScanCPU_Special(array,int(log2(N)),N)
        print("( GPU : ",result, " )")
      else:
      """  
     
    np_arr = np.array(new_array)
    #print("new array",np_arr)

    trspose_arr = transpose(np_arr)
    trspose_arr = trspose_arr.copy_to_host()
    step2_array = []
    #print(trspose_arr)
    
    for p in trspose_arr:
      step2_array.append(scanGPU(p))
    #print(step2_array)
    step2_array = np.array(step2_array)
    new_trspose_arr = transpose(step2_array)
    new_trspose_arr = new_trspose_arr.copy_to_host()
    return new_trspose_arr

"""
t = nb weak classifier
nb_images = nombres d'images à traiter
"""
def bench_train(t):
    with open("drive/MyDrive/FaceDetection/training.pkl", 'rb') as f:
        training = pickle.load(f)
    clf = ViolaJones(T=t)
    clf.train(training, 2429, 4548)
    #evaluate(clf, training)
    #clf.save(str(t))
def test(filename):
    with open("drive/MyDrive/FaceDetection/test.pkl", 'rb') as f:
        test = pickle.load(f)
    clf = ViolaJones.load(filename)
    evaluate(clf, test)
def evaluate(clf, data):
    correct = 0
    for x, y in data:
        correct += 1 if clf.classify(x) == y else 0
    print("Classified %d out of %d test examples" % (correct, len(data)))
bench_train(1)
#test("10")
