import numpy as np
from PIL import Image as im
import matplotlib.pyplot as plt
import sklearn.datasets as datasets

centers = [[0, 0], [2, 2], [-3, 2], [2, -4]]
X, _ = datasets.make_blobs(n_samples=300, cluster_std=1, random_state=0)


# Initialization methods: Random, Farthest First, KMeans++, Manual

# 1: Random: Choose two random points from the data as initial centers
def random_init(data, k):
    # Data: np.array k: int
    return data[np.random.choice(len(data) - 1, size=k, replace=False)]

#2 Farthest First: Choose the two farthest points from the data as initial centers
def farthest_points(data, k):
    # Initialize the list of centers with the first point chosen randomly
    centers = [data[np.random.choice(len(data))]]
    
    # Iteratively choose the next farthest point
    for _ in range(k - 1):
        max_distance = 0
        next_center = None
        
        for point in data:
            # Calculate the minimum distance from this point to any of the chosen centers
            min_distance_to_centers = min(np.linalg.norm(point - center) for center in centers)
            
            # If this distance is greater than the current max distance, update the next center
            if min_distance_to_centers > max_distance:
                max_distance = min_distance_to_centers
                next_center = point
        
        # Add the chosen point to the list of centers
        centers.append(next_center)
    
    return np.array(centers)
    
#3: KMeans++: Choose the first center randomly, then choose the next center as the point with the maximum distance from the previous center
def kmeans_plus_plus(data, k):
    centers = []
    centers.append(data[np.random.choice(len(data) - 1, size=1, replace=False)])
    for i in range(k - 1):
        distances = []
        for j in range(len(data)):
            min_distance = np.inf
            for center in centers:
                distance = np.linalg.norm(data[j] - center)
                if distance < min_distance:
                    min_distance = distance
            distances.append(min_distance)
        next_center = data[np.argmax(distances)]
        centers.append(next_center)
    return np.array(centers)

#4: Manual: Choose the centers manually
def manual_init():
    return np.array([[0, 0], [2, 2], [-3, 2], [2, -4]])

# KMeans Algorithm Calls

def kmeans_random():
    # Make a dataset to pass into the KMeans class
    return 0
    


class KMeans():

    def __init__(self, data, k):
        self.data = data
        self.k = k
        self.assignment = [-1 for _ in range(len(data))]
        self.snaps = []
    
    def snap(self, centers):
        TEMPFILE = "temp.png"

        fig, ax = plt.subplots()
        ax.scatter(X[:, 0], X[:, 1], c=self.assignment)
        ax.scatter(centers[:,0], centers[:, 1], c='r')
        fig.savefig(TEMPFILE)
        plt.close()
        self.snaps.append(im.fromarray(np.asarray(im.open(TEMPFILE))))

    def isunassigned(self, i):
        return self.assignment[i] == -1

    def initialize(self):
        return self.data[np.random.choice(len(self.data) - 1, size=self.k, replace=False)]

    def make_clusters(self, centers):
        for i in range(len(self.assignment)):
            for j in range(self.k):
                if self.isunassigned(i):
                    self.assignment[i] = j
                    dist = self.dist(centers[j], self.data[i])
                else:
                    new_dist = self.dist(centers[j], self.data[i])
                    if new_dist < dist:
                        self.assignment[i] = j
                        dist = new_dist
                    
        
    def compute_centers(self):
        centers = []
        for i in range(self.k):
            cluster = []
            for j in range(len(self.assignment)):
                if self.assignment[j] == i:
                    cluster.append(self.data[j])
            centers.append(np.mean(np.array(cluster), axis=0))

        return np.array(centers)
    
    def unassign(self):
        self.assignment = [-1 for _ in range(len(self.data))]

    def are_diff(self, centers, new_centers):
        for i in range(self.k):
            if self.dist(centers[i], new_centers[i]) != 0:
                return True
        return False

    def dist(self, x, y):
        # Euclidean distance
        return sum((x - y)**2) ** (1/2)

    def lloyds(self):
        centers = self.initialize()
        self.make_clusters(centers)
        new_centers = self.compute_centers()
        self.snap(new_centers)
        while self.are_diff(centers, new_centers):
            self.unassign()
            centers = new_centers
            self.make_clusters(centers)
            new_centers = self.compute_centers()
            self.snap(new_centers)
        return







kmeans = KMeans(X, 4)
kmeans.lloyds()
images = kmeans.snaps

images[0].save(
    'kmeans.gif',
    optimize=False,
    save_all=True,
    append_images=images[1:],
    loop=0,
    duration=500
)