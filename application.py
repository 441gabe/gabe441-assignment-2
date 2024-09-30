from flask import Flask, jsonify, render_template, request
from sklearn import datasets

app = Flask(__name__)

centers = [[0, 0], [2, 2], [-3, 2], [2, -4]]
X, _ = datasets.make_blobs(n_samples=300, centers=centers, cluster_std=1, random_state=0)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_kmeans', methods=['POST'])
def process_kmeans():
    init_method = request.form['init-method']
    
    if init_method == 'random':
        kmeans = kmeans.KMeans(X, 4, init_method=kmeans.random_init)
    elif init_method == 'farthest-first':
        kmeans = kmeans.KMeans(X, 4, init_method=kmeans.farthest_points)
    elif init_method == 'kmeans++':
        kmeans = kmeans.KMeans(X, 4, init_method=kmeans.kmeans_plus_plus)
    elif init_method == 'manual':
        kmeans = kmeans.KMeans(X, 4, init_method=kmeans.manual_init)
    
    
    kmeans = kmeans.KMeans(X, 4)
    return jsonify(kmeans)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)