import os
import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from resnet50 import ResNet50
from keras.layers import Input
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity as ssim
from skimage import io, color, transform

# Dataset and its sampled one for computing time #
def load_sampled_data():
    # Load images in the images folder into an array
    cwd_path = os.getcwd()
    data_path = cwd_path + '/poster_dataset'
    data_dir_list = os.listdir(data_path)
    data_dir_list.sort()
    sampled_data_index = filter(lambda x: x % 150 == 0, range(len(data_dir_list)))
    sampled_data = list(map(lambda i: data_dir_list[i], sampled_data_index))
    print("Original poster dataset has %d images, but the sampled dataset has only %d images" % (len(data_dir_list), len(sampled_data)))
    return sampled_data

def get_feature_vector_fromPIL(img):
    feature_vector = feature_model.predict(img)
    a, b, c, n = feature_vector.shape
    feature_vector= feature_vector.reshape(b,n)
    return feature_vector

def calculate_similarity_cosine(vector1, vector2):
 return cosine_similarity(vector1, vector2)

# This distance can be in range of [0,∞]. And this distance is converted to a [0,1]
def calculate_similarity_euclidean(vector1, vector2):
 return 1/(1 + np.linalg.norm(vector1- vector2))  

def main():
    sampled_data = load_sampled_data()

    #Use ResNet-50 model as an image feature extractor
    image_input = Input(shape=(224, 224, 3))
    feature_model = ResNet50(input_tensor=image_input, include_top=False,weights='imagenet')
    img_data_list=[]
    for image_name in sampled_data:
            img_path = 'poster_dataset/' + image_name
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            img_data_list.append(x)

    resnet50_euclidean_similarity_matrix = np.zeros((len(sampled_data), len(sample_data)))
    resnet50_cosine_similarity_matrix = np.zeros((len(sampled_data), len(sample_data)))
    for i_query in range(len(sampled_data)):
        for i_ref in range(len(sampled_data)):
            if i_query != i_ref:  # Exclude self-similarity
                feature_vector_1 = get_feature_vector_fromPIL(img_data_list[i_query])
                feature_vector_2 = get_feature_vector_fromPIL(img_data_list[i_ref])
                resnet50_euclidean_similarity_matrix[i_query][i_ref] = calculate_similarity_euclidean(feature_vector_1, feature_vector_2)
                resnet50_cosine_similarity_matrix[i_query][i_ref] = calculate_similarity_cosine(feature_vector_1, feature_vector_2)

    with open('outputs/resnet50_euclidean_similarity_matrix.txt','wb') as txt_file:
        for line in np.matrix(resnet50_euclidean_similarity_matrix):
            np.savetxt(txt_file, line, fmt='%.4f')

    with open('outputs/resnet50_cosine_similarity_matrix.txt','wb') as txt_file:
        for line in np.matrix(resnet50_cosine_similarity_matrix):
            np.savetxt(txt_file, line, fmt='%.4f')

    # Create a graph using networkx
    G = nx.Graph()

    # Load thumbnail images and add them as nodes
    thumbnail_size = (200, 200)  # Adjust the size of the thumbnails as needed
    for i, image_filename in enumerate(sampled_data):
        image = io.imread('poster_dataset/' + image_filename)
        thumbnail = transform.resize(image, thumbnail_size, mode='reflect')
        G.add_node(i, image=thumbnail)

    # Add edges based on the similarity matrix
    for i_query in range(len(sampled_data)):
        for i_ref in range(len(sampled_data)):
            if i_query != i_ref:  # Exclude self-similarity
                similarity = resnet50_cosine_similarity_matrix[i_query, i_ref]
                G.add_edge(i_query, i_ref, weight=similarity)

    # Draw the graph with thumbnail images as nodes
    pos = nx.spring_layout(G, dim=2)  # 2D layout

    plt.figure(figsize=(12.5,10 ))  # Adjust the figure size as needed

    # Draw nodes with images
    for node, (x, y) in pos.items():
        image = G.nodes[node]["image"]
        plt.imshow(image, extent=[x - 0.1, x + 0.1, y - 0.1, y + 0.1], aspect="auto")

    # Draw edges
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        plt.plot([x0, x1], [y0, y1], 'k-', alpha=0.025, lw=2)

    plt.axis("off")
    plt.savefig("outputs/deep_learning_based.pdf", format="pdf", bbox_inches="tight")

if __name__ == "__main__":
    main()
