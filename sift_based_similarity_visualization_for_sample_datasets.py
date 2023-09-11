import os
import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from sklearn.metrics.pairwise import cosine_similarity
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

def measure_similarity_homography_feature(image1_path, image2_path, feature_type='sift', threshold=0.75, min_inliers=10):
    # Load images
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    # Choose feature detector
    if feature_type == 'sift':
        detector = cv2.SIFT_create()
    elif feature_type == 'surf':
        detector = cv2.SURF_create()
    elif feature_type == 'orb':
        detector = cv2.ORB_create()
    else:
        raise ValueError("Invalid feature type. Choose 'sift', 'surf', or 'orb'.")

    # Detect and compute keypoints and descriptors
    keypoints1, descriptors1 = detector.detectAndCompute(image1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(image2, None)

    # Initialize a BFMatcher
    bf = cv2.BFMatcher()

    # Match descriptors
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test and compute confidence scores for matches
    good_matches = []
    confidence_scores = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good_matches.append(m)
            confidence_scores.append(1 - m.distance / n.distance)

    # Check if enough good matches are found for homography estimation
    if len(good_matches) >= min_inliers:
        # Extract matched keypoints
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Estimate homography matrix using RANSAC
        homography_matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Calculate the number of inliers
        inlier_count = np.sum(_)

        # Calculate a weighted similarity score based on confidence scores and inlier ratio
        weighted_similarity_score = (inlier_count / len(good_matches)) * np.mean(confidence_scores)
    else:
        weighted_similarity_score = 0.0

    return weighted_similarity_score

def main():
    sampled_data = load_sampled_data()

    # Initialize an empty similarity matrix
    
    sift_feature_based_similarity_matrix = np.zeros((len(sampled_data), len(sampled_data)))
    for i_query in range(len(sampled_data)):
        for i_ref in range(len(sampled_data)):
            if i_query != i_ref:  # Exclude self-similarity
                # Load the images (replace these paths with your image file paths)
                image1_path = 'poster_dataset/' + sampled_data[i_query]
                image2_path = 'poster_dataset/' + sampled_data[i_ref]
                # Calculate SSIM
                sift_feature_based_similarity_matrix[i_query,i_ref] = measure_similarity_homography_feature(image1_path, image2_path, feature_type='sift')

    with open('outputs/sift_feature_based_similarity_matrix.txt','wb') as txt_file:
        for line in np.matrix(sift_feature_based_similarity_matrix):
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
                similarity = sift_feature_based_similarity_matrix[i_query, i_ref]
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
    plt.savefig("outputs/features_based.pdf", format="pdf", bbox_inches="tight")
    
if __name__ == "__main__":
    main()
