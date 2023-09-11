import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from skimage.metrics import structural_similarity as ssim
from skimage import io, transform

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


def main():
    sampled_data = load_sampled_data()

    # Initialize an empty similarity matrix
    ssim_similarity_matrix = np.zeros((len(sampled_data), len(sampled_data)))

    # Loop through each pair of images
    for i_query, query_image_filename in enumerate(sampled_data):
        for i_ref, ref_image_filename in enumerate(sampled_data):
            if i_query != i_ref:  # Exclude self-similarity
                # Load the images
                image1 = io.imread('poster_dataset/' + query_image_filename, as_gray=True)
                image2 = io.imread('poster_dataset/' + ref_image_filename, as_gray=True)

                # Calculate SSIM
                ssim_value = ssim(image1, image2, data_range=image1.max() - image1.min())

                # Store the SSIM value in the similarity matrix
                ssim_similarity_matrix[i_query, i_ref] = ssim_value

    with open('outputs/ssim_similarity_matrix.txt','wb') as txt_file:
        for line in np.matrix(ssim_similarity_matrix):
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
                similarity = ssim_similarity_matrix[i_query, i_ref]
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
    plt.savefig("outputs/pixel_based.pdf", format="pdf", bbox_inches="tight")

if __name__ == "__main__":
    main()
