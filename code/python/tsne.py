import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from load_dataset import load_classes
import pandas as pd
import numpy as np

base_path = '../../data/ido/proc/'
labels, one_hot_labels, num_graphs, num_classes = load_classes(base_path)
print(labels)
embeddings = pd.read_csv("../../data/ido/embedding/graph_real.csv")
embeddings = embeddings.drop(["id"], axis=1)
embeddings = np.asarray(embeddings)
color_names = plt.cm.rainbow(np.linspace(0, 1, num_classes))
colors = [color_names[label] for label in labels]
print(colors)
reduced_embedding = TSNE(n_components=2).fit_transform(embeddings)
plt.scatter(reduced_embedding[:, 0], reduced_embedding[:, 1], marker='.', c=colors)
plt.show()