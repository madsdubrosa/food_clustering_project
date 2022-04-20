import os
from pprint import pprint

import scipy.cluster.hierarchy as sch
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from random import shuffle
from random import seed
from copy import deepcopy
from sklearn.model_selection import train_test_split
from collections import Counter, defaultdict

# import matplotlib
# matplotlib.use('TkAgg')

seed(12)


def plot_dendrogram():
    fig = pl.figure()
    data = np.array([[1,2,3],[1,1,1],[5,5,5]])
    datalable = ['first','second','third']
    hClsMat = sch.linkage(data, method='complete')
    sch.dendrogram(hClsMat, labels= datalable, leaf_rotation = 45)
    fig.show()
    resultingClusters = sch.fcluster(hClsMat,t= 3, criterion = 'distance')
    print(resultingClusters)


def randIndex(truth, predicted):
    if len(truth) != len(predicted):
        print("different sizes of the label assignments")
        return -1
    elif (len(truth) == 1):
        return 1
    sizeLabel = len(truth)
    agree_same = 0
    disagree_same = 0
    count = 0
    for i in range(sizeLabel-1):
        for j in range(i+1,sizeLabel):
            if ((truth[i] == truth[j]) and (predicted[i] == predicted[j])):
                agree_same += 1
            elif ((truth[i] != truth[j]) and (predicted[i] != predicted[j])):
                disagree_same +=1
            count += 1
    return (agree_same+disagree_same)/float(count)


class FoodGroupData:
    def __init__(self):
        """
        self.data = {"Cereal-grains-pasta": [initial data read in], [initial x], [initial y], [min], [max], [normalized from group min/max], [normalized to all min/max]
                     "Fats-oils": [initial data read in], [initial x], [initial y], [min], [max], [normalized from group min/max], [normalized to all min/max],
                     "Finfish-shellfish": [initial data read in], [initial x], [initial y], [min], [max], [normalized from group min/max], [normalized to all min/max],
                     "Vegetables": [initial data read in], [initial x], [initial y], [min], [max], [normalized from group min/max], [normalized to all min/max],
                     "All": [initial data read in], [initial x], [initial y], [min], [max], [normalized from group min/max], [normalized to all min/max]}
        """
        self.data = {"Cereal-grains-pasta": [[],[],[],[],[],[],[]],
                     "Fats-oils": [[],[],[],[],[],[],[]],
                     "Finfish-shellfish": [[],[],[],[],[],[],[]],
                     "Vegetables": [[],[],[],[],[],[],[]],
                     "All": [[],[],[],[],[],[],[]]}
        self.features = None

    def read_data(self, filename, data_flag):
        if self.data.get(data_flag) is None and data_flag != "descriptions":
            print("Data flag is not supported")
            return

        with open(filename, "r") as fh:
            content = fh.readlines()

        if data_flag == "descriptions":
            self.features = content
            return

        self.data[data_flag][0] = content

        new_data = []
        for key in self.data.keys():
            if key != "All":
                self.data["All"][0] += self.data[key][0]

        return

    def get_initial_data(self, data_flag):
        if self.data.get(data_flag) is None and data_flag != "descriptions":
            print("Data flag is not supported")
            return

        total_vectors = []
        y_data = []

        if data_flag == "descriptions":
            data = self.features
        else:
            data = self.data[data_flag][0]

        for i in range(len(data)):
            vector_content = data[i]
            vector = [item.strip() for item in vector_content.split("^")]
            y_data.append(vector[0])
            total_vectors.append(vector[1:len(vector)-1])

        if data_flag == "descriptions":
            self.features = np.array(total_vectors)
            return

        self.data[data_flag][1] = np.array(total_vectors)
        self.data[data_flag][2] = np.array(y_data)

        self.data[data_flag][1] = self.data[data_flag][1].astype(np.float)

        self.data["All"][1: 3] = [], []
        for key in self.data.keys():
            if key != "All":
                self.data["All"][1].extend(self.data[key][1])
                self.data["All"][2].extend([key] * len(self.data[key][2]))

        self.data["All"][1] = np.array(self.data["All"][1])
        self.data["All"][2] = np.array(self.data["All"][2])

        self.data["All"][1] = self.data["All"][1].astype(np.float)

        return

    def get_min_max(self, data_flag):
        if self.data.get(data_flag) is None:
            print("Data flag is not supported")
            return

        data = self.data[data_flag][1]

        min_features = np.amin(data, axis=0)
        max_features = np.amax(data, axis=0)

        self.data[data_flag][3] = np.array(min_features)
        self.data[data_flag][4] = np.array(max_features)

        return

    def normalize_data(self, data_flag):
        if self.data.get(data_flag) is None:
            print("Data flag is not supported")
            return

        data = self.data[data_flag][1]
        min_features = self.data[data_flag][3]
        max_features = self.data[data_flag][4]

        all_normalized_data = (data - min_features) / (max_features - min_features)

        self.data[data_flag][5] = np.nan_to_num(all_normalized_data)

        min_features2 = self.data["All"][3]
        max_features2 = self.data["All"][4]

        all_normalized_data2 = (data - min_features2) / (max_features2 - min_features2)

        self.data[data_flag][6] = np.nan_to_num(all_normalized_data2)

        return

class FoodCluster:
    def __init__(self, numclusters, x_data, gts):
        self.numclusters = numclusters
        self.x_data = x_data
        self.gts = gts
        self.kmeans = None
        self.random_gt = None
        self.num_gt = None
        self.results_for_20_runs = None
        self.randindex_for_20_runs = None

    def run_kmeans(self):
        kmeans = KMeans(random_state=0, n_clusters=self.numclusters).fit(self.x_data)
        self.kmeans = kmeans
        return

    def randomize_gt(self):
        new_gt = deepcopy(self.gts)
        shuffle(new_gt)
        self.random_gt = new_gt
        return

    def convert_gts_to_num(self):
        classes = {"Vegetables": 0,
                   "Finfish-shellfish": 1,
                   "Fats-oils": 2,
                   "Cereal-grains-pasta": 3}
        self.num_gt = [classes.get(value) for value in self.gts]
        return

    def run_kmeans_twenty_times(self):
        objective_results = []
        randindex_results = []
        for i in range(20):
            kmeans = KMeans(random_state=None, n_init=1, n_clusters=self.numclusters).fit(self.x_data)
            objective_results.append(kmeans.inertia_)
            randindex_results.append(randIndex(self.num_gt, kmeans.labels_))
        self.results_for_20_runs = objective_results
        self.randindex_for_20_runs = randindex_results
        return

    def plot_dendrogram(self, class0data, class1data, class2data, class3data):
        data = []
        datalable = []
        label0 = np.zeros(len(class0data))
        label1 = np.ones(len(class1data))
        label2 = np.ones(len(class2data)).__add__(np.ones(len(class2data)))
        label3 = np.ones(len(class3data)).__add__(np.ones(len(class3data))).__add__(np.ones(len(class3data)))

        x0, x, y0, y = train_test_split(class0data, label0, random_state=12, train_size=(30/len(label0)))
        x1, x, y1, y = train_test_split(class1data, label1, random_state=12, train_size=(30/len(label1)))
        x2, x, y2, y = train_test_split(class2data, label2, random_state=12, train_size=(30/len(label2)))
        x3, x, y3, y = train_test_split(class3data, label3, random_state=12, train_size=(30/len(label3)))

        for i in range(len(x0)):
            data.append(x0[i])
            datalable.append(y0[i])

        for j in range(len(x1)):
            data.append(x1[j])
            datalable.append(y1[j])

        for k in range(len(x2)):
            data.append(x2[k])
            datalable.append(y2[k])

        for l in range(len(x3)):
            data.append(x3[l])
            datalable.append(y3[l])

        # data = np.array([[1, 2, 3], [1, 1, 1], [5, 5, 5]])
        # datalable = ['first', 'second', 'third']
        hClsMat = sch.linkage(data, method='complete')  # Complete clustering
        # sch.dendrogram(hClsMat, labels=datalable, leaf_rotation=45)
        # plt.interactive(False)
        # plt.show()
        resultingClusters = sch.fcluster(hClsMat, t=0.84, criterion='distance')
        print(resultingClusters)
        resultingClusters_randIndex = randIndex(resultingClusters, datalable)
        print(resultingClusters_randIndex)

    def try_diff_k_sizes(self, data, y_data):
        x_data = data
        k_list = [5, 10, 25, 50, 75]

        # k_dict = {}
        name_dict = defaultdict(lambda: [])
        # name_dict = {}
        # k_to_name_dictict = {}
        # for i in range(len(k_list)):
        #     kmeans = KMeans(n_clusters=k_list[i]).fit(x_data)
        #     cluster_count = Counter(kmeans.labels_)
        #     k_dict[k_list[i]] = cluster_count
        #     # for j in range(len(kmeans.labels_)):
        #     #     name_dict[kmeans.labels_[j]].append(y_data[j])
        #     for j in range(len(kmeans.labels_)):
        #         if kmeans.labels_[j] == k_dict[k_list[i]].most_common(1)[0][0]:
        #             name_dict[kmeans.labels_[j]].append(y_data[j])
        #     k_to_name_dict[k_list[i]] = name_dict
        #     name_dict = {}

        # print(k_dict)
        # for i in k_list:
        #     print(f"k={i}; {k_dict[i].most_common(1)}")
        #     if min(k_dict[i].most_common(1)[0][1], 10):
        #         print(k_to_name_dict[i][k_dict[i].most_common(1)[0][0]][:k_dict[i].most_common(1)[0][1]])

        for i in range(len(k_list)):
            kmeans = KMeans(n_clusters=k_list[i]).fit(x_data)
            clusters, cluster_counts = np.unique(kmeans.labels_, return_counts=True)
            max_count = np.argmax(cluster_counts)
            index_list = []
            for j in range(len(kmeans.labels_)):
                if kmeans.labels_[j] == clusters[max_count]:
                    index_list.append(j)
            name_list = [y_data[k] for k in index_list]
            name_dict[k_list[i]] = name_list

        for i in k_list:
            print(f"k={i}; size of largest cluster = {len(name_dict[i])}")
            print(name_dict[i][:min(len(name_dict[i]), 10)])

        return


def main():
    data_path = ""
    files = []
    for file in os.listdir(data_path):
        if file.endswith(".txt"):
            files.append(os.path.join(data_path, file))
    files.sort()
    data_flags = ["Cereal-grains-pasta", "Fats-oils", "Finfish-shellfish", "Vegetables", "All"]

    fgd = FoodGroupData()
    fgd.read_data(files.pop(1), "descriptions")
    # print(fgd.features)
    fgd.get_initial_data("descriptions")
    # print(fgd.features)

    for i in range(len(files)):
        fgd.read_data(files[i], data_flags[i])
    # for i in range(len(data_flags)):
    #     print(fgd.data[data_flags[i]][0][0])

    for flag in data_flags[:-1]:
        fgd.get_initial_data(flag)
    # for i in range(len(data_flags)):
    #     print(fgd.data[data_flags[i]][1][0])

    for flag in data_flags:
        fgd.get_min_max(flag)

    for flag in data_flags:
        fgd.normalize_data(flag)
    # for i in range(len(data_flags)):
    #     print(fgd.data[data_flags[i]][3])
    #     print(fgd.data[data_flags[i]][4])

    # for i in range(len(data_flags)):
    #     print(fgd.data[data_flags[i]][5])

    # print(fgd.data["All"][5])
    # print(fgd.data["All"][2])

    fc = FoodCluster(4, fgd.data["All"][5], fgd.data["All"][2])
    # cd.run_kmeans()
    # for i in range(len(cd.kmeans.labels_)):
    #     print(cd.kmeans.labels_[i])

    # cd.randomize_gt()

    # randomized_gts_randindex = randIndex(cd.gts, cd.random_gt)

    # cd.convert_gts_to_num()

    # kmean_labels_randindex = randIndex(cd.num_gt, cd.kmeans.labels_)

    # print(randomized_gts_randindex)
    # print(kmean_labels_randindex)

    # cd.run_kmeans_twenty_times()
    # print(cd.results_for_20_runs)
    # print(min(cd.results_for_20_runs))
    # print(cd.randindex_for_20_runs)

    # fc.plot_dendrogram(fgd.data["Vegetables"][6], fgd.data["Finfish-shellfish"][6], fgd.data["Fats-oils"][6], fgd.data["Cereal-grains-pasta"][6])

    fc.try_diff_k_sizes(fgd.data["Cereal-grains-pasta"][5], fgd.data["Cereal-grains-pasta"][2])

main()
