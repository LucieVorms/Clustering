import numpy as np
import matplotlib . pyplot as plt
from scipy . io import arff
import time
from sklearn import cluster, metrics

# Parser un fichier de donnees au format arff
# data est un tableau d ’ exemples avec pour chacun
# la liste des valeurs des features
#
# Dans les jeux de donnees consideres :
# il y a 2 features ( dimension 2 )
# Ex : [[ - 0 . 499261 , -0 . 0612356 ] ,
# [ - 1 . 51369 , 0 . 265446 ] ,
# [ - 1 . 60321 , 0 . 362039 ] , .....
# ]
#
# Note : chaque exemple du jeu de donnees contient aussi un
# numero de cluster . On retire cette information
#path = 'clustering-benchmark/src/main/resources/datasets/artificial'
path = 'dataset-rapport/'
databrut = np.loadtxt(open (path + "zz2.txt" , 'r' ))
datanp = np.array([ [ x [ 0 ] ,x [ 1 ] ] for x in databrut])
# Affichage en 2D
# Extraire chaque valeur de features pour en faire une liste
# Ex pour f0 = [ - 0 . 499261 , -1 . 51369 , -1 . 60321 , ...]
# Ex pour f1 = [ - 0 . 0612356 , 0 . 265446 , 0 . 362039 , ...]
f0 = datanp[ : ,0 ] # tous les elements de la premiere colonne
f1 = datanp[ : ,1 ] # tous les elements de la deuxieme colonne
plt.scatter( f0 , f1 , s = 8 )
plt.title( " Donnees initiales " )
plt.show()


########################EX2############################################""

tab_score = np.zeros(20)
for k in range(2,22):
    tps1 = time.time()
    model = cluster.KMeans(n_clusters=k , init='k-means++')
    model.fit(datanp)
    tps2 = time.time()
    labels = model.labels_
    tab_score[k-2] = metrics.silhouette_score(datanp,labels)
    iteration = model.n_iter_

def find_best_score(tab):
    max = tab[0]
    index_max = 2
    for index in range(len(tab)):
        if tab[index] > max :
            max = tab[index]
            index_max = index + 2

    return max,index_max

best_score, best_k = find_best_score(tab_score)

########## Courbe score ########
plt.figure(1)
plt.plot(range(2,22), tab_score, marker='x', linestyle='-', color='b', label='Score silhouette')
#Ajouter des titres et légendes
plt.title("Variation du score silhouette en fonction de k")
plt.xlabel("Nombre de clusters (k)")
plt.xticks(range(2,22))  # Affiche toutes les valeurs de k sur l'axe des abscisses
plt.ylabel("Score silhouette")
plt.legend()
plt.grid()

########## Affichage clustering final ########
plt.figure(2)
tps1 = time.time()
model = cluster.KMeans(n_clusters=best_k , init='k-means++')
model.fit(datanp)
tps2 = time.time()
labels = model.labels_
iteration = model.n_iter_
plt.scatter(f0, f1 , c = labels , s = 8)
plt.title(" Donnees apres clustering Kmeans")
plt.show()
print("Score : ", best_score)
print("nb clusters = " ,best_k , " , nb iter = " , iteration , " , ..., runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )


#######################EX3########################
############## THRESHOLD ###############
"""
import scipy . cluster . hierarchy as shc
# Donnees dans datanp
print ("Dendrogramme ’single’ donnees initiales")
linked_mat = shc.linkage (datanp , 'single')
plt.figure (figsize = (12 , 12))
shc.dendrogram (linked_mat, orientation = 'top', distance_sort = 'descending', show_leaf_counts = False)
plt.show()

link = ['single', 'average', 'complete', 'ward']
agglo_mat_score = np.zeros((len(np.linspace(0.01, 0.2, 20)), len(link)))
# set distance_threshold ( 0 ensures we compute the full tree )
for i, linkage in enumerate(link):
    for j, threshold in enumerate(np.linspace(0.01, 0.2, 20)):
        tps1 = time.time()
        model = cluster.AgglomerativeClustering (distance_threshold = threshold,linkage = linkage, n_clusters = None)
        model = model.fit(datanp)
        tps2 = time.time()
        labels = model.labels_
        k = model.n_clusters_        
        leaves = model.n_leaves_
        if k > 1 :
            agglo_mat_score[j,i]= metrics.silhouette_score(datanp,labels)
            #agglo_mat_score[j,i]= metrics.calinski_harabasz_score(datanp,labels)

    plt.plot(np.linspace(0.01,0.2,20), agglo_mat_score[:,i], marker='o', label=f'Linkage: {linkage}')

# Ajout des détails du graphique
plt.title("Scores silhouette en fonction des thresholds")
plt.xlabel("Threshold")
plt.ylabel("Score silhouette")
plt.legend()
plt.grid()
plt.show()


def agglo_find_best_score(matrix):
    max_val = matrix[0][0]  # On initialise max_val avec la première valeur de la matrice
    index_max = (0, 0)  # On initialise index_max avec la position (0, 0)

    # On parcourt chaque ligne de la matrice
    for i in range(len(matrix)):
        # On parcourt chaque colonne de la ligne
        for j in range(len(matrix[i])):
            if matrix[i][j] > max_val:
                max_val = matrix[i][j]
                index_max = (i, j)

    return max_val, index_max

def agglo_find_best_score_1D(matrix):
    max_val = matrix[0]  # On initialise max_val avec la première valeur de la matrice
    index_max = 0  # On initialise index_max avec la position 0

    # On parcourt chaque valeur du tableau
    for i in range(len(matrix)):
        if matrix[i] > max_val:
            max_val = matrix[i]
            index_max = i

    return max_val, index_max


agglo_best_score_single, agglo_best_k_single = agglo_find_best_score_1D(agglo_mat_score[:,0])
agglo_best_score_average, agglo_best_k_average = agglo_find_best_score_1D(agglo_mat_score[:,1])
agglo_best_score_complet, agglo_best_k_complete = agglo_find_best_score_1D(agglo_mat_score[:,2])
agglo_best_score_ward, agglo_best_k_ward = agglo_find_best_score_1D(agglo_mat_score[:,3])


############## NB CLUSTER ###############

tab_score_agglo= np.zeros(20)
for k in range(2,22):
    tps1 = time.time ()
    model = cluster.AgglomerativeClustering ( linkage = 'average' , n_clusters = k )
    model = model.fit(datanp)
    tps2 = time.time()
    labels = model.labels_
    kres = model.n_clusters_
    leaves = model.n_leaves_
    tab_score_agglo[k-2] = metrics.silhouette_score(datanp,labels)

best_score_agglo, best_k_agglo = find_best_score(tab_score_agglo)

plt.figure(1)
plt.plot(range(2,22), tab_score_agglo, marker='x', linestyle='-', color='b', label='Score silhouette')

plt.title("Variation du score silhouette en fonction de k")
plt.xlabel("Nombre de clusters (k)")
plt.xticks(range(2,22))  # Affiche toutes les valeurs de k sur l'axe des abscisses
plt.ylabel("Score silhouette")
plt.legend()
plt.grid()

########################## Affichage clustering ###############################
#### NB CLUSTER ########
plt.figure(2)
tps1 = time.time ()
model = cluster.AgglomerativeClustering ( linkage = 'average' , n_clusters = best_k_agglo )
model = model.fit(datanp)
tps2 = time.time()
labels = model.labels_
kres = model.n_clusters_
leaves = model.n_leaves_
plt.scatter(f0, f1 , c = labels , s = 8)
plt.title("Resultat du clustering ")
plt.show()
print ("nb clusters = " ,best_k_agglo , " , nb feuilles = " , leaves, " runtime = " ,round((tps2-tps1 )*1000, 2) ,"ms")
print("Score : ", best_score_agglo)

#### THRESHOLD ########
tps1 = time.time()
model = cluster.AgglomerativeClustering (distance_threshold = agglo_best_k_single,linkage = 'single', n_clusters = None)
model = model.fit(datanp)
tps2 = time.time()
labels = model.labels_
k = model.n_clusters_        
leaves = model.n_leaves_
plt.scatter(f0, f1 , c = labels , s = 8)
plt.title("Resultat du clustering ")
plt.show()
print ("nb clusters = " ,k , " , nb feuilles = " , leaves, " runtime = " ,round((tps2-tps1 )*1000, 2) ,"ms")
print("Score : ", agglo_best_score_single)

tps1 = time.time()
model = cluster.AgglomerativeClustering (distance_threshold = agglo_best_k_average,linkage = 'average', n_clusters = None)
model = model.fit(datanp)
tps2 = time.time()
labels = model.labels_
k = model.n_clusters_        
leaves = model.n_leaves_
plt.scatter(f0, f1 , c = labels , s = 8)
plt.title("Resultat du clustering ")
plt.show()
print ("nb clusters = " ,k , " , nb feuilles = " , leaves, " runtime = " ,round((tps2-tps1 )*1000, 2) ,"ms")
print("Score : ", agglo_best_score_average)

tps1 = time.time()
model = cluster.AgglomerativeClustering (distance_threshold = agglo_best_k_complete,linkage = 'complete', n_clusters = None)
model = model.fit(datanp)
tps2 = time.time()
labels = model.labels_
k = model.n_clusters_        
leaves = model.n_leaves_
plt.scatter(f0, f1 , c = labels , s = 8)
plt.title("Resultat du clustering ")
plt.show()
print ("nb clusters = " ,k , " , nb feuilles = " , leaves, " runtime = " ,round((tps2-tps1 )*1000, 2) ,"ms")
print("Score : ", agglo_best_score_complet)

tps1 = time.time()
model = cluster.AgglomerativeClustering (distance_threshold = agglo_best_k_ward,linkage = 'ward', n_clusters = None)
model = model.fit(datanp)
tps2 = time.time()
labels = model.labels_
k = model.n_clusters_        
leaves = model.n_leaves_
plt.scatter(f0, f1 , c = labels , s = 8)
plt.title("Resultat du clustering ")
plt.show()
print ("nb clusters = " ,k , " , nb feuilles = " , leaves, " runtime = " ,round((tps2-tps1 )*1000, 2) ,"ms")
print("Score : ", agglo_best_score_ward)
"""