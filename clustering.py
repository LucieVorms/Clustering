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
path = 'clustering-benchmark/src/main/resources/datasets/artificial'
databrut = arff.loadarff(open (path + "/smile1.arff" , 'r' ))
datanp = np.array([ [ x [ 0 ] ,x [ 1 ] ] for x in databrut [ 0 ] ])
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
""""
tab_score = np.zeros(20)
for k in range(2,22):
    tps1 = time.time()
    model = cluster.KMeans(n_clusters=k , init='k-means++')
    model.fit(datanp)
    tps2 = time.time()
    labels = model.labels_
    tab_score[k-2] = metrics.silhouette_score(datanp,labels)
    iteration = model.n_iter_
"""
def find_best_score(tab):
    max = tab[0]
    index_max = 2
    for index in range(len(tab)):
        if tab[index] > max :
            max = tab[index]
            index_max = index + 2

    return max,index_max
"""
best_score, best_k = find_best_score(tab_score)



# Tracer la courbe
plt.figure(1)
plt.plot(range(2,22), tab_score, marker='x', linestyle='-', color='b', label='Score silhouette')
# Ajouter des titres et légendes
plt.title("Variation du score silhouette en fonction de k")
plt.xlabel("Nombre de clusters (k)")
plt.xticks(range(2,22))  # Affiche toutes les valeurs de k sur l'axe des abscisses
plt.ylabel("Score silhouette")
plt.legend()
plt.grid()

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

"""
#######################
import scipy . cluster . hierarchy as shc
# Donnees dans datanp
print ("Dendrogramme ’single’ donnees initiales")
linked_mat = shc.linkage (datanp , 'single')
plt.figure (figsize = (12 , 12))
shc.dendrogram (linked_mat, orientation = 'top', distance_sort = 'descending', show_leaf_counts = False)
plt.show()

link = ['single', 'average', 'complete', 'ward', 'linkage']
# set distance_threshold ( 0 ensures we compute the full tree )
for threshold in np.linspace(0.1,0,50):
    for linkage in link:
        tps1 = time.time()
        model = cluster.AgglomerativeClustering (distance_threshold = threshold,linkage = link, n_clusters = None)
        model = model.fit(datanp)
        tps2 = time.time()
        labels = model.labels_
        metrics.silhouette_score(datanp,labels)
        k = model.n_clusters_
        leaves = model.n_leaves_

# set the number of clusters
"""
k = 4
tps1 = time.time ()
model = cluster.AgglomerativeClustering ( linkage = 'single' , n_clusters = k )
model = model.fit(datanp)
tps2 = time.time()
labels = model.labels_
kres = model.n_clusters_
leaves = model.n_leaves_
"""
# Affichage clustering
plt.scatter(f0, f1 , c = labels , s = 8)
plt.title("Resultat du clustering ")
plt.show()
print ("nb clusters = " ,k , " , nb feuilles = " , leaves, " runtime = " ,round((tps2-tps1 )*1000, 2) ,"ms")


