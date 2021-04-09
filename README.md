# Degradation

# Rappel sur Git

# Première chose à faire est de cloner le projet en local

# Dans un terminal, faire :

git clone <lien_a_copier_sur_la_page_du_projet:clone_clone_with_ssh>

# Cette commande va créer un répertoire degradation avec tout le projet dedans

# L'idée est donc de travailler en local puis de soummetre nos modifications via un commit

# Si j'ajoute une fichier par exemple, faire

git add <nom_fichier> (ou * pour tous)

git commit -m "Je commente ce que je vais push !"

git push

# Pour récupérer ce qu'on fait les autres, on fait

git pull

# Pour commit sans faire les jobs (compiler le .tex)

faire [skip CI] dans le commit

# La page web créée est visible [ici](https://fcorset.gricad-pages.univ-grenoble-alpes.fr/degradation/)
