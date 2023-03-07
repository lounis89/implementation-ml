# implementation_ml

L'objectif du projet est d'améliorer l'implémentation existante de Viola-Jones, qui fonctionne sur CPU, en parallélisant certaines fonctions sur GPU. Cette amélioration a permis de réduire le temps de traitement par rapport à la version qui fonctionne uniquement sur CPU.

Le projet a été développé en local puis testé sur la plateforme Google Colab avec l'option accélération GPU. Il est divisé en deux parties : la parallélisation de la phase de calcul de l'image intégrale, puis celle de la fonction qui calcule les caractéristiques. Ce rendu ne contient que l'implémentation de la fonction d'image intégrale en parallèle.

Le dossier FaceDetection contient une implémentation de cet algorithme par [Anmol Parande](https://github.com/aparande/FaceDetection) et un fichier viola_jones_gpu.py ainsi qu'un fichier viola_jones_gpu.py qui intègre les modifications permettant d'exécuter le code avec Nvidia CUDA.

[Viola, Paul, and Michael Jones. "Rapid object detection using a boosted cascade of simple features." Computer Vision and Pattern Recognition, 2001. CVPR 2001. Proceedings of the 2001 IEEE Computer Society Conference on. Vol. 1. IEEE, 2001.](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf)
