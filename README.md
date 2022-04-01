# implementation_ml

Ce projet a été mené durant le dernier semestre de master 2 MIAGE IA. Le but étant de paralléliser sur GPU certaines fonctions dans le but d'améliorer une implémentation existante de Viola-Jones qui s'exécute sur CPU. Cette amélioration apporte un temps de traitement inférieur par rapport à la version s’exécutant uniquement sur CPU.

Le projet a été développé et testé sur la plateforme Google Colab. Le projet se décompose en deux parties : la parallélisation de la phase de calcul d’image intégral et puis celle de la fonction qui calcul les features. Dans ce rendu, l'implémentation ne comporte que la fonction image intégrale en parallèle.
