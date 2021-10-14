# Matrix Completion via PnP-ADMM with Low Rank Regularizer

En la actualidad una amplia gama de información es organizada en forma matricial. Un problema que se puede generar en diferentes aplicaciones como la adquisición de imágenes o los sistemas de recomendación es el problema de adquirir una matriz parcialmente observada donde se desean recuperar la información no registrada.

Para este tipo de problemas nace el estudio de técnicas de matrix completion. Estas técnicas buscan completar elementos faltantes en una matriz basado en la información disponible e información previa que se tiene de la matriz.

Entre los métodos para esta tarea se puede destacar los métodos basados en optimización convexa, los cuales hacen uso de cierta información previa de la matriz a reconstruir para condicionar el problema de recuperación. Un principio muy utilizado por este enfoque es el principio de bajo rango, el cual condiciona el rango de la matriz para recuperar la información faltante.

Una limitación de usar un enfoque de bajo rango es que estas se basan en las relaciones globales de la matriz, pero no tiene en cuenta relaciones específicas entre los elementos. Para superar esta limitación, se plantea el uso de una estrategia de Plug-n-Play con un algoritmo no local adicionado a este regularizador de bajo rango para resolver el problema de matrix completion.
