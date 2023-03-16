# implementation_ml

This project aims to improve the existing implementation of the Viola-Jones face detection algorithm, which runs on CPU, by parallelizing some of its functions on GPU. This enhancement has reduced the processing time compared to the version that only runs on CPU.

The project was developed locally and tested on the Google Colab platform with the GPU acceleration option. It is divided into two parts: parallelization of the integral image calculation phase, followed by parallelization of the function that calculates the features. This particular release only contains the implementation of the parallelized integral image function.

The FaceDetection directory contains an implementation of this algorithm by [Anmol Parande](https://github.com/aparande/FaceDetection) along with a file named ``viola_jones_gpu.py``, which incorporates the modifications required to execute the code with Nvidia CUDA.

The Viola-Jones algorithm, as originally proposed by Viola and Jones in their 2001 paper,
[Viola, Paul, and Michael Jones. "Rapid object detection using a boosted cascade of simple features." Computer Vision and Pattern Recognition, 2001. CVPR 2001. Proceedings of the 2001 IEEE Computer Society Conference on. Vol. 1. IEEE, 2001.](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf)
