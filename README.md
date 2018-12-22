
## Compiling ##
Environment:
Linux
opencv 3.4.0
cuda 9.1
Eigen3


B:two for optical flow (`run_OF_*`) and two for depth from stereo (`run_DE_*`).
For each problem, a fast variant operating on intensity images (`run_*_INT`) and 
a slower variant operating on RGB images (`run_*_RGB`) is provided.
A:only run_DE_RGB


```
cd A(B)
mkdir build
cd build
cmake ../
make -j
```         
      
## Usage ##
The interface for all four binaries (`run_*_*`) is the same.

For depth map, run_DE_INT is faster

test video : test2.avi test2b.avi

./run_DE_INT test2.avi test2b.avi

output: A:background image,background depth
	B:mask,img_mask,depth_mask

Parameters:(can be changed in depthmap.cpp)
```
1. Coarsest scale                               (here: 5)
2. Finest scale                                 (here: 3)
3/4. Min./Max. iterations                       (here: 12)
5./6./7. Early stopping parameters
8. Patch size                                   (here: 8)
9. Patch overlap                                (here: 0.4)
10.Use forward-backward consistency             (here: 0/no)
11.Mean-normalize patches                       (here: 1/yes)
12.Cost function                                (here: 0/L2)  Alternatives: 1/L1, 2/Huber, 10/NCC
13.Use TV refinement                            (here: 1/yes)
14./15./16. TV parameters alpha,gamma,delta     (here 10,10,5)
17. Number of TV outer iterations               (here: 1)
18. Number of TV solver iterations              (here: 3)
19. TV SOR value                                (here: 1.6)
20. Verbosity                                   (here: 2) Alternatives: 0/no output, 1/only flow runtime, 2/total runtime
```


The optical flow output is saves as .flo file.
(http://sintel.is.tue.mpg.de/downloads)

The interface for depth from stereo is exactly the same. The output is saves as pfm file.
(http://vision.middlebury.edu/stereo/code/)


NOTES:
1. For better quality, increase the number iterations (param 3/4), use finer scales (param. 2), higher patch overlap (param. 9), more outer TV iterations (param. 17)
2. L1/Huber cost functions (param. 12) provide better results, but require more iterations (param. 3/4)





The optical flow algorithm we use Fast DIS
`@inproceedings{kroegerECCV2016,
   Author    = {Till Kroeger and Radu Timofte and Dengxin Dai and Luc Van Gool},
   Title     = {Fast Optical Flow using Dense Inverse Search},
   Booktitle = {Proceedings of the European Conference on Computer Vision ({ECCV})},
   Year      = {2016}} `
















