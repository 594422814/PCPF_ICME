# Robust Object Tracking Via Part-Based Correlation Particle Filter
Code for the method described in the paper **Robust Object Tracking Via Part-Based Correlation Particle Filter** in ICME 2018.
Paper [Link](https://594422814.github.io/PCPF/PCPF.pdf)


### Prerequisites
To run this tracker, please download the VGG-19 and compile the Matconvnet.
 - The VGG-19 model is available at http://www.vlfeat.org/matconvnet/pretrained/.
 - The Matconvnet is available at https://github.com/vlfeat/matconvnet.
 - The code is mostly in MATLAB, except the workhorse of `fhog.m`, which is written in C and comes from Piotr Dollar toolbox http://vision.ucsd.edu/~pdollar/toolbox
 - gradientMex and mexResize have been compiled and tested for Ubuntu and Windows 8 (64 bit). You can easily recompile the sources in case of need.

### Citing
If you find this work useful in your research, please consider citing:
```
@inproceedings{wang2018robust,
  title={Robust Object Tracking Via Part-Based Correlation Particle Filter},
  author={Wang, Ning and Zhou, Wengang and Li, Houqiang},
  booktitle={2018 IEEE International Conference on Multimedia and Expo (ICME)},
  pages={1--6},
  year={2018},
  organization={IEEE}
}
```

### Acknowledgments
Some codes of this work are adopted from previous trackers (Staple, HCF, MCCT).
- L. Bertinetto, J. Valmadre, S. Golodetz, O. Miksik, and P. Torr. Staple: Complementary learners for real-time tracking. In CVPR, 2016.
- C. Ma, J.-B. Huang, X. Yang, and M.-H. Yang. Hierarchical convolutional features for visual tracking. In ICCV, 2015.
