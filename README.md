# ncnn-deep-head-pose
Simple implement inference deep head pose ncnn version with high performance and optimized resource in inference. This project based on [deep-head-pose](https://github.com/natanielruiz/deep-head-pose) project by **Nataniel Ruiz**. And detail paper in [CVPR Workshop](https://arxiv.org/abs/1710.00925). I use [Retinaface](https://github.com/deepinsight/insightface/tree/master/detection/retinaface) for face detection step.

# Workflow
## Re-build ncnn for arbitrary platform.
  - Official ncnn document shown in detail [how to use and build ncnn](https://github.com/Tencent/ncnn#howto) for arbitrary platform.
  - And if you use my docker environment, i had build ncnn library inside docker environment with path: `/home/ncnn_original/build` it's contain ncnn shared static library and tools for convert and quantize ncnn models.
## Convert models ncnn format.
  - As original deep head pose project used Pytorch framwork. So, We need convert Pytorch model to ncnn model.
  - In ncnn wiki had detailed this guide [here](https://github.com/Tencent/ncnn/wiki/use-ncnn-with-pytorch-or-onnx). After convert pytorch to onnx format, you have to use ncnn build [tools](https://github.com/Tencent/ncnn/tree/master/tools/onnx) to convert onnx->ncnn. Inside my docker env, already to use in `/home/ncnn_original/build/tools/onnx`
  - Notice, [Netron](https://netron.app/) support to visualize network in intuitive easy to get `input node` and `output node` as [specification of ncnn](https://github.com/Tencent/ncnn/wiki/param-and-model-file-structure)
## Build run test
### Run environment
  - `git clone https://github.com/docongminh/ncnn-deep-head-pose`
  - `cd ncnn-deep-head-pose`
  - execute env: `docker exec -it deep-head-pose bash`
  - cd to source mounted: `cd /source`
  - cd to ncnn build library: `cd /home/ncnn_original`
### Cmake project
  - In project root inside docker: `mkdir -p build && cd build
  - Cmake and build: `cmake ..` && `make`
  - Run test: `./main`
 
  - Examples:
   
     ![cr7](https://github.com/docongminh/ncnn-deep-head-pose/blob/master/images/cr7_output.jpg)
     
      
     ![m10](https://github.com/docongminh/ncnn-deep-head-pose/blob/master/images/m10_output.jpg)
     
## Notice
  - This project in progress. So, it have many issues and coding performance during develop process.
## References
  - [deep-head-pose](https://github.com/natanielruiz/deep-head-pose)
  - [retinaface](https://github.com/deepinsight/insightface/tree/master/detection/retinaface)
  - [ncnn](https://github.com/Tencent/ncnn/issues?q=is%3Aissue+is%3Aopen+normalize)
