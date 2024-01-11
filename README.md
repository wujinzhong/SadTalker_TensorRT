# SadTalker_TensorRT
This repo is my inference performance optimizing oriented NVIDIA CUDA GPUs. For SadTalker's tech background, you can goto its official repo at here for more information, https://github.com/OpenTalker/SadTalker/tree/main. Many thanks to OpenTalkers sharing such good tech open source.

## system config

>git clone https://github.com/wujinzhong/SadTalker_TensorRT
>
>follow instructions to set up the system configuration(download pretrained models, etc) at here, https://github.com/OpenTalker/SadTalker/tree/main.

Here I install facexlib and GFPGAN from source code, as we will modify its source code for TensorRT optimization some of it's AI models.
>
>mkdir onnx_trt
>copy TensorRT engines from google drive, . Put the files in ./onnx_trt.
>clear && python inference.py --driven_audio ./examples/driven_audio/bus_chinese.wav --source_image examples/source_image/full_body_1.png --enhancer gfpgan --result_dir ./results/ --still --preprocess full

The results is at ./results folder.

### Install CV-CUDA
CV-CUDA is a good performance optimizing tool for pre/post-processing of CV pipelines.

Download CV-CUDA's files from here, .
> cd /cv-cuda/
>
> tar -xvf nvcv-lib-0.4.0_beta-cuda11-x86_64-linux.tar.xz
>
> tar -xvf nvcv-dev-0.4.0_beta-cuda11-x86_64-linux.tar.xz
>
> tar -xvf nvcv-python3.8-0.4.0_beta-cuda11-x86_64-linux.tar.xz
>
> export LD_LIBRARY_PATH=/cv-cuda/opt/nvidia/cvcuda0/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH
> export PYTHONPATH=/cv-cuda/opt/nvidia/cvcuda0/lib/x86_64-linux-gnu/python/:$PYTHONPATH

### download TensorRT engines for v8.6.0
goto my google drive at here, https://drive.google.com/drive/folders/1TWh6VLRer9wzh5VO4tH0ebzAydkq7IRL?usp=sharing. Download all the files and put in the right folders.

### SeamlessClone CUDA optimized Ubuntu .so lib for python wrapping usage
Goto my repo at here, https://github.com/wujinzhong/seamlessCloneOptimization. Follow the instructions to build seamlessclone_cuda.so and SeamlessClone.so. cp them to ./SadTalker foler root folder.

## Optimization scheme
Here are some of our optimizing schemes, we will update the doc later.

>Converting detect face AI model to TensorRT
>
>Converting FAN to TensorRT
>
>Move ops to initialization
>
>Enhancer stage
>
>Converting GFPGan to TensorRT
>
>Converting ParseNet to TensorRT
>
>Convert numpy/OpenCV operators to torch/CV-CUDA
>
>Cv2.gaussianblur to torchvision gaussianblur
>
>cv2.warpAffine to CV-CUDA
>
>cv2.resize to cv-cuda
>
>Cuda->tensor.cpu().numpy() interations pattern
>
>SeamlessClone stage
>
>Re-implementing cv2.seamlessClone with CUDA
>

### Something about TRT optimizing of animate_from_coeff.generator AI model

We tried adding a gridsample TRT plugin for animate_from_coeff.generator, to make it can be saved via torch.onnx.export to ONNX and built TRT engine successfully. But after that, we found the accuracy decrease a bit, resulting in the generated video having almost no movement at lip area. I digged into it, and found that there are some differences in the output tensor between this TRT converted engine to the torch implemented counterpart. So I just uploaded it to GitHub and commented it, just in case you just want to test the TRT conversion performance. 

You can open it by find "animate_from_coeff.generator_trt_engine = None" line in inference.py and comment it.

Even without this, our optimization still gains about 4.5X (RTX6000) performance speed up. Hope this can help you too.

