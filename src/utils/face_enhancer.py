import os
import torch 

from gfpgan import GFPGANer

from tqdm import tqdm

from src.utils.videoio import load_video_to_cv2

import cv2
from InferenceUtil import (
    NVTXUtil,
    SynchronizeUtil,
    build_TensorRT_engine_CLI,
    TRT_Engine,
    check_onnx,
    USE_TRT,
    TorchUtil,
)

class GeneratorWithLen(object):
    """ From https://stackoverflow.com/a/7460929 """

    def __init__(self, gen, length):
        self.gen = gen
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        return self.gen

def enhancer_list(images, method='gfpgan', bg_upsampler='realesrgan',
                  mm=None, torchutil=None,
                  restorer=None):
    gen = enhancer_generator_no_len(images, method=method, bg_upsampler=bg_upsampler,
                                    mm=mm, torchutil=torchutil,
                                    restorer=restorer)
    return list(gen)

def enhancer_generator_with_len(images, method='gfpgan', bg_upsampler='realesrgan', mm=None, torchutil=None,
                                restorer=None):
    """ Provide a generator with a __len__ method so that it can passed to functions that
    call len()"""

    if os.path.isfile(images): # handle video to images
        # TODO: Create a generator version of load_video_to_cv2
        images = load_video_to_cv2(images)

    gen = enhancer_generator_no_len(images, method=method, bg_upsampler=bg_upsampler, mm=mm, torchutil=torchutil,
                                    restorer=restorer)
    gen_with_len = GeneratorWithLen(gen, len(images))
    return gen_with_len

def enhancer_generator_no_len(images, method='gfpgan', bg_upsampler='realesrgan',
                              mm=None, torchutil=None,
                              restorer=None):
    """ Provide a generator function so that all of the enhanced images don't need
    to be stored in memory at the same time. This can save tons of RAM compared to
    the enhancer function. """

    assert mm and torchutil
    print('face enhancer....')
    if not isinstance(images, list) and os.path.isfile(images): # handle video to images
        images = load_video_to_cv2(images)

    assert method == 'gfpgan' # dont remove this line, as I move GFPGANer creator to inference.py and use method 'gfpgan'
    assert bg_upsampler is None # dont remove this line, as I move GFPGANer creator to inference.py and use method 'gfpgan'
    if method != 'gfpgan' or bg_upsampler is not None or restorer is None:
        restorer = None # create a new restorer
        # ------------------------ set up GFPGAN restorer ------------------------
        if  method == 'gfpgan':
            arch = 'clean'
            channel_multiplier = 2
            model_name = 'GFPGANv1.4'
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
        elif method == 'RestoreFormer':
            arch = 'RestoreFormer'
            channel_multiplier = 2
            model_name = 'RestoreFormer'
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'
        elif method == 'codeformer': # TODO:
            arch = 'CodeFormer'
            channel_multiplier = 2
            model_name = 'CodeFormer'
            url = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth'
        else:
            raise ValueError(f'Wrong model version {method}.')


        # ------------------------ set up background upsampler ------------------------
        if bg_upsampler == 'realesrgan':
            if not torch.cuda.is_available():  # CPU
                import warnings
                warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                            'If you really want to use it, please modify the corresponding codes.')
                bg_upsampler = None
            else:
                from basicsr.archs.rrdbnet_arch import RRDBNet
                from realesrgan import RealESRGANer
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
                bg_upsampler = RealESRGANer(
                    scale=2,
                    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                    model=model,
                    tile=400,
                    tile_pad=10,
                    pre_pad=0,
                    half=True)  # need to set False in CPU mode
        else:
            bg_upsampler = None

        # determine model paths
        model_path = os.path.join('gfpgan/weights', model_name + '.pth')
        
        if not os.path.isfile(model_path):
            model_path = os.path.join('checkpoints', model_name + '.pth')
        
        if not os.path.isfile(model_path):
            # download pre-trained models from url
            model_path = url

        with NVTXUtil("GFPGANer", "blue", mm), SynchronizeUtil(torchutil.torch_stream):
            restorer = GFPGANer(
                model_path=model_path,
                upscale=2,
                arch=arch,
                channel_multiplier=channel_multiplier,
                bg_upsampler=bg_upsampler)
            
            with NVTXUtil("onnx_face_detector", "blue", mm), SynchronizeUtil(torchutil.torch_stream):
                onnx_model_path = "./onnx_trt/face_detector_bs1_fp32_6000.onnx"
                torch_onnx_export_face_detector(restorer.face_helper.face_det, input_shape=(1,3,1200,800),
                                                            fp16=False, 
                                                            onnx_model_path=onnx_model_path, 
                                                            maxBatch=1 )
            with NVTXUtil("trt_face_detector", "blue", mm), SynchronizeUtil(torchutil.torch_stream):
                detect_faces_trt_engine = None
                trt_engine_path = "./onnx_trt/face_detector_bs1_fp32_6000.engine"
            
                build_TensorRT_engine_CLI( src_onnx_path=onnx_model_path, 
                                        dst_trt_engine_path=trt_engine_path )
            
                if detect_faces_trt_engine is None: 
                    inputs_name = ["source_image",]
                    outputs_name = ["loc", "conf", "landmarks"]
                    detect_faces_trt_engine = TRT_Engine(trt_engine_path, gpu_id=0, torch_stream=torchutil.torch_stream,
                                                        onnx_inputs_name = inputs_name,
                                                        onnx_outputs_name = outputs_name,)
                    assert detect_faces_trt_engine
                    restorer.face_helper.face_det.trt_engine = detect_faces_trt_engine if USE_TRT else None

            with NVTXUtil("onnx_gfpgan", "blue", mm), SynchronizeUtil(torchutil.torch_stream):
                onnx_model_path = "./onnx_trt/gfpgan_bs1_fp32_6000.onnx"
                torch_onnx_export_gfpgan(restorer.gfpgan, 
                                        fp16=False, 
                                        onnx_model_path=onnx_model_path, 
                                        maxBatch=1 )
            with NVTXUtil("trt_gfpgan", "blue", mm), SynchronizeUtil(torchutil.torch_stream):
                gfpgan_trt_engine = None
                trt_engine_path = "./onnx_trt/gfpgan_bs1_fp32_6000.engine"
            
                build_TensorRT_engine_CLI( src_onnx_path=onnx_model_path, 
                                        dst_trt_engine_path=trt_engine_path )
            
                if gfpgan_trt_engine is None: 
                    gfpgan_trt_engine = TRT_Engine(trt_engine_path, gpu_id=0, torch_stream=torchutil.torch_stream)
                    assert gfpgan_trt_engine
                restorer.gfpgan_trt_engine = gfpgan_trt_engine if USE_TRT else None

            with NVTXUtil("onnx_face_parse", "blue", mm), SynchronizeUtil(torchutil.torch_stream):
                onnx_model_path = "./onnx_trt/face_parse_bs1_fp32_6000.onnx"
                torch_onnx_export_face_pase(restorer.face_helper.face_parse, 
                                        fp16=False, 
                                        onnx_model_path=onnx_model_path, 
                                        maxBatch=1 )
            with NVTXUtil("trt_face_parse", "blue", mm), SynchronizeUtil(torchutil.torch_stream):
                face_parse_trt_engine = None
                trt_engine_path = "./onnx_trt/face_parse_bs1_fp32_6000.engine"
            
                build_TensorRT_engine_CLI( src_onnx_path=onnx_model_path, 
                                        dst_trt_engine_path=trt_engine_path )
            
                if face_parse_trt_engine is None: 
                    face_parse_trt_engine = TRT_Engine(trt_engine_path, gpu_id=0, torch_stream=torchutil.torch_stream)
                    assert face_parse_trt_engine
                restorer.face_helper.face_parse_trt_engine = face_parse_trt_engine if USE_TRT else None

    # ------------------------ restore ------------------------
    torchutil_list = []
    for i in range(2):
        torchutil_list.append(TorchUtil(gpu=0, memory_manager=mm, cvcuda_stream=None))
    
    for idx in tqdm(range(len(images)), 'Face Enhancer:'):
        with NVTXUtil(f"cvtColor{idx}", "blue", mm), SynchronizeUtil(torchutil_list[idx%len(torchutil_list)].torch_stream):
            img = cv2.cvtColor(images[idx], cv2.COLOR_RGB2BGR)
        # restore faces and background if necessary
        with NVTXUtil(f"enhance{idx}", "blue", mm), SynchronizeUtil(torchutil_list[idx%len(torchutil_list)].torch_stream, is_in_sync=False, is_out_sync=False):
            with torch.cuda.stream(torchutil_list[idx%len(torchutil_list)].torch_stream):
                assert restorer
                cropped_faces, restored_faces, r_img = restorer.enhance(
                    img,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True,
                    torch_stream=torchutil_list[idx%len(torchutil_list)].torch_stream)
        
        #r_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)
        yield r_img
