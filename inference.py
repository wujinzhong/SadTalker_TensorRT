from glob import glob
import shutil
import torch
from time import  strftime
import os, sys, time
from argparse import ArgumentParser

from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff  
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path
from src.face3d.visualize import gen_composed_video
        
from InferenceUtil import (
    Memory_Manager,
    TorchUtil,
    NVTXUtil,
    SynchronizeUtil,
    check_onnx,
    build_TensorRT_engine_CLI,
    TRT_Engine,
    USE_TRT,
    USE_WARM_UP
)

import cv2
from SeamlessClone import SeamlessClone
import numpy as np

from gfpgan import GFPGANer

#torch.onnx.export template
def torch_onnx_export_animate_from_coeff_generator(onnx_model, fp16=False, onnx_model_path="model.onnx", maxBatch=1 ):
    if not os.path.exists(onnx_model_path):
        dynamic_axes = {
            "source_image":             {0: "batch_size"},
            "kp_driving_value":         {0: "batch_size"},
            "kp_source_value":          {0: "batch_size"}
        }

        device = torch.device("cuda:0")
        
        onnx_model2= onnx_model #onnx_model2= UNet_x(onnx_model)
        if isinstance(onnx_model2, torch.nn.DataParallel):
            onnx_model2 = onnx_model2.module

        onnx_model2.eval()
        onnx_model2 = onnx_model2.to(device=device)
        
        if fp16: dst_dtype = torch.float16
        else: dst_dtype = torch.float32

        '''
        source_image: (torch.Size([2, 3, 256, 256]), device(type='cuda', index=0), torch.float32)
        kp_source['value']: (torch.Size([2, 15, 3]), device(type='cuda', index=0), torch.float32)
        kp_norm['value']: (torch.Size([2, 15, 3]), device(type='cuda', index=0), torch.float32)
        out_mask: (torch.Size([2, 16, 16, 64, 64]), device(type='cuda', index=0), torch.float32)
        out_occlusion_map: (torch.Size([2, 1, 64, 64]), device(type='cuda', index=0), torch.float32)
        out_prediction: (torch.Size([2, 3, 256, 256]), device(type='cuda', index=0), torch.float32)

        source_image: (torch.Size([2, 3, 256, 256]), device(type='cuda', index=0), torch.float32)
        kp_source[value]: (torch.Size([2, 15, 3]), device(type='cuda', index=0), torch.float32)
        kp_norm[value]: (torch.Size([2, 15, 3]), device(type='cuda', index=0), torch.float32)
        out[mask]: (torch.Size([2, 16, 16, 64, 64]), device(type='cuda', index=0), torch.float32)
        out[occlusion_map]: (torch.Size([2, 1, 64, 64]), device(type='cuda', index=0), torch.float32)
        out[prediction]: (torch.Size([2, 3, 256, 256]), device(type='cuda', index=0), torch.float32)
        '''
        dummy_inputs = {
            "source_image": torch.randn((2, 3, 256, 256), dtype=dst_dtype).to(device).contiguous(),
            "kp_source_value": torch.randn((2, 15, 3), dtype=dst_dtype).to(device).contiguous(),
            'kp_norm_value': torch.randn((2, 15, 3), dtype=dst_dtype).to(device).contiguous(),
        }
        output_names = ["mask", "occlusion_map", "prediction"]

        #import apex
        with torch.no_grad():
            #with warnings.catch_warnings():
            if True:
                #warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
                #warnings.filterwarnings("ignore", category=UserWarning)
                if True:
                    torch.onnx.export(
                        onnx_model,
                        tuple(dummy_inputs.values()),
                        onnx_model_path, #f,
                        export_params=True,
                        verbose=True,
                        opset_version=16,
                        do_constant_folding=False,
                        input_names=list(dummy_inputs.keys()),
                        output_names=output_names,
                        #dynamic_axes=dynamic_axes,
                    )  
                else:
                    with open(onnx_model_path, "wb") as f:
                        torch.onnx.export(
                        onnx_model,
                        tuple(dummy_inputs.values()),
                        f,
                        export_params=True,
                        verbose=True,
                        opset_version=18,
                        do_constant_folding=False,
                        input_names=list(dummy_inputs.keys()),
                        output_names=output_names,
                        #dynamic_axes=dynamic_axes,
                        )  
    check_onnx(onnx_model_path)
    onnx_model.to('cpu')
    return

def modify_onnx_animate_from_coeff_generator(onnx_model_path_modified, onnx_model_path):
    import onnx
    import onnx_graphsurgeon as gs

    model = onnx.load(onnx_model_path)
    graph = gs.import_onnx(model)
    for node in graph.nodes:
        if "GridSample" in node.name:
            print(f"node funnd: {node}")
            node.attrs = {"name": "GridSample3D", "version": 1, "namespace": ""}
            node.op = "GridSample3D"

    onnx.save(gs.export_onnx(graph), onnx_model_path_modified)

def torch_onnx_export_fan(onnx_model, fp16=False, onnx_model_path="model.onnx", maxBatch=1 ):
    if not os.path.exists(onnx_model_path):
        dynamic_axes = {
            "latent_model_input":   {0: "bs_x_2"},
            "prompt_embeds":        {0: "bs_x_2"},
            "noise_pred":           {0: "batch_size"}
        }

        device = torch.device("cuda:0")
        
        onnx_model2= onnx_model #onnx_model2= UNet_x(onnx_model)
        if isinstance(onnx_model2, torch.nn.DataParallel):
            onnx_model2 = onnx_model2.module

        onnx_model2.eval()
        onnx_model2 = onnx_model2.to(device=device)
        
        if fp16: dst_dtype = torch.float16
        else: dst_dtype = torch.float32

        '''
        inp: (torch.Size([1, 3, 256, 256]), device(type='cuda', index=0), torch.float32)
        output: (torch.Size([1, 99, 64, 64]), device(type='cuda', index=0), torch.float32)
        '''
        dummy_inputs = {
            "inp": torch.randn((1, 3, 256, 256), dtype=dst_dtype).to(device).contiguous(),
        }
        output_names = ["output"]

        #import apex
        with torch.no_grad():
            #with warnings.catch_warnings():
            if True:
                #warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
                #warnings.filterwarnings("ignore", category=UserWarning)
                if True:
                    torch.onnx.export(
                        onnx_model,
                        tuple(dummy_inputs.values()),
                        onnx_model_path, #f,
                        export_params=True,
                        verbose=True,
                        opset_version=18,
                        do_constant_folding=False,
                        input_names=list(dummy_inputs.keys()),
                        output_names=output_names,
                        #dynamic_axes=dynamic_axes,
                    )  
                else:
                    with open(onnx_model_path, "wb") as f:
                        torch.onnx.export(
                        onnx_model,
                        tuple(dummy_inputs.values()),
                        f,
                        export_params=True,
                        verbose=True,
                        opset_version=18,
                        do_constant_folding=False,
                        input_names=list(dummy_inputs.keys()),
                        output_names=output_names,
                        #dynamic_axes=dynamic_axes,
                        )  
    check_onnx(onnx_model_path)
    onnx_model.to('cpu')
    return

def torch_onnx_export_net_recon(onnx_model, fp16=False, onnx_model_path="model.onnx", maxBatch=1 ):
    if not os.path.exists(onnx_model_path):
        dynamic_axes = {
            "latent_model_input":   {0: "bs_x_2"},
            "prompt_embeds":        {0: "bs_x_2"},
            "noise_pred":           {0: "batch_size"}
        }

        device = torch.device("cuda:0")
        
        onnx_model2= onnx_model #onnx_model2= UNet_x(onnx_model)
        if isinstance(onnx_model2, torch.nn.DataParallel):
            onnx_model2 = onnx_model2.module

        onnx_model2.eval()
        onnx_model2 = onnx_model2.to(device=device)
        
        if fp16: dst_dtype = torch.float16
        else: dst_dtype = torch.float32

        '''
        inp: (torch.Size([1, 3, 256, 256]), device(type='cuda', index=0), torch.float32)
        output: (torch.Size([1, 99, 64, 64]), device(type='cuda', index=0), torch.float32)
        '''
        dummy_inputs = {
            "im_t": torch.randn((1, 3, 256, 256), dtype=dst_dtype).to(device).contiguous(),
        }
        output_names = ["full_coeff"]

        #import apex
        with torch.no_grad():
            #with warnings.catch_warnings():
            if True:
                #warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
                #warnings.filterwarnings("ignore", category=UserWarning)
                if True:
                    torch.onnx.export(
                        onnx_model,
                        tuple(dummy_inputs.values()),
                        onnx_model_path, #f,
                        export_params=True,
                        verbose=True,
                        opset_version=18,
                        do_constant_folding=False,
                        input_names=list(dummy_inputs.keys()),
                        output_names=output_names,
                        #dynamic_axes=dynamic_axes,
                    )  
                else:
                    with open(onnx_model_path, "wb") as f:
                        torch.onnx.export(
                        onnx_model,
                        tuple(dummy_inputs.values()),
                        f,
                        export_params=True,
                        verbose=True,
                        opset_version=18,
                        do_constant_folding=False,
                        input_names=list(dummy_inputs.keys()),
                        output_names=output_names,
                        #dynamic_axes=dynamic_axes,
                        )  
    check_onnx(onnx_model_path)
    onnx_model.to('cpu')
    return

def torch_onnx_export_face_detector(onnx_model, input_shape, fp16=False, onnx_model_path="model.onnx", maxBatch=1 ):
    if not os.path.exists(onnx_model_path):
        dynamic_axes = {
            "latent_model_input":   {0: "bs_x_2"},
            "prompt_embeds":        {0: "bs_x_2"},
            "noise_pred":           {0: "batch_size"}
        }

        device = torch.device("cuda:0")
        
        onnx_model2= onnx_model #onnx_model2= UNet_x(onnx_model)
        if isinstance(onnx_model2, torch.nn.DataParallel):
            onnx_model2 = onnx_model2.module

        onnx_model2.eval()
        onnx_model2 = onnx_model2.to(device=device)
        
        if fp16: dst_dtype = torch.float16
        else: dst_dtype = torch.float32

        '''
        inputs: (torch.Size([1, 3, 1200, 800]), device(type='cuda', index=0), torch.float32)
        loc: (torch.Size([1, 39400, 4]), device(type='cuda', index=0), torch.float32)
        conf: (torch.Size([1, 39400, 2]), device(type='cuda', index=0), torch.float32)
        landmarks: (torch.Size([1, 39400, 10]), device(type='cuda', index=0), torch.float32)
        '''
        dummy_inputs = {
            #"source_image": torch.randn((1, 3, 1200, 800), dtype=dst_dtype).to(device).contiguous(),
            "source_image": torch.randn(input_shape, dtype=dst_dtype).to(device).contiguous(),
        }
        output_names = ["loc", "conf", "landmarks"]

        #import apex
        with torch.no_grad():
            #with warnings.catch_warnings():
            if True:
                #warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
                #warnings.filterwarnings("ignore", category=UserWarning)
                if True:
                    torch.onnx.export(
                        onnx_model,
                        tuple(dummy_inputs.values()),
                        onnx_model_path, #f,
                        export_params=True,
                        verbose=True,
                        opset_version=17,
                        do_constant_folding=False,
                        input_names=list(dummy_inputs.keys()),
                        output_names=output_names,
                        #dynamic_axes=dynamic_axes,
                    )  
                else:
                    with open(onnx_model_path, "wb") as f:
                        torch.onnx.export(
                        onnx_model,
                        tuple(dummy_inputs.values()),
                        f,
                        export_params=True,
                        verbose=True,
                        opset_version=18,
                        do_constant_folding=False,
                        input_names=list(dummy_inputs.keys()),
                        output_names=output_names,
                        #dynamic_axes=dynamic_axes,
                        )  
    check_onnx(onnx_model_path)
    onnx_model.to('cpu')
    return

def torch_onnx_export_gfpgan(onnx_model, fp16=False, onnx_model_path="model.onnx", maxBatch=1 ):
    if not os.path.exists(onnx_model_path):
        dynamic_axes = {
            "latent_model_input":   {0: "bs_x_2"},
            "prompt_embeds":        {0: "bs_x_2"},
            "noise_pred":           {0: "batch_size"}
        }

        device = torch.device("cuda:0")
        
        onnx_model2= onnx_model #onnx_model2= UNet_x(onnx_model)
        if isinstance(onnx_model2, torch.nn.DataParallel):
            onnx_model2 = onnx_model2.module

        onnx_model2.eval()
        onnx_model2 = onnx_model2.to(device=device)
        
        if fp16: dst_dtype = torch.float16
        else: dst_dtype = torch.float32

        '''
        cropped_face_t: (torch.Size([1, 3, 512, 512]), device(type='cuda', index=0), torch.float32)
        weight: 0.5
        output: (torch.Size([1, 3, 512, 512]), device(type='cuda', index=0), torch.float32)
        '''
        dummy_inputs = {
            "cropped_face_t": torch.randn((1, 3, 512, 512), dtype=dst_dtype).to(device).contiguous(),
        }
        output_names = ["image"]

        #import apex
        with torch.no_grad():
            #with warnings.catch_warnings():
            if True:
                #warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
                #warnings.filterwarnings("ignore", category=UserWarning)
                if True:
                    torch.onnx.export(
                        onnx_model,
                        tuple(dummy_inputs.values()),
                        onnx_model_path, #f,
                        export_params=True,
                        verbose=True,
                        opset_version=17,
                        do_constant_folding=False,
                        input_names=list(dummy_inputs.keys()),
                        output_names=output_names,
                        #dynamic_axes=dynamic_axes,
                    )  
                else:
                    with open(onnx_model_path, "wb") as f:
                        torch.onnx.export(
                        onnx_model,
                        tuple(dummy_inputs.values()),
                        f,
                        export_params=True,
                        verbose=True,
                        opset_version=18,
                        do_constant_folding=False,
                        input_names=list(dummy_inputs.keys()),
                        output_names=output_names,
                        #dynamic_axes=dynamic_axes,
                        )  
    check_onnx(onnx_model_path)
    onnx_model.to('cpu')
    return

def torch_onnx_export_face_pase(onnx_model, fp16=False, onnx_model_path="model.onnx", maxBatch=1 ):
    if not os.path.exists(onnx_model_path):
        dynamic_axes = {
            "latent_model_input":   {0: "bs_x_2"},
            "prompt_embeds":        {0: "bs_x_2"},
            "noise_pred":           {0: "batch_size"}
        }

        device = torch.device("cuda:0")
        
        onnx_model2= onnx_model #onnx_model2= UNet_x(onnx_model)
        if isinstance(onnx_model2, torch.nn.DataParallel):
            onnx_model2 = onnx_model2.module

        onnx_model2.eval()
        onnx_model2 = onnx_model2.to(device=device)
        
        if fp16: dst_dtype = torch.float16
        else: dst_dtype = torch.float32

        '''
        face_input: (torch.Size([1, 3, 512, 512]), device(type='cuda', index=0), torch.float32)
        out: (torch.Size([1, 19, 512, 512]), device(type='cuda', index=0), torch.float32)
        '''
        dummy_inputs = {
            "face_input": torch.randn((1, 3, 512, 512), dtype=dst_dtype).to(device).contiguous(),
        }
        output_names = ["out"]

        #import apex
        with torch.no_grad():
            #with warnings.catch_warnings():
            if True:
                #warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
                #warnings.filterwarnings("ignore", category=UserWarning)
                if True:
                    torch.onnx.export(
                        onnx_model,
                        tuple(dummy_inputs.values()),
                        onnx_model_path, #f,
                        export_params=True,
                        verbose=True,
                        opset_version=17,
                        do_constant_folding=False,
                        input_names=list(dummy_inputs.keys()),
                        output_names=output_names,
                        #dynamic_axes=dynamic_axes,
                    )  
                else:
                    with open(onnx_model_path, "wb") as f:
                        torch.onnx.export(
                        onnx_model,
                        tuple(dummy_inputs.values()),
                        f,
                        export_params=True,
                        verbose=True,
                        opset_version=18,
                        do_constant_folding=False,
                        input_names=list(dummy_inputs.keys()),
                        output_names=output_names,
                        #dynamic_axes=dynamic_axes,
                        )  
    check_onnx(onnx_model_path)
    onnx_model.to('cpu')
    return

import cv2
import numpy as np
import torch.nn.functional as F
import torch
import math
import cvcuda

def main(args):
    assert args.enhancer=='gfpgan' # don't remove this line 
    assert args.background_enhancer is None # don't remove this line 

    mm = Memory_Manager()
    mm.add_foot_print("prev-E2E")
    torchutil = TorchUtil(gpu=0, memory_manager=mm, cvcuda_stream=None)

    #torch.backends.cudnn.enabled = False
    with NVTXUtil("init", "red", mm), SynchronizeUtil(torchutil.torch_stream):
        pic_path = args.source_image
        audio_path = args.driven_audio
        save_dir = os.path.join(args.result_dir, strftime("%Y_%m_%d_%H.%M.%S"))
        os.makedirs(save_dir, exist_ok=True)
        pose_style = args.pose_style
        device = args.device
        batch_size = args.batch_size
        input_yaw_list = args.input_yaw
        input_pitch_list = args.input_pitch
        input_roll_list = args.input_roll
        ref_eyeblink = args.ref_eyeblink
        ref_pose = args.ref_pose

        current_root_path = os.path.split(sys.argv[0])[0]

        sadtalker_paths = init_path(args.checkpoint_dir, os.path.join(current_root_path, 'src/config'), args.size, args.old_version, args.preprocess)

    with NVTXUtil("load models", "red", mm), SynchronizeUtil(torchutil.torch_stream):
        #init model
        with NVTXUtil("CropAndExtract", "blue", mm), SynchronizeUtil(torchutil.torch_stream):
            preprocess_model = CropAndExtract(sadtalker_paths, device)

            #with NVTXUtil("onnx_face_detector", "blue", mm), SynchronizeUtil(torchutil.torch_stream):
            #    onnx_model_path = "./onnx_trt/face_detector_bs1_fp32_6000.onnx"
            #    torch_onnx_export_face_detector(restorer.face_helper.face_det, 
            #                                                fp16=False, 
            #                                                onnx_model_path=onnx_model_path, 
            #                                                maxBatch=1 )
            with NVTXUtil("trt_face_detector", "blue", mm), SynchronizeUtil(torchutil.torch_stream):
                detect_faces_trt_engine = None
                trt_engine_path = "./onnx_trt/face_detector_bs1_fp32_6000.engine"
            
                #build_TensorRT_engine_CLI( src_onnx_path=onnx_model_path, 
                #                        dst_trt_engine_path=trt_engine_path )
            
                if detect_faces_trt_engine is None: 
                    inputs_name = ["source_image",]
                    outputs_name = ["loc", "conf", "landmarks"]
                    detect_faces_trt_engine = TRT_Engine(trt_engine_path, gpu_id=0, torch_stream=torchutil.torch_stream,
                                                        onnx_inputs_name = inputs_name,
                                                        onnx_outputs_name = outputs_name,)
                    assert detect_faces_trt_engine
                    preprocess_model.propress.predictor.det_net.trt_engine = detect_faces_trt_engine if USE_TRT else None

            #with NVTXUtil("onnx_face_detector", "blue", mm), SynchronizeUtil(torchutil.torch_stream):
            #    onnx_model_path = "./onnx_trt/face_detector_bs1_fp32_6000_1x3x256x256.onnx"
            #    torch_onnx_export_face_detector(restorer.face_helper.face_det, input_shape=(1,3,256,256),
            #                                                fp16=False, 
            #                                                onnx_model_path=onnx_model_path, 
            #                                                maxBatch=1 )
            with NVTXUtil("trt_face_detector", "blue", mm), SynchronizeUtil(torchutil.torch_stream):
                detect_faces_trt_engine_1x3x256x256 = None
                trt_engine_path = "./onnx_trt/face_detector_bs1_fp32_6000_1x3x256x256.engine"
            
                #build_TensorRT_engine_CLI( src_onnx_path=onnx_model_path, 
                #                        dst_trt_engine_path=trt_engine_path )
            
                if detect_faces_trt_engine_1x3x256x256 is None: 
                    inputs_name = ["source_image",]
                    outputs_name = ["loc", "conf", "landmarks"]
                    detect_faces_trt_engine_1x3x256x256 = TRT_Engine(trt_engine_path, gpu_id=0, torch_stream=torchutil.torch_stream,
                                                        onnx_inputs_name = inputs_name,
                                                        onnx_outputs_name = outputs_name,)
                    assert detect_faces_trt_engine_1x3x256x256
                    preprocess_model.propress.predictor.det_net.trt_engine_1x3x256x256 = detect_faces_trt_engine_1x3x256x256 if USE_TRT else None


            with NVTXUtil("onnx_fan", "blue", mm), SynchronizeUtil(torchutil.torch_stream):
                onnx_model_path = "./onnx_trt/fan_bs1_fp32_6000.onnx"
                torch_onnx_export_fan(preprocess_model.propress.predictor.detector, 
                                                            fp16=False, 
                                                            onnx_model_path=onnx_model_path, 
                                                            maxBatch=1 )
            with NVTXUtil("trt_fan", "blue", mm), SynchronizeUtil(torchutil.torch_stream):
                fan_trt_engine = None
                trt_engine_path = "./onnx_trt/fan_bs1_fp32_6000.engine"
            
                build_TensorRT_engine_CLI( src_onnx_path=onnx_model_path, 
                                        dst_trt_engine_path=trt_engine_path )
            
                if fan_trt_engine is None: 
                    inputs_name = ["inp",]
                    outputs_name = ["output"]
                    fan_trt_engine = TRT_Engine(trt_engine_path, gpu_id=0, torch_stream=torchutil.torch_stream,
                                                        onnx_inputs_name = inputs_name,
                                                        onnx_outputs_name = outputs_name,)
                    assert fan_trt_engine
                    preprocess_model.propress.predictor.detector.trt_engine = fan_trt_engine if USE_TRT else None

            #with NVTXUtil("onnx_net_recon", "blue", mm), SynchronizeUtil(torchutil.torch_stream):
            #    onnx_model_path = "./onnx_trt/net_recon_bs1_fp32_6000.onnx"
            #    torch_onnx_export_net_recon(preprocess_model.net_recon, 
            #                            fp16=False, 
            #                            onnx_model_path=onnx_model_path, 
            #                            maxBatch=1 )
            #with NVTXUtil("trt_net_recon", "blue", mm), SynchronizeUtil(torchutil.torch_stream):
            #    net_recon_trt_engine = None
            #    trt_engine_path = "./onnx_trt/net_recon_bs1_fp32_6000.engine"
            #
            #    build_TensorRT_engine_CLI( src_onnx_path=onnx_model_path, 
            #                            dst_trt_engine_path=trt_engine_path )
            #
            #    if net_recon_trt_engine is None: 
            #        inputs_name = ["inp",]
            #        outputs_name = ["output"]
            #        net_recon_trt_engine = TRT_Engine(trt_engine_path, gpu_id=0, torch_stream=torchutil.torch_stream,
            #                                            onnx_inputs_name = inputs_name,
            #                                            onnx_outputs_name = outputs_name,)
            #        assert net_recon_trt_engine
            #        preprocess_model.net_recon_trt_engine = net_recon_trt_engine if USE_TRT else None


        with NVTXUtil("Audio2Coeff", "blue", mm), SynchronizeUtil(torchutil.torch_stream):
            audio_to_coeff = Audio2Coeff(sadtalker_paths,  device)
        
        with NVTXUtil("AnimateFromCoeff", "blue", mm), SynchronizeUtil(torchutil.torch_stream):
            animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device)
            pth_model_path = "./onnx_trt/animate_from_coeff_generator_bs1_fp32_6000.pth"
            if not os.path.exists(pth_model_path): 
                torch.save(animate_from_coeff.generator.state_dict(), pth_model_path)

            # as GridSample 5D is not support in current latest torch.onnx.export, and it will be added in onnx version 1.16.
            # so just leave it there.
            with NVTXUtil("onnx", "blue", mm), SynchronizeUtil(torchutil.torch_stream):
                onnx_model_path = "./onnx_trt/animate_from_coeff_generator_bs1_fp32_6000.onnx"
                if not os.path.exists(onnx_model_path): 
                    torch_onnx_export_animate_from_coeff_generator(animate_from_coeff.generator, 
                                                               fp16=False, 
                                                               onnx_model_path=onnx_model_path, 
                                                               maxBatch=1 )
                onnx_model_path_modified = "./onnx_trt/animate_from_coeff_generator_bs1_fp32_6000_modified.onnx"
                if not os.path.exists(onnx_model_path_modified):
                    modify_onnx_animate_from_coeff_generator(onnx_model_path_modified, onnx_model_path)
                onnx_model_path = onnx_model_path_modified
            
            with NVTXUtil("trt", "blue", mm), SynchronizeUtil(torchutil.torch_stream):
                animate_from_coeff_generator_trt_engine = None
                trt_engine_path = "./onnx_trt/animate_from_coeff_generator_bs1_fp32_6000.engine"
                trt_plugins_so_paths = ["./grid-sample3d-trt-plugin/build/libgrid_sample_3d_plugin.so"]
                if not os.path.exists(trt_engine_path): 
                    input_shape = None #"--minShapes='source_image':2x3x256x256,'kp_source_value':2x15x3,'kp_norm_value':2x15x3 --optShapes='source_image':2x3x256x256,'kp_source_value':2x15x3,'kp_norm_value':2x15x3 --maxShapes='source_image':2x3x256x256,'kp_source_value':2x15x3,'kp_norm_value':2x15x3"
                    build_TensorRT_engine_CLI( src_onnx_path=onnx_model_path, 
                                          dst_trt_engine_path=trt_engine_path,
                                           trt_plugins_so_paths=trt_plugins_so_paths,
                                         input_shape=input_shape )
            
                if animate_from_coeff_generator_trt_engine is None: 
                    animate_from_coeff_generator_trt_engine = TRT_Engine(trt_engine_path, gpu_id=0, torch_stream=torchutil.torch_stream, trt_plugins_so_paths=trt_plugins_so_paths)
                    assert animate_from_coeff_generator_trt_engine
                animate_from_coeff.generator_trt_engine = animate_from_coeff_generator_trt_engine if USE_TRT else None
            #animate_from_coeff.generator_trt_engine = None
            
            method = 'gfpgan' # dont remove this line, as I move GFPGANer creator from face_enhancer.py to here and use method 'gfpgan' by default
            bg_upsampler = None
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
    
            # determine model paths
            model_path = os.path.join('gfpgan/weights', model_name + '.pth')
            
            if not os.path.isfile(model_path):
                model_path = os.path.join('checkpoints', model_name + '.pth')
            
            if not os.path.isfile(model_path):
                # download pre-trained models from url
                model_path = url

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
            animate_from_coeff.restorer = restorer
    #crop image and extract 3dmm from image
    first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)
    
    with NVTXUtil("preprocess_model", "red", mm), SynchronizeUtil(torchutil.torch_stream):
        print('3DMM Extraction for source image')
        if USE_WARM_UP:
            with NVTXUtil("3DMM_warmup", "red", mm), SynchronizeUtil(torchutil.torch_stream):
                first_coeff_path, crop_pic_path, crop_info =  preprocess_model.generate(pic_path, first_frame_dir, args.preprocess,\
                                                                                        source_image_flag=True, pic_size=args.size,
                                                                                        mm=mm, torchutil=torchutil)
        with NVTXUtil("3DMM", "red", mm), SynchronizeUtil(torchutil.torch_stream):
            first_coeff_path, crop_pic_path, crop_info =  preprocess_model.generate(pic_path, first_frame_dir, args.preprocess,\
                                                                                    source_image_flag=True, pic_size=args.size,
                                                                                    mm=mm, torchutil=torchutil)
        if first_coeff_path is None:
            print("Can't get the coeffs of the input")
            return

        if ref_eyeblink is not None:
            ref_eyeblink_videoname = os.path.splitext(os.path.split(ref_eyeblink)[-1])[0]
            ref_eyeblink_frame_dir = os.path.join(save_dir, ref_eyeblink_videoname)
            os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
            print('3DMM Extraction for the reference video providing eye blinking')
            with NVTXUtil("ref_eyeblink", "red", mm), SynchronizeUtil(torchutil.torch_stream):
                ref_eyeblink_coeff_path, _, _ =  preprocess_model.generate(ref_eyeblink, ref_eyeblink_frame_dir, args.preprocess, source_image_flag=False,
                                                                           mm=mm, torchutil=torchutil)
        else:
            ref_eyeblink_coeff_path=None

        if ref_pose is not None:
            if ref_pose == ref_eyeblink: 
                ref_pose_coeff_path = ref_eyeblink_coeff_path
            else:
                ref_pose_videoname = os.path.splitext(os.path.split(ref_pose)[-1])[0]
                ref_pose_frame_dir = os.path.join(save_dir, ref_pose_videoname)
                os.makedirs(ref_pose_frame_dir, exist_ok=True)
                print('3DMM Extraction for the reference video providing pose')
                with NVTXUtil("video_pose", "red", mm), SynchronizeUtil(torchutil.torch_stream):
                    ref_pose_coeff_path, _, _ =  preprocess_model.generate(ref_pose, ref_pose_frame_dir, args.preprocess, source_image_flag=False,
                                                                           mm=mm, torchutil=torchutil)
        else:
            ref_pose_coeff_path=None

    #audio2ceoff
    with NVTXUtil("get_data", "red", mm), SynchronizeUtil(torchutil.torch_stream):
        batch = get_data(first_coeff_path, audio_path, device, ref_eyeblink_coeff_path, still=args.still)
    with NVTXUtil("audio_to_coeff", "red", mm), SynchronizeUtil(torchutil.torch_stream):
        coeff_path = audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)

    # 3dface render
    if args.face3dvis:
        with NVTXUtil("gen_composed_video", "red", mm), SynchronizeUtil(torchutil.torch_stream):
            gen_composed_video(args, device, first_coeff_path, coeff_path, audio_path, os.path.join(save_dir, '3dface.mp4'))
    
    #coeff2video
    with NVTXUtil("get_facerender_data", "red", mm), SynchronizeUtil(torchutil.torch_stream):
        data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path, 
                                batch_size, input_yaw_list, input_pitch_list, input_roll_list,
                                expression_scale=args.expression_scale, still_mode=args.still, preprocess=args.preprocess, size=args.size)
    
    with NVTXUtil("animate_from_coeff", "red", mm), SynchronizeUtil(torchutil.torch_stream):
        result = animate_from_coeff.generate(data, save_dir, pic_path, crop_info, \
                                enhancer=args.enhancer, background_enhancer=args.background_enhancer, preprocess=args.preprocess, img_size=args.size,
                                mm = mm,
                                torchutil = torchutil,
                                restorer=animate_from_coeff.restorer)
    
    with NVTXUtil("shutil.move", "red", mm), SynchronizeUtil(torchutil.torch_stream):
        shutil.move(result, save_dir+'.mp4')
    print('The generated video is named:', save_dir+'.mp4')

    for seamless_clone in animate_from_coeff.seamless_clone_instances:
        seamless_clone.destroy()

    if not args.verbose:
        shutil.rmtree(save_dir)

def test_seamless_clone():
    seamless_clone = SeamlessClone();

    images = ["./images/airplane.jpg", "./images/sky.jpg", 
              "./images/img_cropped4_109x164.jpg", "./images/img_orl4.jpg" ]
    center = [800, 150, 
              252,469]
    
    for j in range(25):
        for i in range(len(images)//2):
            print(images[i*2+0])
            print(images[i*2+1])
            face = cv2.imread(images[i*2+0]);
            body = cv2.imread(images[i*2+1]);
            mask = np.full((face.shape[0], face.shape[1], 1), 255, dtype=np.uint8);

            centerX=center[i*2+0]
            centerY=center[i*2+1]
            gpu_id=0

            seamless_clone.loadMatsInSeamlessClone( face, body, mask, centerX, centerY, gpu_id );
            blendedMat = seamless_clone.seamlessClone();
            blend_file = "./output/blendedMat_{}_{}.jpg".format(j, i);
            cv2.imwrite(blend_file, blendedMat);

if __name__ == '__main__':
    #test_seamless_clone()

    parser = ArgumentParser()  
    parser.add_argument("--driven_audio", default='./examples/driven_audio/bus_chinese.wav', help="path to driven audio")
    parser.add_argument("--source_image", default='./examples/source_image/full_body_1.png', help="path to source image")
    parser.add_argument("--ref_eyeblink", default=None, help="path to reference video providing eye blinking")
    parser.add_argument("--ref_pose", default=None, help="path to reference video providing pose")
    parser.add_argument("--checkpoint_dir", default='./checkpoints', help="path to output")
    parser.add_argument("--result_dir", default='./results', help="path to output")
    parser.add_argument("--pose_style", type=int, default=0,  help="input pose style from [0, 46)")
    parser.add_argument("--batch_size", type=int, default=2,  help="the batch size of facerender")
    parser.add_argument("--size", type=int, default=256,  help="the image size of the facerender")
    parser.add_argument("--expression_scale", type=float, default=1.,  help="the batch size of facerender")
    parser.add_argument('--input_yaw', nargs='+', type=int, default=None, help="the input yaw degree of the user ")
    parser.add_argument('--input_pitch', nargs='+', type=int, default=None, help="the input pitch degree of the user")
    parser.add_argument('--input_roll', nargs='+', type=int, default=None, help="the input roll degree of the user")
    parser.add_argument('--enhancer',  type=str, default=None, help="Face enhancer, [gfpgan, RestoreFormer]")
    parser.add_argument('--background_enhancer',  type=str, default=None, help="background enhancer, [realesrgan]")
    parser.add_argument("--cpu", dest="cpu", action="store_true") 
    parser.add_argument("--face3dvis", action="store_true", help="generate 3d face and 3d landmarks") 
    parser.add_argument("--still", action="store_true", help="can crop back to the original videos for the full body aniamtion") 
    parser.add_argument("--preprocess", default='crop', choices=['crop', 'extcrop', 'resize', 'full', 'extfull'], help="how to preprocess the images" ) 
    parser.add_argument("--verbose",action="store_true", help="saving the intermedia output or not" ) 
    parser.add_argument("--old_version",action="store_true", help="use the pth other than safetensor version" ) 


    # net structure and parameters
    parser.add_argument('--net_recon', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50'], help='useless')
    parser.add_argument('--init_path', type=str, default=None, help='Useless')
    parser.add_argument('--use_last_fc',default=False, help='zero initialize the last fc')
    parser.add_argument('--bfm_folder', type=str, default='./checkpoints/BFM_Fitting/')
    parser.add_argument('--bfm_model', type=str, default='BFM_model_front.mat', help='bfm model')

    # default renderer parameters
    parser.add_argument('--focal', type=float, default=1015.)
    parser.add_argument('--center', type=float, default=112.)
    parser.add_argument('--camera_d', type=float, default=10.)
    parser.add_argument('--z_near', type=float, default=5.)
    parser.add_argument('--z_far', type=float, default=15.)

    args = parser.parse_args()

    if torch.cuda.is_available() and not args.cpu:
        args.device = "cuda"
    else:
        args.device = "cpu"
    print(f"using {args.device}")
    main(args)

