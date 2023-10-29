import cv2
import os
import torch
from basicsr.utils import img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from torchvision.transforms.functional import normalize

from gfpgan.archs.gfpgan_bilinear_arch import GFPGANBilinear
from gfpgan.archs.gfpganv1_arch import GFPGANv1
from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean
import nvtx
from cuda import cudart
from InferenceUtil import (
    cuda_call,
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class GFPGANer():
    """Helper for restoration with GFPGAN.

    It will detect and crop faces, and then resize the faces to 512x512.
    GFPGAN is used to restored the resized faces.
    The background is upsampled with the bg_upsampler.
    Finally, the faces will be pasted back to the upsample background image.

    Args:
        model_path (str): The path to the GFPGAN model. It can be urls (will first download it automatically).
        upscale (float): The upscale of the final output. Default: 2.
        arch (str): The GFPGAN architecture. Option: clean | original. Default: clean.
        channel_multiplier (int): Channel multiplier for large networks of StyleGAN2. Default: 2.
        bg_upsampler (nn.Module): The upsampler for the background. Default: None.
    """

    def __init__(self, model_path, upscale=2, arch='clean', channel_multiplier=2, bg_upsampler=None, device=None):
        self.upscale = upscale
        self.bg_upsampler = bg_upsampler

        # initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        # initialize the GFP-GAN
        print(f"arch: {arch}")
        if arch == 'clean':
            self.gfpgan = GFPGANv1Clean(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=channel_multiplier,
                decoder_load_path=None,
                fix_decoder=False,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True)
        elif arch == 'bilinear':
            self.gfpgan = GFPGANBilinear(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=channel_multiplier,
                decoder_load_path=None,
                fix_decoder=False,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True)
        elif arch == 'original':
            self.gfpgan = GFPGANv1(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=channel_multiplier,
                decoder_load_path=None,
                fix_decoder=True,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True)
        elif arch == 'RestoreFormer':
            from gfpgan.archs.restoreformer_arch import RestoreFormer
            self.gfpgan = RestoreFormer()
        # initialize face helper
        self.face_helper = FaceRestoreHelper(
            upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            use_parse=True,
            device=self.device,
            model_rootpath='gfpgan/weights')
        

        if model_path.startswith('https://'):
            model_path = load_file_from_url(
                url=model_path, model_dir=os.path.join(ROOT_DIR, 'gfpgan/weights'), progress=True, file_name=None)
        loadnet = torch.load(model_path)
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        self.gfpgan.load_state_dict(loadnet[keyname], strict=True)
        self.gfpgan.eval()
        self.gfpgan = self.gfpgan.to(self.device)

        self.gfpgan_trt_engine = None

    @torch.no_grad()
    def enhance(self, img, has_aligned=False, only_center_face=False, paste_back=True, weight=0.5,
                torch_stream=None):
        self.face_helper.clean_all()

        if has_aligned:  # the inputs are already aligned
            img = cv2.resize(img, (512, 512))
            self.face_helper.cropped_faces = [img]
        else:
            self.face_helper.read_image(img)
            # get face landmarks for each face
            self.face_helper.get_face_landmarks_5(only_center_face=only_center_face, eye_dist_threshold=5, 
                                                  torch_stream=torch_stream)
            # eye_dist_threshold=5: skip faces whose eye distance is smaller than 5 pixels
            # TODO: even with eye_dist_threshold, it will still introduce wrong detections and restorations.
            # align and warp each face
            self.face_helper.align_warp_face()

        rng = nvtx.start_range(message="face restoration", color="red")
        # face restoration
        for cropped_face in self.face_helper.cropped_faces:
            # prepare data
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True).to(self.device)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)

            try:
                #cuda_call(cudart.cudaStreamSynchronize(torch.cuda.current_stream().cuda_stream))
                if self.gfpgan_trt_engine:
                    rng1 = nvtx.start_range(message="gfpgan_trt", color="red")
                    trt_output = self.gfpgan_trt_engine.inference(inputs=[cropped_face_t.contiguous(),],
                                                            outputs = self.gfpgan_trt_engine.output_tensors,
                                                            torch_stream=torch_stream)
                    output = self.gfpgan_trt_engine.output_tensors[0]
                    #print(f"trt output: {output.shape, output.device, output.dtype}")
                else:
                    rng1 = nvtx.start_range(message="gfpgan_torch", color="red")
                    output = self.gfpgan(cropped_face_t, return_rgb=False, weight=weight)

                '''
                cropped_face_t: (torch.Size([1, 3, 512, 512]), device(type='cuda', index=0), torch.float32)
                weight: 0.5
                output: (torch.Size([1, 3, 512, 512]), device(type='cuda', index=0), torch.float32)
                '''
                #print(f"cropped_face_t: {cropped_face_t.shape, cropped_face_t.device, cropped_face_t.dtype}")
                #print(f"weight: {weight}")
                #print(f"output: {output.shape, output.device, output.dtype}")

                #cuda_call(cudart.cudaStreamSynchronize(torch.cuda.current_stream().cuda_stream))
                nvtx.end_range(rng1)
                # convert to image
                restored_face = tensor2img(output.squeeze(0), rgb2bgr=True, min_max=(-1, 1))
            except RuntimeError as error:
                print(f'\tFailed inference for GFPGAN: {error}.')
                restored_face = cropped_face

            restored_face = restored_face.astype('uint8')
            self.face_helper.add_restored_face(restored_face)
        nvtx.end_range(rng)

        
        if not has_aligned and paste_back:
            rng = nvtx.start_range(message="paste_back", color="red")
            # upsample the background
            if self.bg_upsampler is not None:
                # Now only support RealESRGAN for upsampling background
                rng0 = nvtx.start_range(message="bg_upsampler", color="red")
                bg_img = self.bg_upsampler.enhance(img, outscale=self.upscale)[0]
                nvtx.end_range(rng0)
            else:
                bg_img = None

            rng1 = nvtx.start_range(message="get_inverse_affine", color="red")
            self.face_helper.get_inverse_affine(None)
            nvtx.end_range(rng1)
            # paste each restored face to the input image

            rng1 = nvtx.start_range(message="paste_faces_to_input_image", color="red")
            restored_img = self.face_helper.paste_faces_to_input_image_torch(upsample_img=bg_img,
                                                                             torch_stream=torch_stream)
            nvtx.end_range(rng1)
            nvtx.end_range(rng)
            return self.face_helper.cropped_faces, self.face_helper.restored_faces, restored_img
        else:
            return self.face_helper.cropped_faces, self.face_helper.restored_faces, None
            