import os
import os.path as osp
from PIL import Image
import numpy as np
import time

import argparse # TODO: Install it
from datetime import datetime

import torch
import torchvision.transforms as tvtrans
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model

from collections import OrderedDict
from lib.model_zoo.ddim import DDIMSampler

controlnet_path = OrderedDict([
    # ['canny'             , ('canny'   , 'pretrained/controlnet/control_sd15_canny_slimmed.safetensors')],
    # ['canny_v11p'        , ('canny'   , 'pretrained/controlnet/control_v11p_sd15_canny_slimmed.safetensors')],
    # ['depth'             , ('depth'   , 'pretrained/controlnet/control_sd15_depth_slimmed.safetensors')],
    # ['hed'               , ('hed'     , 'pretrained/controlnet/control_sd15_hed_slimmed.safetensors')],
    # ['softedge_v11p'     , ('hed'     , 'pretrained/controlnet/control_v11p_sd15_softedge_slimmed.safetensors')],
    # ['mlsd'              , ('mlsd'    , 'pretrained/controlnet/control_sd15_mlsd_slimmed.safetensors')],
    # ['mlsd_v11p'         , ('mlsd'    , 'pretrained/controlnet/control_v11p_sd15_mlsd_slimmed.safetensors')],
    # ['normal'            , ('normal'  , 'pretrained/controlnet/control_sd15_normal_slimmed.safetensors')],
    ['openpose'          , ('openpose', 'pretrained/controlnet/control_sd15_openpose_slimmed.safetensors')],
    ['openpose_v11p'     , ('openpose', 'pretrained/controlnet/control_v11p_sd15_openpose_slimmed.safetensors')],
    # ['scribble'          , ('scribble', 'pretrained/controlnet/control_sd15_scribble_slimmed.safetensors')],
    # ['seg'               , ('none'    , 'pretrained/controlnet/control_sd15_seg_slimmed.safetensors')],
    # ['lineart_v11p'      , ('none'    , 'pretrained/controlnet/control_v11p_sd15_lineart_slimmed.safetensors')],
    # ['lineart_anime_v11p', ('none'    , 'pretrained/controlnet/control_v11p_sd15s2_lineart_anime_slimmed.safetensors')],
])

preprocess_method = [
    # 'canny'                ,
    # 'depth'                ,
    # 'hed'                  ,
    # 'mlsd'                 ,
    # 'normal'               ,
    'openpose'             ,
    'openpose_withface'    ,
    'openpose_withfacehand',
    # 'scribble'             ,
    'none'                 ,
]

diffuser_path = OrderedDict([
    ['SD-v1.5'             , 'pretrained/pfd/diffuser/SD-v1-5.safetensors'],
    # ['OpenJouney-v4'       , 'pretrained/pfd/diffuser/OpenJouney-v4.safetensors'],
    # ['Deliberate-v2.0'     , 'pretrained/pfd/diffuser/Deliberate-v2-0.safetensors'],
    # ['RealisticVision-v2.0', 'pretrained/pfd/diffuser/RealisticVision-v2-0.safetensors'],
    # ['Anything-v4'         , 'pretrained/pfd/diffuser/Anything-v4.safetensors'],
    # ['Oam-v3'              , 'pretrained/pfd/diffuser/AbyssOrangeMix-v3.safetensors'],
    # ['Oam-v2'              , 'pretrained/pfd/diffuser/AbyssOrangeMix-v2.safetensors'],
])

ctxencoder_path = OrderedDict([
    ['SeeCoder'      , 'pretrained/pfd/seecoder/seecoder-v1-0.safetensors'],
    # ['SeeCoder-PA'   , 'pretrained/pfd/seecoder/seecoder-pa-v1-0.safetensors'],
    # ['SeeCoder-Anime', 'pretrained/pfd/seecoder/seecoder-anime-v1-0.safetensors'],
])

n_sample_image = 1

def highlight_print(info):
    print('')
    print(''.join(['#']*(len(info)+4)))
    print('# '+info+' #')
    print(''.join(['#']*(len(info)+4)))
    print('')

def load_sd_from_file(target):
    if osp.splitext(target)[-1] == '.ckpt':
        sd = torch.load(target, map_location='cpu')['state_dict']
    elif osp.splitext(target)[-1] == '.pth':
        sd = torch.load(target, map_location='cpu')
    elif osp.splitext(target)[-1] == '.safetensors':
        from safetensors.torch import load_file as stload
        sd = OrderedDict(stload(target, device='cpu'))
    else:
        assert False, "File type must be .ckpt or .pth or .safetensors"
    return sd   

class prompt_free_diffusion(object):
    def __init__(self, 
                 fp16=False, 
                 tag_ctx=None,
                 tag_diffuser=None,
                 tag_ctl=None,):

        self.tag_ctx = tag_ctx
        self.tag_diffuser = tag_diffuser
        self.tag_ctl = tag_ctl
        self.strict_sd = True

        cfgm = model_cfg_bank()('pfd_seecoder_with_controlnet')
        self.net = get_model()(cfgm)
        
        self.action_load_ctx(tag_ctx)
        self.action_load_diffuser(tag_diffuser)
        self.action_load_ctl(tag_ctl)
 
        if fp16:
            highlight_print('Running in FP16')
            self.net.ctx['image'].fp16 = True
            self.net = self.net.half()
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.net.to('cuda')

        self.net.eval()
        self.sampler = DDIMSampler(self.net)

        self.n_sample_image = n_sample_image
        self.ddim_steps = 50
        self.ddim_eta = 0.0
        self.image_latent_dim = 4

    def load_ctx(self, pretrained):
        sd = load_sd_from_file(pretrained)
        sd_extra = [(ki, vi) for ki, vi in self.net.state_dict().items() \
            if ki.find('ctx.')!=0]
        sd.update(OrderedDict(sd_extra))

        self.net.load_state_dict(sd, strict=True)
        print('Load context encoder from [{}] strict [{}].'.format(pretrained, True))

    def load_diffuser(self, pretrained):
        sd = load_sd_from_file(pretrained)
        if len([ki for ki in sd.keys() if ki.find('diffuser.image.context_blocks.')==0]) == 0:
            sd = [(
                ki.replace('diffuser.text.context_blocks.', 'diffuser.image.context_blocks.'), vi) 
                    for ki, vi in sd.items()]
            sd = OrderedDict(sd)
        sd_extra = [(ki, vi) for ki, vi in self.net.state_dict().items() \
            if ki.find('diffuser.')!=0]
        sd.update(OrderedDict(sd_extra))
        self.net.load_state_dict(sd, strict=True)
        print('Load diffuser from [{}] strict [{}].'.format(pretrained, True))

    def load_ctl(self, pretrained):
        sd = load_sd_from_file(pretrained)
        self.net.ctl.load_state_dict(sd, strict=True)
        print('Load controlnet from [{}] strict [{}].'.format(pretrained, True))

    def action_load_ctx(self, tag):
        pretrained = ctxencoder_path[tag]
        if tag == 'SeeCoder-PA':
            from lib.model_zoo.seecoder import PPE_MLP
            pe_layer = \
                PPE_MLP(freq_num=20, freq_max=None, out_channel=768, mlp_layer=3)
            if self.dtype == torch.float16:
                pe_layer = pe_layer.half()
            if self.use_cuda:
                pe_layer.to('cuda')
            pe_layer.eval()
            self.net.ctx['image'].qtransformer.pe_layer = pe_layer
        else:
            self.net.ctx['image'].qtransformer.pe_layer = None
        if pretrained is not None:
            self.load_ctx(pretrained)
        self.tag_ctx = tag
        return tag

    def action_load_diffuser(self, tag):
        pretrained = diffuser_path[tag]
        if pretrained is not None:
            self.load_diffuser(pretrained)
        self.tag_diffuser = tag
        return tag

    def action_load_ctl(self, tag):
        pretrained = controlnet_path[tag][1]
        if pretrained is not None:
            self.load_ctl(pretrained)
        self.tag_ctl = tag
        return tag

    def action_autoset_hw(self, imctl):
        if imctl is None:
            return 512, 512
        w, h = imctl.size
        w = w//64 * 64
        h = h//64 * 64
        w = w if w >=512 else 512
        w = w if w <=1536 else 1536
        h = h if h >=512 else 512
        h = h if h <=1536 else 1536
        return h, w

    def action_autoset_method(self, tag):
        return controlnet_path[tag][0]

    # TODO: Have to use this function to manually call inference
    # since the GPU node cannot maintain an open port connection for
    # gradio demo to run
    def action_inference(
            self, im, imctl, ctl_method, do_preprocess, 
            h, w, ugscale, seed, 
            tag_ctx, tag_diffuser, tag_ctl,):

        if tag_ctx != self.tag_ctx:
            self.action_load_ctx(tag_ctx)
        if tag_diffuser != self.tag_diffuser:
            self.action_load_diffuser(tag_diffuser)
        if tag_ctl != self.tag_ctl:
            self.action_load_ctl(tag_ctl)

        n_samples = self.n_sample_image

        sampler = self.sampler
        device = self.net.device

        w = w//64 * 64
        h = h//64 * 64
        if imctl is not None:
            imctl = imctl.resize([w, h], Image.Resampling.BICUBIC)

        craw = tvtrans.ToTensor()(im)[None].to(device).to(self.dtype)
        c = self.net.ctx_encode(craw, which='image').repeat(n_samples, 1, 1)
        u = torch.zeros_like(c)

        if tag_ctx in ["SeeCoder-Anime"]:
            u = torch.load('assets/anime_ug.pth')[None].to(device).to(self.dtype)
            pad = c.size(1) - u.size(1)
            u = torch.cat([u, torch.zeros_like(u[:, 0:1].repeat(1, pad, 1))], axis=1)

        if tag_ctl != 'none':
            ccraw = tvtrans.ToTensor()(imctl)[None].to(device).to(self.dtype)
            if do_preprocess:
                cc = self.net.ctl.preprocess(ccraw, type=ctl_method, size=[h, w])
                cc = cc.to(self.dtype)
            else:
                cc = ccraw
        else:
            cc = None

        shape = [n_samples, self.image_latent_dim, h//8, w//8]

        if seed < 0:
            np.random.seed(int(time.time()))
            torch.manual_seed(-seed + 100)
        else:
            np.random.seed(seed + 100)
            torch.manual_seed(seed)

        x, _ = sampler.sample(
            steps=self.ddim_steps,
            x_info={'type':'image',},
            c_info={'type':'image', 'conditioning':c, 'unconditional_conditioning':u, 
                    'unconditional_guidance_scale':ugscale,
                    'control':cc,},
            shape=shape,
            verbose=False,
            eta=self.ddim_eta)

        ccout = [tvtrans.ToPILImage()(i) for i in cc] if cc is not None else []
        imout = self.net.vae_decode(x, which='image')
        imout = [tvtrans.ToPILImage()(i) for i in imout]
        return imout + ccout

pfd_inference = prompt_free_diffusion(
    fp16=True, tag_ctx = 'SeeCoder', tag_diffuser = 'SD-v1.5', tag_ctl = 'openpose')

def generate(input_image_fp: str,
             control_image_fp: str,
             fp16: bool = True,
             tag_ctx: str = "SeeCoder",
             tag_diffuser: str = "SD-v1.5",
             tag_ctl: str = "openpose",
             ctl_method: str = "openpose",
             do_preprocess: bool = True,
             output_height: int = 512,
             output_width: int = 512,
             scl_level: float = 2.,
             seed: int = 42,
             output_dir: str = None
             ):
    
    input_image_pil = Image.open(os.path.join(os.getcwd(), input_image_fp))
    control_image_pil = Image.open(os.path.join(os.getcwd(), control_image_fp))

    pfd_inference = prompt_free_diffusion(fp16=fp16,
                                          tag_ctx=tag_ctx,
                                          tag_diffuser=tag_diffuser,
                                          tag_ctl=tag_ctl)
    
    img_outputs = pfd_inference.action_inference(input_image_pil,
                                                 control_image_pil,
                                                 ctl_method,
                                                 do_preprocess,
                                                 output_height,
                                                 output_width,
                                                 scl_level,
                                                 seed,
                                                 tag_ctx,
                                                 tag_diffuser,
                                                 tag_ctl)
    
    if output_dir:
        output_dir = os.path.join(os.getcwd(), output_dir)
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        output_img = img_outputs[0]
        procsessed_control_img = img_outputs[1]

        output_img.save(os.path.join(output_dir, f"generated_output_image.jpg"))
        procsessed_control_img.save(os.path.join(output_dir, f"processed_control_image.jpg"))

        print(f"SUCCESS - Images successfully saved to {output_dir}.")
    
    return img_outputs


if __name__ == "__main__":

    # Get the current timestamp
    current_timestamp = datetime.now()
    # Format the timestamp in DDMMYY-HH:MM:SS format
    formatted_timestamp = current_timestamp.strftime("%d%m%y-%H:%M:%S")
    
    parser = argparse.ArgumentParser(description="Generate image using pre-trained models pipelines.")
    parser.add_argument("input_image_fp", type=str, help="Path to source image to base generation off.")
    parser.add_argument("control_image_fp", type=str, help="Path to control image to base generation off.")
    parser.add_argument("--fp16", action="store_true", help="Flag whether to use fp16.")
    parser.add_argument("--tag_ctx", type=str, default="SeeCoder", help="Name of context model to use.")
    parser.add_argument("--tag_diffuser", type=str, default="SD-v1.5", help="Name of diffusion model to use.")
    parser.add_argument("--tag_ctl", type=str, default="openpose", help="Name of control model to use.")
    parser.add_argument("--ctl_method", type=str, default="openpose", help="Name of control method to use.")
    parser.add_argument("--do_preprocess", action="store_true", help="Flag whether to do preprocessing for control model selected. Put False if using OpenPose skeletal map directly, True otherwise")
    parser.add_argument("--output_height", type=int, default=512, help="Height of output image.")
    parser.add_argument("--output_width", type=int, default=512, help="Width of output image.")
    parser.add_argument("--scl_level", type=float, default=2., help="CFG Scale Level.")
    parser.add_argument("--seed", type=int, default=42, help="Seed to base generation off.")
    parser.add_argument("--output_dir", type=str, default=f"demo/output/{formatted_timestamp}", help='Name of output dir to save generated images to.')

    args = parser.parse_args()

    generate(args.input_image_fp,
             args.control_image_fp,
             args.fp16,
             args.tag_ctx,
             args.tag_diffuser,
             args.tag_ctl,
             args.ctl_method,
             args.do_preprocess,
             args.output_height,
             args.output_width,
             args.scl_level,
             args.seed,
             args.output_dir
             )