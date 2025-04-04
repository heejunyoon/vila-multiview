#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import json
import logging
import os
import os.path as osp
import warnings
from abc import ABC
from collections import OrderedDict, defaultdict, deque
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from hydra.utils import instantiate
from transformers import AutoConfig, GenerationConfig, LogitsProcessor
from transformers.modeling_utils import ContextManagers, no_init_weights

from llava.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, NUM_EXTRA_TOKENS
from llava.mm_utils import process_image, process_images
from llava.model.configuration_llava import LlavaConfig, ResponseFormat
from llava.model.language_model.builder import build_llm_and_tokenizer
from llava.model.multimodal_encoder.builder import build_vision_tower
from llava.model.multimodal_projector.builder import build_mm_projector
from llava.model.utils import get_model_config
from llava.train.sequence_parallel import get_pg_manager
from llava.utils import distributed as dist
from llava.utils.media import extract_media
from llava.utils.tokenizer import tokenize_conversation

# matching용 추가.
import sys
sys.path.append("/home/heejunyoon/1_CODE/VILA/mast3r") #경로추가
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs_idx_sorting, fast_reciprocal_NNs_dist_sorting
from dust3r.inference import inference
import numpy as np
#############################################################################

class MASt3R():
    def __init__(self, *args, **kwargs):
        self.device = 'cuda'
        # self.merge_mode = kwargs.is_merging
    
    def load_mast3r(self, *args, **kwargs):
        model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
        mast3r = AsymmetricMASt3R.from_pretrained(model_name).to(self.device) # 모델 변경 시 dust3r에 model.py 확인
        return mast3r
    
    def merge_features(self, images, image_features, *args, **kwargs):
        import math
        mast3r = self.load_mast3r()
        num_images = images.size(0)
        patch_num_W = int(math.sqrt(image_features.size(1)))
        patch_size = 14 # 수정: 나중에는 config나 argument로 받아서 처리할 것.
        # merging 진행
        # org_image_features = copy.deepcopy(image_features)
        self.merge_mode = False
        merg_percentage = 1 # merg_percentage%의 매칭을 대체하는 방식으로 진행
        
        if self.merge_mode:
            print(f"merg_percentage {merg_percentage*100}%")
            print("!!!!! Token merging mode 진행!!!!!")
            image_features = image_features.view(num_images, patch_num_W, patch_num_W, 1152) # 27x27 grid로 변환
        else:
            return image_features
            
        for pair in range(num_images): #각 이미지 페어 대해 진행
            img1 = images[pair].to(torch.float32) # torch.Size([3, 384, 384])
            img2 = images[(pair+1)%num_images].to(torch.float32)
            dict1 = {"img": img1.unsqueeze(0),
                "true_shape": np.array([[img1.shape[-2], img1.shape[-1]]], dtype=np.int32),
                "idx": 0,
                "instance": '0'}
            
            dict2 = {"img": img2.unsqueeze(0),
                "true_shape": np.array([[img2.shape[-2], img2.shape[-1]]], dtype=np.int32),
                "idx": 1,
                "instance": '1'}
            
            image_temp = [dict1, dict2]  # Store as a list
            output_mast3r = inference([tuple(image_temp)], mast3r, self.device, batch_size=1, verbose=False) # 매칭 잘 되는것 확인

            pred1, pred2 = output_mast3r['pred1'], output_mast3r['pred2']
            desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach() #각각 torch.Size([384, 384, 24])

            # 참고
            # desc1, desc2 그냥 집어넣어서 fast_reciprocal_NNs_dist_sorting 하는 경우 --> 픽셀단위, 매칭 약 1000개 이상
            # desc1, desc2를 avg_pool2d로 14x14로 pooling 후 집어넣어서 fast_reciprocal_NNs_dist_sorting 하는 경우 --> 5~10개 정도 매칭

            # Aggregate features into patch-wise representations
            desc1 = desc1.permute(2, 0, 1).unsqueeze(0)  # (1, 24, H, W)
            desc2 = desc2.permute(2, 0, 1).unsqueeze(0)  # (1, 24, H, W)

            patch_features1 = F.avg_pool2d(desc1, kernel_size=patch_size, stride=patch_size)  # (1, 24, H//P, W//P)
            patch_features2 = F.avg_pool2d(desc2, kernel_size=patch_size, stride=patch_size)  # (1, 24, H//P, W//P)

            # permute the dimension from (B, num_patches, H, W) --> (H, W, DIM)
            patch_features1 = patch_features1.permute(0, 2, 3, 1).squeeze()
            patch_features2 = patch_features2.permute(0, 2, 3, 1).squeeze()
            
            ## distance sorting
            matches_im0, matches_im1 = fast_reciprocal_NNs_dist_sorting(
                patch_features1, patch_features2, subsample_or_initxy1=1, device='cuda')
            num_matches = matches_im0.shape[0]
            n_merge = int(num_matches * merg_percentage)  # merg% of the matches
            match_idx_to_merg = np.round(np.linspace(0, num_matches - 1, n_merge)).astype(int) # unif sampling
            
            ## idx sorting
            # matches_im0, matches_im1 = fast_reciprocal_NNs_idx_sorting(
            #     patch_features1, patch_features2, subsample_or_initxy1=1, device='cpu')
            # num_matches = matches_im0.shape[0]
            # n_merge = int(num_matches * merg_percentage)  # merg% of the matches
            # match_idx_to_merg = np.round(np.linspace(0, num_matches-1 , num_matches)).astype(int) # 상위 n_viz개만 갖고오도록...
            
            merg_matches_im0, merg_matches_im1 = matches_im0[match_idx_to_merg], matches_im1[match_idx_to_merg]

            
            if num_matches < 10: #매칭이 너무 적으면 넘어감. 보통 매치 안되는 경우 10개 내외여서 이렇게 설정. 필요하면 수정
                print(f"pair {pair} match 개수 {num_matches} --> 너무 적어서 넘어감")
                continue
                        
            if self.merge_mode:
                # print(f"image_features {image_features.size()}")
                # print(f"pair {pair}")
                # 20%의 매칭을 대체하는 방식
                for i in range(n_merge):
                    (x0, y0), (x1, y1) = merg_matches_im0[i].T, merg_matches_im1[i].T #패치 단위로 변환
                    
                    # replace 방식으로 merging진행.
                    # image_features[(pair+1)%num_images,x1, y1,:] = image_features[pair,x0,y0,:]

                    # avg 방식으로 merging진행.
                    image_features[(pair+1)%num_images,x1, y1,:] = (image_features[pair,x0,y0,:]+image_features[(pair+1)%num_images,x1, y1,:])/2
            
            from matplotlib import pyplot as plt
            def viz(matches_im0,matches_im1,img1,img2, merg_percentage):
                num_matches = matches_im0.shape[0]
                n_merge = int(num_matches * merg_percentage)  # merg% of the matches
                # match_idx_to_merg = np.round(np.linspace(0, num_matches-1 , num_matches)).astype(int) # 상위 n_viz개만 갖고오도록...
                match_idx_to_merg = np.round(np.linspace(0, num_matches - 1, n_merge)).astype(int) # unif sampling
                merg_matches_im0, merg_matches_im1 = matches_im0[match_idx_to_merg], matches_im1[match_idx_to_merg]
                # # 시각화용 코드.
                
                # color_map_list = [
                #     tuple(map(int, cv2.applyColorMap(np.array([[int(i / (n_merge - 1) * 255)]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0].tolist()))
                #     for i in range(n_merge)]
                
                plt.figure(figsize=(15, 10))
                img1 = img1[:, :patch_num_W*patch_size, :patch_num_W*patch_size]
                img2 = img2[:, :patch_num_W*patch_size, :patch_num_W*patch_size]
                image_mean = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(3, 1, 1) #img 0~1 value
                image_std = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(3, 1, 1)
                viz_imgs = np.ones((img1.size(1), img1.size(2)*2, 3), dtype=np.uint8)  # White background
                viz_imgs[:, :img1.size(2),:] = (img1*image_std+image_mean).permute(1,2,0).cpu().numpy()*255
                viz_imgs[:, img1.size(2):,:] = (img2*image_std+image_mean).permute(1,2,0).cpu().numpy()*255
                            
                # print(f"match 개수 {len(matches_im0)}")
                cmap = plt.get_cmap('jet')
                for i in range(n_merge):
                    (x0, y0), (x1, y1) = 14*merg_matches_im0[i].T+patch_size//2, 14*merg_matches_im1[i].T #패치 단위로 변환

                    # cv2.line(viz_imgs, (int(x0), int(y0)), (int(x1 + images.size(3)), int(y1)), color_map_list[i], 1)
                    plt.plot([x0, x1 + images.size(2)], [y0, y1+patch_size//2], '-+', color=cmap(1- i / (num_matches - 1)), scalex=False, scaley=False)
                plt.imshow(viz_imgs)
                H, W = viz_imgs.shape[:2]
                for x in range(0, W, 14):
                    plt.axvline(x, color='white', linestyle='--', linewidth=0.1)  # Vertical grid lines every 16 pixels
                for y in range(0, H, 14):
                    plt.axhline(y, color='white', linestyle='--', linewidth=0.1)  # Horizontal grid lines every 16 pixels
                plt.axis("off")
                plt.title(f"Match Count: {num_matches}, use {int(merg_percentage*100)}%", fontsize=16)
                plt.savefig(f"dist_sorting_muir53_4_{pair}_{int(merg_percentage*100)}%.png", dpi=300, bbox_inches='tight')
                # plt.show()
        
            # viz(matches_im0,matches_im1,img1,img2, 0)
            # viz(matches_im0,matches_im1,img1,img2, 0.1)
            # viz(matches_im0,matches_im1,img1,img2, 0.2)
            # viz(matches_im0,matches_im1,img1,img2, 0.5)
            # viz(matches_im0,matches_im1,img1,img2, 0.7)
            # viz(matches_im0,matches_im1,img1,img2, 1)
        if self.merge_mode:
            image_features = image_features.view(num_images, -1, 1152) # 다시 사이즈 변환
        # is_same = torch.equal(org_image_features, image_features)
        # print(f"org_image_features & merged EQUAL : {is_same}")
        return image_features

###############################
class LlavaMetaModel(ABC):
    def init_vlm(self, config, *args, **kwargs):
        # TODO(ligeng): figure out how from_config and from_pretrained works in HF implementation.
        if hasattr(self, "llm") or hasattr(self, "vision_tower") or hasattr(self, "mm_projector"):
            # already initialized, skipped
            return

        model_dtype = getattr(config, "model_dtype", "torch.float16")
        if not hasattr(config, "model_dtype"):
            warnings.warn("model_dtype not found in config, defaulting to torch.float16.")
            config.model_dtype = model_dtype

        cfgs = get_model_config(config)
        if len(cfgs) == 3:
            llm_cfg, vision_tower_cfg, mm_projector_cfg = cfgs
        else:
            raise ValueError("`llm_cfg` `mm_projector_cfg` `vision_tower_cfg` not found in the config.")

        # print("Before init in Config")
        # if hasattr(config, "deepspeed") and "mics" in config.deepspeed:
        #     print("Using MiCS_Init")
        #     import deepspeed
        #     with deepspeed.zero.MiCS_Init():
        #         self.llm, self.tokenizer = build_llm_and_tokenizer(llm_cfg, config, *args, **kwargs)
        #         self.vision_tower = build_vision_tower(vision_tower_cfg, config)
        #         self.mm_projector = build_mm_projector(mm_projector_cfg, config)
        # else:
        self.llm, self.tokenizer = build_llm_and_tokenizer(llm_cfg, config, *args, **kwargs)
        self.vision_tower = build_vision_tower(vision_tower_cfg, config)
        self.mm_projector = build_mm_projector(mm_projector_cfg, config)
        # NOTE(ligeng): for xgrammer init, <image> <vila/video> and <vila/sentinel>
        self.vocab_size = config.llm_cfg["vocab_size"] + NUM_EXTRA_TOKENS

        # XGrammar tokenizer and grammar compiler
        # lazy init only when specified json output during inference
        self.grammar_compiler = None

        self.encoders = {}
        for name in ["image", "video"]:
            config = getattr(self.config, f"{name}_encoder")
            if isinstance(config, str):
                config = json.loads(config)
            self.encoders[name] = instantiate(config, parent=self)

        self.post_config()
        self.is_loaded = True

        assert (
            self.llm is not None or self.vision_tower is not None or self.mm_projector is not None
        ), "At least one of the components must be instantiated."

    @classmethod
    def load_from_config(cls, model_path_or_config, *args, **kwargs):
        pass

    ## FIXME we will use this function to load model in the future
    @classmethod
    def load_pretrained(cls, model_path_or_config, *args, **kwargs):
        print("load_pretrained in llava_arch.py is used")
        kwargs.pop("config", None)

        if isinstance(model_path_or_config, str):
            config = AutoConfig.from_pretrained(model_path_or_config)
        elif isinstance(model_path_or_config, LlavaConfig):
            config = model_path_or_config
        else:
            raise NotImplementedError(
                f"wrong type, {type(model_path_or_config)} \
                                      {isinstance(model_path_or_config, LlavaConfig)}"
            )

        model_dtype = getattr(config, "model_dtype", "torch.float16")
        if not hasattr(config, "model_dtype"):
            warnings.warn("model_dtype not found in config, defaulting to torch.float16.")
            config.model_dtype = model_dtype

        cfgs = get_model_config(config)
        if len(cfgs) == 3:
            llm_cfg, vision_tower_cfg, mm_projector_cfg = cfgs
        else:
            raise ValueError("`llm_cfg` `mm_projector_cfg` `vision_tower_cfg` not found in the config.")

        # print(llm_cfg, vision_tower_cfg, mm_projector_cfg); input("DEBUG load_pretrained")
        init_context = [
            no_init_weights(_enable=True),
        ]
        # print("Before Init Context")
        # if hasattr(config, "deepspeed") and "mics" in config.deepspeed:
        #     print("Using MiCS_Init")
        #     import deepspeed
        #     init_context.append(deepspeed.zero.MiCS_Init(config_dict_or_path=config.deepspeed))
        with ContextManagers(init_context):
            vlm = cls(config, *args, **kwargs)
        # print(llm_cfg, vision_tower_cfg, mm_projector_cfg); input("DEBUG load_pretrained finish")

        if hasattr(vlm, "llm") or hasattr(vlm, "vision_tower") or hasattr(vlm, "mm_projector"):
            if vlm.is_loaded:
                return vlm

        vlm.llm, vlm.tokenizer = build_llm_and_tokenizer(llm_cfg, config, *args, **kwargs)
        vlm.vision_tower = build_vision_tower(vision_tower_cfg, config)
        vlm.mm_projector = build_mm_projector(mm_projector_cfg, config)

        self.post_config()
        self.is_loaded = True

        # FIXME(ligeng, yunhao): llm should never be none here.
        assert (
            vlm.llm is not None or vlm.vision_tower is not None or vlm.mm_projector is not None
        ), "At least one of the components must be instantiated."
        return vlm

    ## FIXME we will use this function to save the model in the future
    def save_pretrained(self, output_dir, state_dict=None):
        if state_dict is None:
            # other wise fetch from deepspeed
            # state_dict = accelerator.get_state_dict(is_deepspeed_enabled)
            state_dict = self.state_dict()

        if getattr(self, "tokenizer", None):
            self.tokenizer.save_pretrained(osp.join(output_dir, "llm"))

        if self.get_llm():
            print(f"saving llm to {osp.join(output_dir, 'llm')}")
            self.llm.config._name_or_path = osp.join(output_dir, "llm")
            llm_state_dict = OrderedDict({k.split("llm.")[-1]: v for k, v in state_dict.items() if "llm" in k})
            self.llm.save_pretrained(os.path.join(output_dir, "llm"), state_dict=llm_state_dict)
            self.config.llm_cfg = self.llm.config

        if self.get_vision_tower():
            print(f"saving vision_tower to {osp.join(output_dir, 'vision_tower')}")
            self.vision_tower.config._name_or_path = osp.join(output_dir, "vision_tower")
            vision_tower_state_dict = OrderedDict(
                {k.split("vision_tower.vision_tower.")[-1]: v for k, v in state_dict.items() if "vision_tower" in k}
            )
            self.vision_tower.vision_tower.save_pretrained(
                os.path.join(output_dir, "vision_tower"),
                state_dict=vision_tower_state_dict,
            )
            self.vision_tower.image_processor.save_pretrained(os.path.join(output_dir, "vision_tower"))
            self.config.vision_tower_cfg = self.vision_tower.config
            if hasattr(self.config.vision_tower_cfg, "auto_map"):
                if "radio" not in self.get_vision_tower().__class__.__name__.lower():
                    delattr(self.config.vision_tower_cfg, "auto_map")

        if self.get_mm_projector():
            print(f"saving mm_projector to {osp.join(output_dir, 'mm_projector')}")
            self.mm_projector.config._name_or_path = osp.join(output_dir, "mm_projector")
            mm_projector_state_dict = OrderedDict(
                {k.split("mm_projector.")[-1]: v for k, v in state_dict.items() if "mm_projector" in k}
            )
            self.mm_projector.save_pretrained(
                os.path.join(output_dir, "mm_projector"),
                state_dict=mm_projector_state_dict,
            )
            self.config.mm_projector_cfg = self.mm_projector.config
        ## update and save top-level config
        self.config._name_or_path = output_dir
        self.config.architectures = [self.__class__.__name__]
        self.config.save_pretrained(output_dir)

    def get_llm(self):
        llm = getattr(self, "llm", None)
        if type(llm) is list:
            llm = llm[0]
        return llm

    def get_lm_head(self):
        lm_head = getattr(self.get_llm(), "lm_head", None)
        return lm_head

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_mm_projector(self): #2번 호출됨.... 왜???
        mm_projector = getattr(self, "mm_projector", None)
        if type(mm_projector) is list:
            mm_projector = mm_projector[0]
        return mm_projector

    def post_config(self):
        self.training = self.get_llm().training
        ## configuration
        if getattr(self.config, "llm_cfg", None) is None:
            self.config.llm_cfg = self.llm.config
        if getattr(self.config, "vision_tower_cfg", None) is None:
            self.config.vision_tower_cfg = self.vision_tower.config
        if getattr(self.config, "mm_projector_cfg", None) is None:
            self.config.mm_projector_cfg = self.mm_projector.config

    def freezed_module_patch(self):
        """
        Huggingface will call model.train() at each training_step. To ensure the expected behaviors for modules like dropout, batchnorm, etc., we need to call model.eval() for the freezed modules.
        """
        if self.training:
            if self.get_llm() and not getattr(self.config, "tune_language_model", False):
                pass
                # logging.warning("Caution: Your LLM is currently in training mode, ensuring accurate gradient computation. Please be vigilant, particularly regarding BatchNorm and Dropout operations.")
            if self.get_vision_tower() and not getattr(self.config, "tune_vision_tower", False):
                self.get_vision_tower().eval()
            if self.get_mm_projector() and not getattr(self.config, "tune_mm_projector", False):
                self.get_mm_projector().eval()

    @staticmethod
    def merge_chessboard(x, num_split_h, num_split_w): # 당장은 안쓰이는듯?
        print("merge_chessboard in llava_arch.py is used --> please check what is the purpose of this function")
        """
        x: b * n * c or b * h * w * c
        out: b * c * h * w
        Assuming x contains num_split**2 sub-squares concatenated along batch dimension, merge the sub-squares back to the original whole square.
        """
        B = x.shape[0]
        if x.dim() == 3:
            N = x.shape[1]
            x = rearrange(x, "b (h w) c -> b c h w", h=int(N**0.5), w=int(N**0.5))

        assert B % (num_split_h * num_split_w) == 0
        b = B // (num_split_h * num_split_w)

        x_merge = torch.cat(
            [
                torch.cat(
                    [x[(i * num_split_w + j) * b : (i * num_split_w + j + 1) * b] for j in range(num_split_w)], dim=-1
                )
                for i in range(num_split_h)
            ],
            dim=-2,
        )

        return x_merge

    @staticmethod
    def split_chessboard(x, num_split_h, num_split_w):
        """
        x: b * c * h * w
        out: b * c * h * w
        Deividing x into num_split**2 sub-squares, and concatenate all the sub-squares on the batch dimension
        """
        B, C, H, W = x.shape
        assert H % num_split_h == 0 and W % num_split_w == 0
        h, w = H // num_split_h, W // num_split_w
        x_split = torch.cat(
            [x[:, :, i * h : (i + 1) * h, j * w : (j + 1) * w] for i in range(num_split_h) for j in range(num_split_w)],
            dim=0,
        )
        return x_split

    def merge_features_for_dynamic_s2(self, image_features, block_sizes):
        scales = self.get_vision_tower().scales
        resize_output_to_scale_idx = self.get_vision_tower().resize_output_to_scale_idx

        image_features_each_image = []
        new_block_sizes = []
        block_cnt = 0
        for block_size_each_image in block_sizes:
            if block_size_each_image is None:
                cur_features = image_features[block_cnt : block_cnt + 1]
                cur_features = rearrange(cur_features, "1 (h w) c -> 1 c h w", h=int(cur_features.shape[1] ** 0.5))
                cur_features = cur_features.repeat(1, len(scales), 1, 1)
                image_features_each_image.append(cur_features)
                new_block_sizes.append((1, 1))
                block_cnt += 1
            else:
                cur_features_each_scale = []
                for scale in scales[:-1]:
                    num_blocks_this_scale = (scale // scales[0]) ** 2
                    cur_features_each_scale.append(
                        self.merge_chessboard(
                            image_features[block_cnt : block_cnt + num_blocks_this_scale],
                            num_split_h=scale // scales[0],
                            num_split_w=scale // scales[0],
                        )
                    )  # 1 * C * H * W
                    block_cnt += num_blocks_this_scale
                num_blocks_last_scale = block_size_each_image[0] * block_size_each_image[1]
                cur_features_each_scale.append(
                    self.merge_chessboard(
                        image_features[block_cnt : block_cnt + num_blocks_last_scale],
                        num_split_h=block_size_each_image[0],
                        num_split_w=block_size_each_image[1],
                    )
                )  # 1 * C * H * W
                block_cnt += num_blocks_last_scale

                # resize and concat features from different scales
                output_size = cur_features_each_scale[resize_output_to_scale_idx].shape[-2:]
                cur_features = torch.cat(
                    [
                        F.interpolate(cur_features_each_scale[i].to(torch.float32), size=output_size, mode="area").to(
                            cur_features_each_scale[i].dtype
                        )
                        for i in range(len(cur_features_each_scale))
                    ],
                    dim=1,
                )
                # cur_features = rearrange(cur_features, "1 c h w -> (h w) c")

                image_features_each_image.append(cur_features)

                if resize_output_to_scale_idx == len(scales) - 1 or resize_output_to_scale_idx == -1:
                    new_block_sizes.append(block_size_each_image)
                else:
                    new_block_sizes.append(
                        (
                            scales[resize_output_to_scale_idx] // scales[0],
                            scales[resize_output_to_scale_idx] // scales[0],
                        )
                    )

        assert block_cnt == len(image_features)

        return image_features_each_image, new_block_sizes

    def encode_images(self, images, block_sizes: Optional[Optional[Tuple[int, ...]]] = None):
        if block_sizes is None:
            block_sizes = [None] * len(images)
        if getattr(self.config, "dynamic_s2", False):
            image_features = self.get_vision_tower()(images)
            image_features, new_block_sizes = self.merge_features_for_dynamic_s2(image_features, block_sizes)

            image_features = [
                self.split_chessboard(x, block_size[0], block_size[1])
                for x, block_size in zip(image_features, new_block_sizes)
            ]  # list of B * C * H * W tensors
            image_features = torch.cat(
                [rearrange(x, "b c h w -> b (h w) c") for x in image_features], dim=0
            )  # B * N * C
            image_features = self.get_mm_projector()(image_features)
            image_features = list(
                image_features.split([block_size[0] * block_size[1] for block_size in new_block_sizes], dim=0)
            )
            image_features = [
                self.merge_chessboard(x, block_size[0], block_size[1])
                for x, block_size in zip(image_features, new_block_sizes)
            ]  # list of 1 * C * H * W tensors
            image_features = [rearrange(x, "1 c h w -> (h w) c") for x in image_features]  # list of N * C tensors
            if all([feature.shape[0] == image_features[0].shape[0] for feature in image_features]):
                image_features = torch.stack(image_features, dim=0)
        else:
            # VILA는 기본적으로 여기로 들어가는 듯?
            # print(self.config)
            # images torch.Size([6, 3, 384, 384])
            image_features = self.get_vision_tower()(images) # 출력 tensor, torch.Size([6, 729, 1152])'
            print("DONE WITH VISION TOWER")
            new_image_feature = MASt3R().merge_features(images, image_features, )
            # print(self.config.__dict__.keys())
            print("DONE WITH MASt3R")
                            
            # print(f"llava>model>llava_arch.py>LlavaMetaModel>encode_images : {type(images)}, {np.shape(images)}")
            """
            해야함
            v 1. image feature 구성 어떻게 되어있는지 확인. --> torch.Size([6, 729, 1152])
            v 2. 기존 이미지 어딨는지 확인 후 normalize 필요한지 여부 확인. --> normalize 되어있는듯. print(self.config.__dict__.keys())로 확인
            v 3. 기존 이미지 등 mast3r 구성 요소 맞도록 변경작업
            v 4. mast3r 결과 나온거에서 patch 단위로 avg 진행
            v 5. avg한 패치 내 feature를 fastNN 알고리즘 통과
            v 6. 나온 매칭 결과 중 한 20% 정도 (나중에 확인 후 변경시키기) 기준 하나 잡아서 대체작업(아니면 avg 해서 대체할 것)
            v 7. (poisitional encoding은 안건드려도 됨. 일단 그대로 둘 것.)
            v 8. 그렇게 변경시킨 feature는 projector로 전달.
            """
            ############################################################# 여기에 merging-->일단 그렇게 진행.
            image_features = self.get_mm_projector()(new_image_feature) #출력 torch.Size([6, 196, 2560])
            print("DONE WITH mm_projector")
        # image_features = torch.Size([6, 196, 2560])
        return image_features

    ## @yunhao: is there a better way to handle function call and attributes for llm?
    ## support beam search
    def _temporary_reorder_cache(self, past_key_values, sorted_idx):
        return self.get_llm()._temporary_reorder_cache(past_key_values, sorted_idx)

    def get_input_embeddings(self):
        return self.get_llm().get_input_embeddings()

    def get_output_embeddings(self):
        return self.get_llm().get_output_embeddings()

    def resize_token_embeddings(self, embed_size):
        self.get_llm().resize_token_embeddings(embed_size)


class LlavaMetaForCausalLM(ABC):
    def _embed(
        self,
        input_ids: torch.Tensor,
        media: Dict[str, List[torch.Tensor]],
        media_config: Dict[str, Dict[str, Any]],
        labels: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        labels = labels if labels is not None else torch.full_like(input_ids, IGNORE_INDEX)
        attention_mask = attention_mask if attention_mask is not None else torch.ones_like(input_ids, dtype=torch.bool)

        # Extract text and media embeddings
        text_embeds = self.llm.model.embed_tokens(input_ids) #text_embeds torch.Size([1, 50, 2560])
        media_embeds = self.__embed_media_tokens(media, media_config) #media_embeds['image'][i].size() torch.Size([199, 2560])

        # This is a workaround to make sure the dummy embeddings are consumed
        while media_embeds.get("dummy"):
            dummy_embed = media_embeds["dummy"].popleft()
            text_embeds += torch.sum(dummy_embed) * 0

        # Remove padding
        batch_size = labels.shape[0]
        text_embeds = [text_embeds[k][attention_mask[k]] for k in range(batch_size)]
        labels = [labels[k][attention_mask[k]] for k in range(batch_size)]

        # Build inverse mapping from token ID to media name
        media_tokens = {}
        for name, token_id in self.tokenizer.media_token_ids.items():
            media_tokens[token_id] = name

        # Fuse text and media embeddings
        inputs_m, labels_m = [], []
        for k in range(batch_size):
            inputs_mk, labels_mk = [], []
            pos = 0
            while pos < len(labels[k]):
                if input_ids[k][pos].item() in media_tokens:
                    end = pos + 1
                    name = media_tokens[input_ids[k][pos].item()]
                    input = media_embeds[name].popleft() # 사이즈는 계속 유지됨
                    label = torch.full([input.shape[0]], IGNORE_INDEX, device=labels[k].device, dtype=labels[k].dtype)
                else:
                    end = pos
                    while end < len(labels[k]) and input_ids[k][end].item() not in media_tokens:
                        end += 1
                    input = text_embeds[k][pos:end]
                    label = labels[k][pos:end]
                inputs_mk.append(input)
                labels_mk.append(label)
                pos = end
            inputs_m.append(torch.cat(inputs_mk, dim=0))
            labels_m.append(torch.cat(labels_mk, dim=0))
        inputs, labels = inputs_m, labels_m

        # Check if all media embeddings are consumed
        for name in media_embeds:
            if media_embeds[name]:
                raise ValueError(f"Not all {name} embeddings are consumed!")

        # Truncate sequences to `model_max_length` as media embeddings are inserted
        inputs, labels = self.__truncate_sequence(inputs, labels)

        # Pad sequences to the longest one in the batch
        return self.__batchify_sequence(inputs, labels)

    def __embed_media_tokens(
        self,
        media: Dict[str, List[torch.Tensor]],
        media_config: Dict[str, Dict[str, Any]],
    ) -> Dict[str, List[torch.Tensor]]:
        embeds = defaultdict(deque) # defaultdict(<class 'collections.deque'>, {})
        for name in media:
            if self.training: # training 시
                # Gather metainfo of media objects from all ranks
                info = [{"shape": tensor.shape, "dtype": tensor.dtype} for tensor in media.get(name, [])]
                infos = list(chain(*dist.all_gather(info)))

                # The entire batch does not contain any media objects of this type.
                if not infos:
                    continue

                # Create a dummy tensor to ensure the encoder is called, otherwise the training will hang.
                if not media.get(name):
                    dummy = torch.zeros(infos[0]["shape"], dtype=infos[0]["dtype"], device=self.device)
                    embeds["dummy"].extend(self.encoders[name]([dummy], media_config[name]))
                    continue
            embeds[name] = deque(self.encoders[name](media[name], media_config[name]))
            # embeds['image'] 에 image의 각각의 tensor들 쌓임. len(embeds['image'])=6
        return embeds

    def __truncate_sequence(
        self, inputs: List[torch.Tensor], labels: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if any(len(input) > self.tokenizer.model_max_length for input in inputs):
            warnings.warn(f"Truncating sequences to `model_max_length` ({self.tokenizer.model_max_length}).")
            inputs = [input[: self.tokenizer.model_max_length] for input in inputs]
            labels = [label[: self.tokenizer.model_max_length] for label in labels]
        return inputs, labels

    def __batchify_sequence(
        self, inputs: List[torch.Tensor], labels: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = len(inputs)
        device = inputs[0].device
        hidden_size = inputs[0].shape[1]
        max_length = max(inputs[k].shape[0] for k in range(batch_size))
        attention_mask = torch.ones((batch_size, max_length), dtype=torch.bool, device=device)

        inputs_p, labels_p = [], []
        for k in range(batch_size):
            size_pk = max_length - inputs[k].shape[0]
            inputs_pk = torch.zeros((size_pk, hidden_size), dtype=inputs[k].dtype, device=device)
            labels_pk = torch.full((size_pk,), IGNORE_INDEX, dtype=labels[k].dtype, device=device)
            if self.tokenizer.padding_side == "right":
                attention_mask[k, inputs[k].shape[0] :] = False
                inputs_pk = torch.cat([inputs[k], inputs_pk], dim=0)
                labels_pk = torch.cat([labels[k], labels_pk], dim=0)
            else:
                attention_mask[k, : -inputs[k].shape[0]] = False
                inputs_pk = torch.cat([inputs_pk, inputs[k]], dim=0)
                labels_pk = torch.cat([labels_pk, labels[k]], dim=0)
            inputs_p.append(inputs_pk)
            labels_p.append(labels_pk)

        inputs = torch.stack(inputs_p, dim=0)
        labels = torch.stack(labels_p, dim=0)
        return inputs, labels, attention_mask

    def repack_multimodal_data(self, inputs_embeds, attention_mask, position_ids, labels):
        # Handle sequence parallelism
        PROCESS_GROUP_MANAGER = get_pg_manager()

        # We do re-sharding instead of packing here to ensure the sequence length is the same across all ranks.
        if PROCESS_GROUP_MANAGER is not None:
            sp_degree = PROCESS_GROUP_MANAGER.sp_degree
            sp_rank = PROCESS_GROUP_MANAGER.sp_rankFalse
            sp_group = PROCESS_GROUP_MANAGER.sp_pg
            ring_degree = PROCESS_GROUP_MANAGER.ring_degree
            ring_rank = PROCESS_GROUP_MANAGER.ring_rank
            ring_type = PROCESS_GROUP_MANAGER.ring_type
            ulysses_degree = PROCESS_GROUP_MANAGER.ulysses_degree
            ulysses_rank = PROCESS_GROUP_MANAGER.ulysses_rank

            bs, shard_seqlen = position_ids.shape
            sp_seq_len = [torch.zeros(1, dtype=torch.int64, device=position_ids.device) for _ in range(sp_degree)]
            dist.all_gather(sp_seq_len, torch.tensor(shard_seqlen, device=position_ids.device), group=sp_group)
            sp_seq_len_cat = torch.cat(sp_seq_len, dim=0)

            if sp_rank == 0:
                original_start_id = 0
            else:
                original_start_id = torch.sum(sp_seq_len_cat[:sp_rank]).item()
            original_end_id = torch.sum(sp_seq_len_cat[: sp_rank + 1]).item()

            # Gather attention_mask, position_ids, labels and input_embeds
            all_inputs_embeds = torch.zeros(
                bs,
                torch.sum(sp_seq_len_cat),
                inputs_embeds.shape[-1],
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            ).contiguous()
            all_inputs_embeds[:, original_start_id:original_end_id, :] += inputs_embeds
            dist.barrier(group=sp_group)
            dist.all_reduce(all_inputs_embeds, group=sp_group)
            dist.barrier(group=sp_group)

            attention_mask_list = [
                torch.zeros((bs, sp_seq_len[i]), dtype=attention_mask.dtype, device=attention_mask.device)
                for i in range(sp_degree)
            ]
            position_ids_list = [
                torch.zeros((bs, sp_seq_len[i]), dtype=position_ids.dtype, device=position_ids.device)
                for i in range(sp_degree)
            ]
            labels_list = [
                torch.zeros((bs, sp_seq_len[i]), dtype=labels.dtype, device=labels.device) for i in range(sp_degree)
            ]

            dist.all_gather(attention_mask_list, attention_mask, group=sp_group)
            dist.all_gather(position_ids_list, position_ids, group=sp_group)
            dist.all_gather(labels_list, labels, group=sp_group)

            effective_seqlen_list = [attention_mask_list[i].sum(dim=-1) for i in range(sp_degree)]
            effective_seqlen = torch.stack(effective_seqlen_list, dim=-1)
            effective_seqlen_batch_list = torch.unbind(effective_seqlen, dim=0)

            global_attention_mask_list = []
            global_position_ids_list = []
            global_labels_list = []
            global_inputs_embeds_list = []
            for i in range(bs):
                global_attention_mask_batch_list = []
                global_position_ids_batch_list = []
                global_labels_batch_list = []
                global_inputs_embeds_batch_list = []
                for j in range(sp_degree):
                    eff_len = effective_seqlen_batch_list[i][j]
                    prev_len = torch.sum(sp_seq_len_cat[:j]).item() if j > 0 else 0

                    global_attention_mask_batch_list.append(attention_mask_list[j][i, :eff_len])
                    global_position_ids_batch_list.append(position_ids_list[j][i, :eff_len])
                    global_labels_batch_list.append(labels_list[j][i, :eff_len])
                    global_inputs_embeds_batch_list.append(all_inputs_embeds[i, prev_len : prev_len + eff_len, :])
                global_attention_mask_list.append(torch.cat(global_attention_mask_batch_list, dim=0))
                global_position_ids_list.append(torch.cat(global_position_ids_batch_list, dim=0))
                global_labels_list.append(torch.cat(global_labels_batch_list, dim=0))
                global_inputs_embeds_list.append(torch.cat(global_inputs_embeds_batch_list, dim=0))

                global_attention_mask = torch.nn.utils.rnn.pad_sequence(
                    global_attention_mask_list, batch_first=True, padding_value=False
                )
                global_position_ids = torch.nn.utils.rnn.pad_sequence(
                    global_position_ids_list, batch_first=True, padding_value=-1
                )
                global_labels = torch.nn.utils.rnn.pad_sequence(
                    global_labels_list, batch_first=True, padding_value=IGNORE_INDEX
                )
                global_inputs_embeds = torch.nn.utils.rnn.pad_sequence(
                    global_inputs_embeds_list, batch_first=True, padding_value=0
                )

            # Re-shard the inputs
            if ring_degree > 1:
                total_effective_seqlen = torch.sum(effective_seqlen, dim=1)
                new_seqlen_per_rank = total_effective_seqlen // sp_degree
                assert torch.all(
                    total_effective_seqlen % sp_degree == 0
                ), "total_effective_seqlen must be divisible by sp_degree"

                max_new_seqlen = torch.max(new_seqlen_per_rank).item()

                new_attention_mask = torch.zeros(
                    (bs, max_new_seqlen), dtype=global_attention_mask.dtype, device=global_attention_mask.device
                )
                new_position_ids = torch.zeros(
                    (bs, max_new_seqlen), dtype=global_position_ids.dtype, device=global_position_ids.device
                )
                new_labels = torch.full(
                    (bs, max_new_seqlen), IGNORE_INDEX, dtype=global_labels.dtype, device=global_labels.device
                )
                new_inputs_embeds = torch.zeros(
                    (bs, max_new_seqlen, global_inputs_embeds.shape[-1]),
                    dtype=global_inputs_embeds.dtype,
                    device=global_inputs_embeds.device,
                )

                if ring_type == "ring_varlen":
                    for i in range(bs):
                        start_idx = new_seqlen_per_rank[i] * sp_rank
                        end_idx = start_idx + new_seqlen_per_rank[i]
                        new_attention_mask[i, : new_seqlen_per_rank[i]] = global_attention_mask[i, start_idx:end_idx]
                        new_position_ids[i, : new_seqlen_per_rank[i]] = global_position_ids[i, start_idx:end_idx]
                        new_labels[i, : new_seqlen_per_rank[i]] = global_labels[i, start_idx:end_idx]
                        new_inputs_embeds[i, : new_seqlen_per_rank[i], :] = global_inputs_embeds[
                            i, start_idx:end_idx, :config
                        ]
                elif ring_type == "zigzag_ring_varlen":
                    chunk_size = total_effective_seqlen // (2 * sp_degree)
                    for i in range(bs):
                        # Zigzag pattern indices
                        if sp_degree == ring_degree:
                            forward_rank_idx = sp_rank
                            backward_rank_idx = 2 * sp_degree - sp_rank - 1
                        else:
                            ulysses_offset = ulysses_rank * ring_degree * 2
                            forward_rank_idx = ring_rank + ulysses_offset
                            backward_rank_idx = sp_degree - ring_rank - 1 + ulysses_offset

                        # Calculate start and end indices for the forward and backward zigzag
                        start_idx_fwd = forward_rank_idx * chunk_size[i]
                        end_idx_fwd = start_idx_fwd + chunk_size[i]

                        start_idx_bwd = backward_rank_idx * chunk_size[i]
                        end_idx_bwd = start_idx_bwd + chunk_size[i]

                        # Fill new tensors with zigzag data
                        new_attention_mask[i, : chunk_size[i]] = global_attention_mask[i, start_idx_fwd:end_idx_fwd]
                        new_attention_mask[i, chunk_size[i] : 2 * chunk_size[i]] = global_attention_mask[
                            i, start_idx_bwd:end_idx_bwd
                        ]

                        new_position_ids[i, : chunk_size[i]] = global_position_ids[i, start_idx_fwd:end_idx_fwd]
                        new_position_ids[i, chunk_size[i] : 2 * chunk_size[i]] = global_position_ids[
                            i, start_idx_bwd:end_idx_bwd
                        ]

                        new_labels[i, : chunk_size[i]] = global_labels[i, start_idx_fwd:end_idx_fwd]
                        new_labels[i, chunk_size[i] : 2 * chunk_size[i]] = global_labels[i, start_idx_bwd:end_idx_bwd]

                        new_inputs_embeds[i, : chunk_size[i], :] = global_inputs_embeds[i, start_idx_fwd:end_idx_fwd, :]
                        new_inputs_embeds[i, chunk_size[i] : 2 * chunk_size[i], :] = global_inputs_embeds[
                            i, start_idx_bwd:end_idx_bwd, :
                        ]
                else:
                    raise ValueError(f"Invalid ring_type: {ring_type}")
            else:
                global_seq_len = global_attention_mask.shape[-1]
                seq_len_sharded = global_seq_len // sp_degree
                start_idx_reshard = seq_len_sharded * sp_rank
                end_idx_reshard = start_idx_reshard + seq_len_sharded if sp_rank < sp_degree - 1 else global_seq_len

                new_attention_mask = torch.narrow(
                    global_attention_mask, 1, start_idx_reshard, end_idx_reshard - start_idx_reshard
                )
                new_position_ids = torch.narrow(
                    global_position_ids, 1, start_idx_reshard, end_idx_reshard - start_idx_reshard
                )
                new_labels = torch.narrow(global_labels, 1, start_idx_reshard, end_idx_reshard - start_idx_reshard)
                new_inputs_embeds = torch.narrow(
                    global_inputs_embeds, 1, start_idx_reshard, end_idx_reshard - start_idx_reshard
                )

            return new_inputs_embeds, new_attention_mask, new_position_ids, new_labels

        device = inputs_embeds.device
        batch_size = inputs_embeds.shape[0]
        seqlens = [attention_mask[k].sum().item() for k in range(batch_size)]

        # Pack all sequences together
        inputs_embeds_p = [inputs_embeds[k][attention_mask[k]] for k in range(batch_size)]
        attention_mask_p = [torch.ones(seqlens[k], dtype=torch.int, device=device) for k in range(batch_size)]
        position_ids_p = [torch.arange(seqlens[k], dtype=torch.int, device=device) for k in range(batch_size)]
        labels_p = [labels[k][attention_mask[k]] for k in range(batch_size)]

        # Add one dummy token at the end of the packed sequence to ensure that `_get_unpacked_data` will be called
        inputs_embeds_p.append(torch.zeros(1, inputs_embeds.shape[-1], dtype=inputs_embeds.dtype, device=device))
        attention_mask_p.append(torch.tensor([0], dtype=torch.int, device=device))
        position_ids_p.append(torch.tensor([0], dtype=torch.int, device=device))
        labels_p.append(torch.tensor([IGNORE_INDEX], dtype=torch.int, device=device))

        # Mask the first token of each sequence to avoid contamination
        for label in labels_p:
            label[0] = IGNORE_INDEX

        # Batch the data
        inputs_embeds_p = torch.cat(inputs_embeds_p, dim=0).unsqueeze(0)
        attention_mask_p = torch.cat(attention_mask_p, dim=0).unsqueeze(0)
        position_ids_p = torch.cat(position_ids_p, dim=0).unsqueeze(0)
        labels_p = torch.cat(labels_p, dim=0).unsqueeze(0)

        if hasattr(
            self, "pad_to_multiple_of"
        ):  # related to quantization, please refer to ModelArguments for more information.
            assert len(labels_p.shape) == 2
            batch_size, max_length, cur_length = labels_p.shape[0], labels_p.shape[1], labels_p.shape[1]
            hidden_size = inputs_embeds_p.shape[-1]

            if max_length % self.pad_to_multiple_of != 0:
                max_length = ((max_length // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of
                difference = max_length - cur_length

                inputs_embeds_p = torch.cat(
                    (
                        inputs_embeds_p,
                        torch.full((batch_size, difference, hidden_size), self.llm.pad_token_id).to(inputs_embeds_p),
                    ),
                    dim=1,
                )
                labels_p = torch.cat((labels_p, torch.full((batch_size, difference), IGNORE_INDEX).to(labels_p)), dim=1)
                attention_mask_p = torch.cat(
                    (
                        attention_mask_p,
                        torch.zeros((batch_size, difference), dtype=torch.bool).to(attention_mask_p),
                    ),
                    dim=1,
                )
                position_ids_p = torch.cat(
                    (position_ids_p, torch.full((batch_size, difference), -1).to(position_ids_p)), dim=1
                )

        return inputs_embeds_p, attention_mask_p, position_ids_p, labels_p

    def get_xgr_logits_processor(self, response_format: ResponseFormat) -> List[LogitsProcessor]:
        # Convert response format to logits processor
        import xgrammar as xgr

        logging.info("[XGrammar] Compiling grammar for contrained output")

        if self.grammar_compiler is None:
            self.grammar_compiler = xgr.GrammarCompiler(
                xgr.TokenizerInfo.from_huggingface(self.tokenizer, vocab_size=self.vocab_size)
            )

        if response_format.type == "json_schema":
            compiled_grammar = self.grammar_compiler.compile_json_schema(
                response_format.json_schema.schema_,
                indent=2,
            )
        else:
            compiled_grammar = self.grammar_compiler.compile_builtin_json_grammar()

        return [xgr.contrib.hf.LogitsProcessor(compiled_grammar)]

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Optional[torch.FloatTensor] = None,
        media: Optional[Dict[str, List[torch.Tensor]]] = None,
        media_config: Dict[str, Dict[str, Any]] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **generation_kwargs,
    ):
        inputs_embeds, _, attention_mask = self._embed(input_ids, media, media_config, None, attention_mask)

        return self.llm.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **generation_kwargs)

    @torch.inference_mode()
    def generate_content( #이게 llava>cli>infer.py 에서 모델 불러온 다음에 바로 실행됨.
        self,
        prompt: Union[str, List],
        generation_config: Optional[GenerationConfig] = None,
        response_format: Optional[ResponseFormat] = None,
    ) -> str:
        # TODO(zhijianl): Support directly taking conversation as input
        conversation = [{"from": "human", "value": prompt}]

        # Convert response format to logits processor
        if response_format:
            xgr_logits_processor = self.get_xgr_logits_processor(response_format)
        else:
            xgr_logits_processor = None

        # Extract media from the conversation

        # TODO (extract and preprocess should be done together, as the preprocess of image and video can be different, i.e. when dynamic res is used)
        media = extract_media(conversation, self.config)
        # media에는 이미지 6장 들어왔으면 dict 형태로 저장되어있음
        # defaultdict(<class 'list'>, {'image': [<PIL.PngImagePlugin.PngImageFile image mode=RGB size=256x192 at 0x70E241BDF6D0>, 
        # <PIL.PngImagePlugin.PngImageFile image mode=RGB size=256x192 at 0x70E241BDD0F0>, ... 이렇게 6개]})

        # Process media
        media_config = defaultdict(dict)
        for name in media:
            if name == "image":
                if len(media["image"]) == 1 and self.config.image_aspect_ratio in ["dynamic", "dynamic_s2"]:
                    self.config.image_processor = self.vision_tower.image_processor
                    if self.config.image_aspect_ratio == "dynamic":
                        images = process_image(media["image"][0], self.config, None, enable_dynamic_res=True).half()
                        conversation[0]["value"] = conversation[0]["value"].replace(
                            DEFAULT_IMAGE_TOKEN, f"{DEFAULT_IMAGE_TOKEN}\n" * images.shape[0]
                        )
                    else:
                        if type(self.config.s2_scales) is str:
                            self.config.s2_scales = list(map(int, self.config.s2_scales.split(",")))
                        images, block_sizes = process_image(
                            media["image"][0], self.config, None, enable_dynamic_s2=True
                        )
                        images = images.half()
                        media_config[name]["block_sizes"] = [block_sizes]
                else: #이미지 여러 장 들어왔을 때
                    images = process_images(media["image"], self.vision_tower.image_processor, self.config).half()
                    # images는 temsor 형태. 6장의 이미지가 들어왔으면 torch.Size([6, 3, 384, 384])
                media[name] = [image for image in images]
                # print((media["image"][0].size())) 하면 torch.Size([3, 384, 384]) 나옴
            elif name == "video":
                media[name] = [
                    process_images(images, self.vision_tower.image_processor, self.config).half()
                    for images in media[name]
                ]
            else:
                raise ValueError(f"Unsupported media type: {name}")
        # Tokenize the conversation
        input_ids = tokenize_conversation(conversation, self.tokenizer, add_generation_prompt=True).cuda().unsqueeze(0)
        # input_ids= tensor([[    1,   319, 13563,  1546,   263, 12758,  1404,   322,   385, 23116,...
        # conversation = [{'from': 'human', 'value': '<image><image><image><image><image><image>Please describe the image'}, {'from': 'gpt', 'value': None}]
        
        # Set up the generation config
        generation_config = generation_config or self.default_generation_config

        # Generate the response
        try:
            output_ids = self.generate(
                input_ids=input_ids,
                media=media,
                media_config=media_config,
                generation_config=generation_config,
                logits_processor=xgr_logits_processor,  # structured generation
            )
        except ValueError:
            if not generation_config.do_sample:
                raise
            # FIXME(zhijianl): This is a temporary workaround for the sampling issue
            logging.warning("Generation failed with sampling, retrying with greedy decoding.")
            generation_config.do_sample = False
            output_ids = self.generate(
                input_ids=input_ids,
                media=media,
                media_config=media_config,
                generation_config=generation_config,
                logits_processor=xgr_logits_processor,
            )

        # Decode the response
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        return response

    @property
    def default_generation_config(self) -> GenerationConfig:
        generation_config = copy.deepcopy(self.generation_config or GenerationConfig())
        if self.tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must have an EOS token")
        if generation_config.max_length == GenerationConfig().max_length:
            generation_config.max_length = self.tokenizer.model_max_length
        if generation_config.pad_token_id is None:
            generation_config.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        if generation_config.bos_token_id is None:
            generation_config.bos_token_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        if generation_config.eos_token_id is None:
            generation_config.eos_token_id = self.tokenizer.stop_token_ids
        return generation_config


