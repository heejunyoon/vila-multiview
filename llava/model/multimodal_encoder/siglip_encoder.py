# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import torch
from transformers import PretrainedConfig, SiglipImageProcessor
"""
    Constructs a SigLIP image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by
            `do_resize` in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"height": 224, "width": 224}`):
            Size of the image after resizing. Can be overridden by `size` in the `preprocess` method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            Resampling filter to use if resizing the image. Can be overridden by `resample` in the `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by `do_rescale` in
            the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by `rescale_factor` in the `preprocess`
            method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image by the specified mean and standard deviation. Can be overridden by
            `do_normalize` in the `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `[0.5, 0.5, 0.5]`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `[0.5, 0.5, 0.5]`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
    """
from llava.model.multimodal_encoder.vision_encoder import VisionTower, VisionTowerDynamicS2, VisionTowerS2

from .siglip import SiglipVisionModel


class SiglipVisionTower(VisionTower):
    def __init__(self, model_name_or_path: str, config: PretrainedConfig) -> None:
        super().__init__(model_name_or_path, config)
        print("siglip_encoder.py > SiglipVisionTower")
        # TODO(ligengl): why pass config here leading to errors?
        self.vision_tower = SiglipVisionModel.from_pretrained(
            model_name_or_path,
            attn_implementation="flash_attention_2",
            torch_dtype=eval(config.model_dtype),
        )
        self.image_processor = SiglipImageProcessor.from_pretrained(model_name_or_path)
        self.is_loaded = True


class SiglipVisionTowerS2(VisionTowerS2):
    def __init__(self, model_name_or_path: str, config: PretrainedConfig) -> None:
        super().__init__(model_name_or_path, config)
        self.vision_tower = SiglipVisionModel.from_pretrained(
            model_name_or_path,
            attn_implementation="flash_attention_2",
            torch_dtype=eval(config.model_dtype),
        )
        self.image_processor = SiglipImageProcessor.from_pretrained(model_name_or_path)
        # Make sure it crops/resizes the image to the largest scale in self.scales to maintain high-res information
        self.image_processor.size["height"] = self.image_processor.size["width"] = self.scales[-1]
        self.is_loaded = True


class SiglipVisionTowerDynamicS2(VisionTowerDynamicS2):
    def __init__(self, model_name_or_path: str, config: PretrainedConfig) -> None:
        super().__init__(model_name_or_path, config)
        self.vision_tower = SiglipVisionModel.from_pretrained(
            model_name_or_path,
            attn_implementation="flash_attention_2",
            torch_dtype=eval(config.model_dtype),
        )
        self.image_processor = SiglipImageProcessor.from_pretrained(model_name_or_path)
        # Make sure it crops/resizes the image to the largest scale in self.scales to maintain high-res information
        self.image_processor.size["height"] = self.image_processor.size["width"] = self.scales[0]
        self.is_loaded = True


