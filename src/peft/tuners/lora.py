# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
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
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from ..import_utils import is_bnb_available
from ..utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    PeftConfig,
    PeftType,
    _freeze_adapter,
    _get_submodules,
    transpose,
)

if is_bnb_available():
    import bitsandbytes as bnb


@dataclass
class LoraConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`LoraModel`].

    Args:
        r (`int`): Lora attention dimension.
        target_modules (`Union[List[str],str]`): The names of the modules to apply Lora to.
        lora_alpha (`float`): The alpha parameter for Lora scaling.
        lora_dropout (`float`): The dropout probability for Lora layers.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out).
        For example, gpt-2 uses `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.:
        bias (`str`): Bias type for Lora. Can be 'none', 'all' or 'lora_only'
        modules_to_save (`List[str]`):List of modules apart from LoRA layers to be set as trainable
            and saved in the final checkpoint.
    """

    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
                    "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    lora_alpha: int = field(default=None, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=None, metadata={"help": "Lora dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    bias: str = field(default="none", metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
                    "For example, in Sequence Classification or Token Classification tasks, "
                    "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    init_lora_weights: bool = field(
        default=True,
        metadata={"help": "Whether to initialize the weights of the Lora layers."},
    )
    r_sum: int = field(default=0)  # modified. This argument represents the dim of the previous LoRA parameters.
    save_loranew: bool = field(
        default=False)  # modified. This arguments represents whether modules named of 'loranew_A/B' are saved independently, rather than being combined with "lora_A/B".

    def __post_init__(self):
        self.peft_type = PeftType.LORA


'''
class LoraModel(torch.nn.Module):
    """
    Creates Low Rank Adapter (Lora) model from a pretrained transformers model.

    Args:
        model ([`~transformers.PreTrainedModel`]): The model to be adapted.
        config ([`LoraConfig`]): The configuration of the Lora model.

    Returns:
        `torch.nn.Module`: The Lora model.

    Example:

        ```py
        >>> from transformers import AutoModelForSeq2SeqLM, LoraConfig
        >>> from peft import LoraModel, LoraConfig

        >>> config = LoraConfig(
        ...     peft_type="LORA",
        ...     task_type="SEQ_2_SEQ_LM",
        ...     r=8,
        ...     lora_alpha=32,
        ...     target_modules=["q", "v"],
        ...     lora_dropout=0.01,
        ... )

        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        >>> lora_model = LoraModel(config, model)
        ```

        ```py
        >>> import transformers
        >>> from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_int8_training

        >>> target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out", "wte"]
        >>> config = LoraConfig(
        ...     r=4, lora_alpha=16, target_modules=target_modules, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"
        ... )

        >>> model = transformers.GPTJForCausalLM.from_pretrained(
        ...     "kakaobrain/kogpt",
        ...     revision="KoGPT6B-ryan1.5b-float16",  # or float32 version: revision=KoGPT6B-ryan1.5b
        ...     pad_token_id=tokenizer.eos_token_id,
        ...     use_cache=False,
        ...     device_map={"": rank},
        ...     torch_dtype=torch.float16,
        ...     load_in_8bit=True,
        ... )
        >>> model = prepare_model_for_int8_training(model)
        >>> lora_model = get_peft_model(model, config)
        ```

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`LoraConfig`]): The configuration of the Lora model.
    """

    def __init__(self, model, config, adapter_name):
        super().__init__()
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config

        # 如果是 Baseline LoRA 模式，使用共享的适配器名称 'baseline_lora'
        if config.get("use_baseline_lora", False):
            adapter_name = "baseline_lora"

        if adapter_name not in self.peft_config:
            self.add_adapter(adapter_name, self.peft_config[adapter_name])

    def get_adapter_name(self, adapter_name):
        return adapter_name or "baseline_lora"

    def add_adapter(self, adapter_name, config=None):
        # 检查是否已经存在该适配器
        if adapter_name in self.peft_config:
            warnings.warn(f"Adapter {adapter_name} already exists, skipping creation.")
            return

        # 如果配置不为空，准备 LoRA 配置
        if config is not None:
            model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
            config = self._prepare_lora_config(config, model_config)
            self.peft_config[adapter_name] = config

        # 找到目标模块并替换为 LoRA 版本
        self._find_and_replace(adapter_name)

        # 如果有多个适配器，不允许带有 bias 的适配器
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError(
                "LoraModel supports only 1 adapter with bias. When using multiple adapters, set bias to 'none' for all adapters."
            )
        # 设置为仅训练 LoRA 层
        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)

        # 如果是推理模式，冻结适配器
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)

    def _find_and_replace(self, adapter_name):
        lora_config = self.peft_config[adapter_name]
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if loaded_in_8bit and not is_bnb_available():
            raise ImportError(
                "To use Lora with 8-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )
        is_target_modules_in_base_model = False
        kwargs = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
        }
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(lora_config.target_modules, str):
                target_module_found = re.fullmatch(lora_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in lora_config.target_modules)
            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = _get_submodules(self.model, key)
                if hasattr(target, "bias"):
                    bias = target.bias is not None

                if isinstance(target, LoraLayer):
                    target.update_layer(
                        adapter_name,
                        lora_config.r,
                        lora_config.lora_alpha,
                        lora_config.lora_dropout,
                        lora_config.init_lora_weights,
                    )
                else:
                    if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
                        eightbit_kwargs = kwargs.copy()
                        eightbit_kwargs.update(
                            {
                                "has_fp16_weights": target.state.has_fp16_weights,
                                "memory_efficient_backward": target.state.memory_efficient_backward,
                                "threshold": target.state.threshold,
                                "index": target.index,
                            }
                        )
                        new_module = Linear8bitLt(
                            adapter_name, target.in_features, target.out_features, bias=bias, **eightbit_kwargs
                        )
                    elif isinstance(target, torch.nn.Embedding):
                        embedding_kwargs = kwargs.copy()
                        embedding_kwargs.pop("fan_in_fan_out", None)
                        in_features, out_features = target.num_embeddings, target.embedding_dim
                        new_module = Embedding(adapter_name, in_features, out_features, **embedding_kwargs)
                    else:
                        if isinstance(target, torch.nn.Linear):
                            in_features, out_features = target.in_features, target.out_features
                            if kwargs["fan_in_fan_out"]:
                                warnings.warn(
                                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                                    "Setting fan_in_fan_out to False."
                                )
                                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
                        elif isinstance(target, Conv1D):
                            in_features, out_features = (
                                target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                            )
                            if not kwargs["fan_in_fan_out"]:
                                warnings.warn(
                                    "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                                    "Setting fan_in_fan_out to True."
                                )
                                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
                        else:
                            raise ValueError(
                                f"Target module {target} is not supported. "
                                f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
                            )
                        new_module = Linear(adapter_name, in_features, out_features, bias=bias, r_sum=lora_config.r_sum, **kwargs) # modified

                    self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {lora_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if hasattr(old_module, "bias"):
            if old_module.bias is not None:
                new_module.bias = old_module.bias

        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name):

        adapter_name = adapter_name or "baseline_lora"

        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.active_adapter = adapter_name

    def merge_adapter(self, adapter_name=None):
        # 确保适配器名称为 'baseline_lora'
        adapter_name = adapter_name or "baseline_lora"

        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                module.merge()

    def unmerge_adapter(self, adapter_name=None):
        # 确保适配器名称为 'baseline_lora'
        adapter_name = adapter_name or "baseline_lora"

        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                module.unmerge()

    @staticmethod
    def _prepare_lora_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
        if peft_config.inference_mode:
            peft_config.merge_weights = True
        return peft_config

    def merge_and_unload(self):
        r"""
        This method merges the LoRa layers into the base model. This is needed if someone wants to use the base model
        as a standalone model.
        """
        if getattr(self.config, "model_type", None) == "gpt2":
            raise ValueError("GPT2 models are not supported for merging LORA layers")

        if getattr(self.model, "is_loaded_in_8bit", False):
            raise ValueError("Cannot merge LORA layers when the model is loaded in 8-bit mode")

        key_list = [key for key, _ in self.model.named_modules() if "lora" not in key]
        for key in key_list:
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            if isinstance(target, LoraLayer):
                bias = target.bias is not None
                new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
                target.merge()
                self._replace_module(parent, target_name, new_module, target)

            # save any additional trainable modules part of `modules_to_save`
            if isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model

    def add_weighted_adapter(self, adapters, weights, adapter_name):
        if len({self.peft_config[adapter].r for adapter in adapters}) != 1:
            raise ValueError("All adapters must have the same r value")
        self.peft_config[adapter_name] = self.peft_config[adapters[0]]
        self.peft_config[adapter_name].lora_alpha = self.peft_config[adapters[0]].r
        self._find_and_replace(adapter_name)
        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)
        _freeze_adapter(self.model, adapter_name)
        key_list = [key for key, _ in self.model.named_modules() if "lora" not in key]
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, LoraLayer):
                if adapter_name in target.lora_A:
                    target.lora_A[adapter_name].weight.data = target.lora_A[adapter_name].weight.data * 0.0
                    target.lora_B[adapter_name].weight.data = target.lora_B[adapter_name].weight.data * 0.0
                    for adapter, weight in zip(adapters, weights):
                        if adapter not in target.lora_A:
                            continue
                        target.lora_A[adapter_name].weight.data += (
                            target.lora_A[adapter].weight.data * weight * target.scaling[adapter]
                        )
                        target.lora_B[adapter_name].weight.data += target.lora_B[adapter].weight.data * weight

                elif adapter_name in target.lora_embedding_A:
                    target.lora_embedding_A[adapter_name].data = target.lora_embedding_A[adapter_name].data * 0.0
                    target.lora_embedding_B[adapter_name].data = target.lora_embedding_B[adapter_name].data * 0.0
                    for adapter, weight in zip(adapters, weights):
                        if adapter not in target.lora_embedding_A:
                            continue
                        target.lora_embedding_A[adapter_name].data += (
                            target.lora_embedding_A[adapter].data * weight * target.scaling[adapter]
                        )
                        target.lora_embedding_B[adapter_name].data += target.lora_embedding_B[adapter].data * weight
'''


class LoraModel(torch.nn.Module):
    """
    Creates Low Rank Adapter (Lora) model from a pretrained transformers model.
    """

    def __init__(self, model, config, adapter_name):
        super().__init__()
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config

        # 集中化适配器名称逻辑
        adapter_name = self._get_adapter_name(adapter_name)

        if adapter_name not in self.peft_config:
            self.add_adapter(adapter_name, self.peft_config.get(adapter_name))

    def _get_adapter_name(self, adapter_name):
        """
        Helper function to get the default adapter name if not provided.
        """
        return adapter_name or "baseline_lora"

    def add_adapter(self, adapter_name, config=None):
        # 检查是否已经存在该适配器
        if adapter_name in self.peft_config:
            warnings.warn(f"Adapter {adapter_name} already exists, skipping creation.")
            return

        # 如果配置为空，抛出异常
        if config is None:
            raise ValueError(f"Configuration for adapter {adapter_name} must be provided.")

        # 准备 LoRA 配置
        model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
        config = self._prepare_lora_config(config, model_config)
        self.peft_config[adapter_name] = config

        # 找到目标模块并替换为 LoRA 版本
        self._find_and_replace(adapter_name)

        # 如果有多个适配器，不允许带有 bias 的适配器
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError(
                "LoraModel supports only 1 adapter with bias. When using multiple adapters, set bias to 'none' for all adapters."
            )

        # 设置为仅训练 LoRA 层
        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)

        # 如果是推理模式，冻结适配器
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)

    def _find_and_replace(self, adapter_name):
        lora_config = self.peft_config[adapter_name]
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)

        if loaded_in_8bit and not is_bnb_available():
            raise ImportError(
                "To use Lora with 8-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )

        is_target_modules_in_base_model = False
        kwargs = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
        }
        key_list = [key for key, _ in self.model.named_modules()]

        for key in key_list:
            if isinstance(lora_config.target_modules, str):
                target_module_found = re.fullmatch(lora_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in lora_config.target_modules)

            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True

                parent, target, target_name = _get_submodules(self.model, key)

                if hasattr(target, "bias"):
                    bias = target.bias is not None

                if isinstance(target, LoraLayer):
                    target.update_layer(
                        adapter_name,
                        lora_config.r,
                        lora_config.lora_alpha,
                        lora_config.lora_dropout,
                        lora_config.init_lora_weights,
                    )
                else:
                    new_module = self._create_new_module(target, adapter_name, kwargs, bias, loaded_in_8bit)
                    self._replace_module(parent, target_name, new_module, target)

        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {lora_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _create_new_module(self, target, adapter_name, kwargs, bias, loaded_in_8bit):
        """
        Helper function to create a new module based on the target type.
        """
        if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
            eightbit_kwargs = kwargs.copy()
            eightbit_kwargs.update(
                {
                    "has_fp16_weights": target.state.has_fp16_weights,
                    "memory_efficient_backward": target.state.memory_efficient_backward,
                    "threshold": target.state.threshold,
                    "index": target.index,
                }
            )
            return Linear8bitLt(adapter_name, target.in_features, target.out_features, bias=bias, **eightbit_kwargs)
        elif isinstance(target, torch.nn.Embedding):
            embedding_kwargs = kwargs.copy()
            embedding_kwargs.pop("fan_in_fan_out", None)
            in_features, out_features = target.num_embeddings, target.embedding_dim
            return Embedding(adapter_name, in_features, out_features, **embedding_kwargs)
        elif isinstance(target, torch.nn.Linear):
            in_features, out_features = target.in_features, target.out_features
            if kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = False
            return Linear(adapter_name, in_features, out_features, bias=bias, **kwargs)
        elif isinstance(target, Conv1D):
            in_features, out_features = (
                target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
            )
            if not kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                    "Setting fan_in_fan_out to True."
                )
                kwargs["fan_in_fan_out"] = True
            return Linear(adapter_name, in_features, out_features, bias=bias, **kwargs)
        else:
            raise ValueError(
                f"Target module {target} is not supported. "
                f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
            )

    def merge_adapter(self, adapter_name=None):
        # 默认对所有适配器进行操作
        adapter_names = [adapter_name] if adapter_name else list(self.peft_config.keys())

        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                for adapter in adapter_names:
                    module.merge(adapter)

    def unmerge_adapter(self, adapter_name=None):
        # 默认对所有适配器进行操作
        adapter_names = [adapter_name] if adapter_name else list(self.peft_config.keys())

        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                for adapter in adapter_names:
                    module.unmerge(adapter)

    # 其余方法保留，若有进一步修改的需求再调整

    @staticmethod
    def _prepare_lora_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
        if peft_config.inference_mode:
            peft_config.merge_weights = True
        return peft_config

    def merge_and_unload(self):
        r"""
        This method merges the LoRa layers into the base model. This is needed if someone wants to use the base model
        as a standalone model.
        """
        if getattr(self.config, "model_type", None) == "gpt2":
            raise ValueError("GPT2 models are not supported for merging LORA layers")

        if getattr(self.model, "is_loaded_in_8bit", False):
            raise ValueError("Cannot merge LORA layers when the model is loaded in 8-bit mode")

        key_list = [key for key, _ in self.model.named_modules() if "lora" not in key]
        for key in key_list:
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            if isinstance(target, LoraLayer):
                bias = target.bias is not None
                new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
                target.merge()
                self._replace_module(parent, target_name, new_module, target)

            # save any additional trainable modules part of `modules_to_save`
            if isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model

    def add_weighted_adapter(self, adapters, weights, adapter_name):
        if len({self.peft_config[adapter].r for adapter in adapters}) != 1:
            raise ValueError("All adapters must have the same r value")
        self.peft_config[adapter_name] = self.peft_config[adapters[0]]
        self.peft_config[adapter_name].lora_alpha = self.peft_config[adapters[0]].r
        self._find_and_replace(adapter_name)
        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)
        _freeze_adapter(self.model, adapter_name)
        key_list = [key for key, _ in self.model.named_modules() if "lora" not in key]
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, LoraLayer):
                if adapter_name in target.lora_A:
                    target.lora_A[adapter_name].weight.data = target.lora_A[adapter_name].weight.data * 0.0
                    target.lora_B[adapter_name].weight.data = target.lora_B[adapter_name].weight.data * 0.0
                    for adapter, weight in zip(adapters, weights):
                        if adapter not in target.lora_A:
                            continue
                        target.lora_A[adapter_name].weight.data += (
                            target.lora_A[adapter].weight.data * weight * target.scaling[adapter]
                        )
                        target.lora_B[adapter_name].weight.data += target.lora_B[adapter].weight.data * weight

                elif adapter_name in target.lora_embedding_A:
                    target.lora_embedding_A[adapter_name].data = target.lora_embedding_A[adapter_name].data * 0.0
                    target.lora_embedding_B[adapter_name].data = target.lora_embedding_B[adapter_name].data * 0.0
                    for adapter, weight in zip(adapters, weights):
                        if adapter not in target.lora_embedding_A:
                            continue
                        target.lora_embedding_A[adapter_name].data += (
                            target.lora_embedding_A[adapter].data * weight * target.scaling[adapter]
                        )
                        target.lora_embedding_B[adapter_name].data += target.lora_embedding_B[adapter].data * weight

# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


# had to adapt it for `lora_only` to work


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    """Freeze all modules except LoRA's and depending on 'bias' value unfreezes bias weights.

    Args:
        model: model with LoRA layers
        bias:
            ``"none"``: all bias weights will be frozen,
            ``"lora_only"``: only bias weight for LoRA layers will be unfrozen,
            ``"all"``: all bias weights will be unfrozen.

    Raises:
        NotImplementedError: if `bias` not in ["none", "lora_only", "all"]
    """
    # freeze all layers except LoRA's
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False

    # depending on the `bias` value unfreeze bias weights
    if bias == "none":
        return
    if bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


class LoraLayer:
    def __init__(
            self,
            in_features: int,
            out_features: int,
    ):
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        self.loranew_A = nn.ModuleDict({})  # modified
        self.loranew_B = nn.ModuleDict({})  # modified
        # For Embedding layer
        self.lora_embedding_A = nn.ParameterDict({})
        self.lora_embedding_B = nn.ParameterDict({})
        # Mark the weight as unmerged
        self.merged = False
        self.disable_adapters = False
        self.in_features = in_features
        self.out_features = out_features

    def _initialize_adapter(self, adapter_name, r, lora_alpha, lora_dropout, r_sum, init_lora_weights):
        # 设置 LoRA 参数
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        self.scaling[adapter_name] = lora_alpha / r

        # 设置 Dropout 层
        self.lora_dropout[adapter_name] = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        # 设置 trainable 参数
        if r > 0:
            self.loranew_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
            self.loranew_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)
            self.lora_A[adapter_name] = nn.Linear(self.in_features, r_sum, bias=False)
            self.lora_B[adapter_name] = nn.Linear(r_sum, self.out_features, bias=False)

        # 初始化 LoRA 权重
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, r_sum,
                     use_baseline_lora=False):  # modified
        # 如果适配器已经存在，则直接跳过
        if use_baseline_lora and "baseline_lora" in self.lora_A:
            adapter_name = "baseline_lora"
            # 更新适配器的超参数
            self.r[adapter_name] = r
            self.lora_alpha[adapter_name] = lora_alpha
            self.scaling[adapter_name] = lora_alpha / r

            # 更新 dropout 层
            if lora_dropout > 0.0:
                self.lora_dropout[adapter_name] = nn.Dropout(p=lora_dropout)
            else:
                self.lora_dropout[adapter_name] = nn.Identity()

            return

        self._initialize_adapter(adapter_name, r, lora_alpha, lora_dropout, r_sum, init_lora_weights)

        self.to(self.weight.device)

    def update_layer_embedding(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights,
                               use_baseline_lora=False):
        # 如果使用 Baseline LoRA，强制使用共享的适配器名称
        if use_baseline_lora:
            adapter_name = "baseline_lora"

        # 检查是否已经存在适配器，避免重复初始化
        if adapter_name in self.lora_embedding_A:
            warnings.warn(f"Adapter {adapter_name} already exists in embedding, skipping creation.")
            return

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha

        # 设置 dropout 层
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))

        # 实际可训练参数的初始化
        if r > 0:
            self.lora_embedding_A.update(
                nn.ParameterDict({adapter_name: nn.Parameter(self.weight.new_zeros((r, self.in_features)))})
            )
            self.lora_embedding_B.update(
                nn.ParameterDict({adapter_name: nn.Parameter(self.weight.new_zeros((self.out_features, r)))})
            )
            self.scaling[adapter_name] = lora_alpha / r

        # 初始化 LoRA 权重
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)

        # 确保新加的模块和原始模块在相同设备上
        self.to(self.weight.device)

    def reset_lora_parameters(self, adapter_name):

        # modified
        if adapter_name in self.lora_A.keys():
            # initialize A and B to zero
            nn.init.zeros_(self.lora_A[adapter_name].weight)
            nn.init.zeros_(self.lora_B[adapter_name].weight)

        if adapter_name in self.lora_embedding_A.keys():
            # initialize a the same way as the default for nn.linear and b to zero
            nn.init.zeros_(self.lora_embedding_A[adapter_name])
            nn.init.normal_(self.lora_embedding_B[adapter_name])

        # modified
        if adapter_name in self.loranew_A.keys():
            nn.init.kaiming_uniform_(self.loranew_A[adapter_name].weight, a=math.sqrt(5))
            nn.init.zeros_(self.loranew_B[adapter_name].weight)


class Linear(nn.Linear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
            self,
            adapter_name: str,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            r_sum: int = 0,  # modified
            use_baseline_lora: bool = False,  # 新增参数
            **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, in_features=in_features, out_features=out_features)
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, r_sum,
                          use_baseline_lora)  # modified
        self.active_adapter = adapter_name

    def _merge_weights(self, adapter_name):
        if self.r[adapter_name] > 0:
            self.weight.data += (
                transpose(
                    self.lora_B[adapter_name].weight @ self.lora_A[adapter_name].weight,
                    self.fan_in_fan_out,
                ) * self.scaling[adapter_name]
            )
            if adapter_name in self.loranew_A:
                self.weight.data += (
                    transpose(
                        self.loranew_B[adapter_name].weight @ self.loranew_A[adapter_name].weight,
                        self.fan_in_fan_out,
                    ) * self.scaling[adapter_name]
                )

    def merge(self):
        adapter_name = self._get_active_adapter_name()
        if adapter_name not in self.lora_A:
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        self._merge_weights(adapter_name)
        self.merged = True

    def _unmerge_weights(self, adapter_name):
        if self.r[adapter_name] > 0:
            self.weight.data -= (
                transpose(
                    self.lora_B[adapter_name].weight @ self.lora_A[adapter_name].weight,
                    self.fan_in_fan_out,
                ) * self.scaling[adapter_name]
            )
            if adapter_name in self.loranew_A:
                self.weight.data -= (
                    transpose(
                        self.loranew_B[adapter_name].weight @ self.loranew_A[adapter_name].weight,
                        self.fan_in_fan_out,
                    ) * self.scaling[adapter_name]
                )

    def unmerge(self):
        adapter_name = self._get_active_adapter_name()
        if adapter_name not in self.lora_A:
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        self._unmerge_weights(adapter_name)
        self.merged = False

    def _get_active_adapter_name(self):
        # 使用 Baseline LoRA 的适配器名称（如果存在）
        return "baseline_lora" if "baseline_lora" in self.lora_A else self.active_adapter

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype

        adapter_name = self._get_active_adapter_name()

        if adapter_name not in self.lora_A:
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        if self.disable_adapters:
            if self.r[adapter_name] > 0 and self.merged:
                self.unmerge()
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r[adapter_name] > 0 and not self.merged:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

            x = x.to(self.lora_A[adapter_name].weight.dtype)
            x = self.lora_dropout[adapter_name](x)

            result += (
                self.lora_B[adapter_name](
                    self.lora_A[adapter_name](x)
                ) * self.scaling[adapter_name]
            )

            # 添加对 loranew_A 和 loranew_B 的处理
            if adapter_name in self.loranew_A:
                result += (
                    self.loranew_B[adapter_name](
                        self.loranew_A[adapter_name](x)
                    ) * self.scaling[adapter_name]
                )
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        result = result.to(previous_dtype)

        return result


class Embedding(nn.Embedding, LoraLayer):
    # LoRA implemented in an Embedding layer
    def __init__(
            self,
            adapter_name: str,
            num_embeddings: int,
            embedding_dim: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            use_baseline_lora: bool = False,  # 新增参数
            **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)

        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoraLayer.__init__(self, in_features=num_embeddings, out_features=embedding_dim)

        self.weight.requires_grad = False

        nn.Embedding.reset_parameters(self)
        self.update_layer_embedding(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_baseline_lora)
        self.active_adapter = adapter_name

    def _get_active_adapter_name(self):
        # 使用 Baseline LoRA 的适配器名称（如果存在）
        return "baseline_lora" if "baseline_lora" in self.lora_embedding_A else self.active_adapter

    def _merge_weights(self, adapter_name):
        if self.r[adapter_name] > 0:
            self.weight.data += (
                transpose(
                    self.lora_embedding_B[adapter_name] @ self.lora_embedding_A[adapter_name], True
                ) * self.scaling[adapter_name]
            )

    def _unmerge_weights(self, adapter_name):
        if self.r[adapter_name] > 0:
            self.weight.data -= (
                transpose(
                    self.lora_embedding_B[adapter_name] @ self.lora_embedding_A[adapter_name], True
                ) * self.scaling[adapter_name]
            )

    def merge(self):
        adapter_name = self._get_active_adapter_name()
        if adapter_name not in self.lora_embedding_A:
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        self._merge_weights(adapter_name)
        self.merged = True

    def unmerge(self):
        adapter_name = self._get_active_adapter_name()
        if adapter_name not in self.lora_embedding_A:
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        self._unmerge_weights(adapter_name)
        self.merged = False

    def forward(self, x: torch.Tensor):
        adapter_name = self._get_active_adapter_name()

        if adapter_name not in self.lora_embedding_A:
            return nn.Embedding.forward(self, x)

        if self.disable_adapters:
            if self.r[adapter_name] > 0 and self.merged:
                self.unmerge()
            return nn.Embedding.forward(self, x)

        result = nn.Embedding.forward(self, x)
        if self.r[adapter_name] > 0 and not self.merged:
            after_A = F.embedding(
                x,
                self.lora_embedding_A[adapter_name].T,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )
            result += (after_A @ self.lora_embedding_B[adapter_name].T) * self.scaling[adapter_name]

        return result
