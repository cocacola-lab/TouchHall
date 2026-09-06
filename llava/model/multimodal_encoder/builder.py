import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from transformers.models.clip.modeling_clip import CLIP_VISION_INPUTS_DOCSTRING,CLIPVisionTransformer
from .open_clip.transformer import PatchDropout
from transformers import CLIPModel as HFCLIPModel, CLIPVisionConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling
from typing import Optional, Tuple, Union
import torch
from einops import rearrange
from torch import nn
from transformers.utils import replace_return_docstrings, add_start_docstrings_to_model_forward

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs) #基于CLIP构建视觉编码器

    raise ValueError(f'Unknown vision tower: {vision_tower}')




class Touch_Encoder(nn.Module):
    def __init__(self, config, delay_load=False):
        super(Touch_Encoder, self).__init__()
        self.vision_config = config.vision_config
        self.hidden_size = config.vision_config.hidden_size
        # self.touch_projection = nn.Linear(config.vision_config.hidden_size, self.projection_dim, bias=False)
        self.T = 1

        if not delay_load:
            self.load_model(config)
      


    # def load_model(self, device_map=None):
    #     if self.is_loaded:
    #         print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
    #         return

    #     self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
    #     self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
    #     self.vision_tower.requires_grad_(False)

    #     self.is_loaded = True
                
    def load_model(self):
        super(Touch_Encoder, self).__init__()
        self.touch_model = CLIPVisionTransformer(self.vision_config)
        self.touch_model.patch_dropout = PatchDropout(0.2)
        self.touch_model.forward = self.touch_model_forward

    
    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPVisionConfig)       
    def touch_model_forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.touch_model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.touch_model.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.touch_model.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        if len(pixel_values.shape) == 7:
            b_new, pair_new, T, bs_new, channel_new, h_new, w_new = pixel_values.shape
            # print(pixel_values.shape)
            B = b_new * pair_new * bs_new
            pixel_values = pixel_values.reshape(B*T, channel_new, h_new, w_new)

        elif len(pixel_values.shape) == 5:
            B, _, T, _, _ = pixel_values.shape
            # print(pixel_values.shape)
            pixel_values = rearrange(pixel_values, 'b c t h w -> (b t) c h w')
        else:
            # print(pixel_values.shape)
            B, _, _, _ = pixel_values.shape
            T = 1
        # print('111==>', pixel_values.shape)
        hidden_states = self.touch_model.embeddings(pixel_values)
        #print('hidden_states', hidden_states.shape)
        #
        # if self.temporal_embedding is not None and get_global_value()['NUM_FRAMES'] != 1:
        #     n = hidden_states.shape[1]
        #     hidden_states = rearrange(hidden_states, '(b t) n d -> (b n) t d', t=T)
        #     hidden_states = hidden_states + self.temporal_embedding[:, :T, :]
        #     hidden_states = rearrange(hidden_states, '(b n) t d -> (b t) n d', n=n)
        T = self.T
        # print('B.shape, T.shape', B.shape, T.shape)
        # hidden_states = self.touch_model.patch_dropout(hidden_states, B, T)
        # print('patch_dropout', hidden_states.shape)
        hidden_states = self.touch_model.pre_layrnorm(hidden_states)

        encoder_outputs = self.touch_model.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.touch_model.post_layernorm(pooled_output)

        pooled_output = pooled_output.reshape(B, T, -1).mean(1)
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )    
        
        

def build_touch_tower(touch_tower_cfg, delay_load=False):
    return Touch_Encoder(touch_tower_cfg, delay_load)


class SingleModal_Touch_Encoder(nn.Module):
    def __init__(self, config, delay_load=False):
        super(SingleModal_Touch_Encoder, self).__init__()
        self.vision_config = config.vision_config
        self.weight_path = config.weight_path
        self.hidden_size = config.vision_config.hidden_size
        # self.touch_projection = nn.Linear(config.vision_config.hidden_size, self.projection_dim, bias=False)
        self.T = 1
        # if not delay_load:
        #     self.load_model()
        self.is_loaded = False
      


    # def load_model(self, device_map=None):
    #     if self.is_loaded:
    #         print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
    #         return

    #     self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
    #     self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
    #     self.vision_tower.requires_grad_(False)

    #     self.is_loaded = True
                
    def load_model(self):
        super(SingleModal_Touch_Encoder, self).__init__()
        self.touch_model = CLIPVisionTransformer(self.vision_config)
        self.touch_model.patch_dropout = PatchDropout(0.2)
        self.touch_model.forward = self.touch_model_forward
        

        ckpt_data = torch.load(self.weight_path, map_location=torch.device('cuda'))
        vision_state_dict = {
            k.replace("touch_enc.", ""): v
            for k, v in ckpt_data.items()
            if k.startswith("touch_enc.")
        }

        self.touch_model.load_state_dict(vision_state_dict, strict=True)
        self.is_loaded = True

    
    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPVisionConfig)       
    def touch_model_forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.touch_model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.touch_model.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.touch_model.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        if len(pixel_values.shape) == 7:
            b_new, pair_new, T, bs_new, channel_new, h_new, w_new = pixel_values.shape
            # print(pixel_values.shape)
            B = b_new * pair_new * bs_new
            pixel_values = pixel_values.reshape(B*T, channel_new, h_new, w_new)

        elif len(pixel_values.shape) == 5:
            B, _, T, _, _ = pixel_values.shape
            # print(pixel_values.shape)
            pixel_values = rearrange(pixel_values, 'b c t h w -> (b t) c h w')
        else:
            # print(pixel_values.shape)
            B, _, _, _ = pixel_values.shape
            T = 1
        # print('111==>', pixel_values.shape)
        hidden_states = self.touch_model.embeddings(pixel_values)
        #print('hidden_states', hidden_states.shape)
        #
        # if self.temporal_embedding is not None and get_global_value()['NUM_FRAMES'] != 1:
        #     n = hidden_states.shape[1]
        #     hidden_states = rearrange(hidden_states, '(b t) n d -> (b n) t d', t=T)
        #     hidden_states = hidden_states + self.temporal_embedding[:, :T, :]
        #     hidden_states = rearrange(hidden_states, '(b n) t d -> (b t) n d', n=n)
        T = self.T
        # print('B.shape, T.shape', B.shape, T.shape)
        # hidden_states = self.touch_model.patch_dropout(hidden_states, B, T)
        # print('patch_dropout', hidden_states.shape)
        #TODO 这里hidden state 经过pre layernorm 之后全部都是0
        hidden_states = self.touch_model.pre_layrnorm(hidden_states)
        encoder_outputs = self.touch_model.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.touch_model.post_layernorm(pooled_output)

        pooled_output = pooled_output.reshape(B, T, -1).mean(1)
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )    
        
        

def build_single_modal_tower(touch_tower_cfg, delay_load=False):
    return SingleModal_Touch_Encoder(touch_tower_cfg, delay_load)