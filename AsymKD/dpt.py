import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download

from AsymKD.blocks import FeatureFusionBlock, _make_scratch
from depth_anything.dpt import DepthAnything
from AsymKD.util.transform import Resize, NormalizeImage, PrepareForNet
from segment_anything import sam_model_registry, SamPredictor
from torchvision.transforms import Compose
import cv2
from moe import MoE
import numpy as np

def _make_fusion_block(features, use_bn, size = None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class AsymKD_DPTHead(nn.Module):
    def __init__(self, nclass, in_channels, features=256, use_bn=False, out_channels=[256, 512, 1024, 1024], use_clstoken=False):
        super(AsymKD_DPTHead, self).__init__()
        
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.nclass = nclass
        self.use_clstoken = use_clstoken
        
        '''input output channel 변경 필요'''
        #in_channels = 1024
        selected_feature_channels = 256 #
        selected_out_channels=[64, 128, 256, 256]
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=selected_feature_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in selected_out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=selected_out_channels[0],
                out_channels=selected_out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=selected_out_channels[1],
                out_channels=selected_out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=selected_out_channels[3],
                out_channels=selected_out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])

        


        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
                
        '''Solution 3 구현'''
        self.adapt_H = 20
        self.adapt_W = 30

        self.adaptive_max_pool = nn.AdaptiveMaxPool2d((self.adapt_H, self.adapt_W))
        self.MoE_Loss = None

        element1_moe_num = 256
        element1_input_channel = in_channels*2 #2048
        expert_num = element1_input_channel//element1_moe_num #8
        self.k = 4
        self.expert_num = expert_num
        
        MOE_layer1_element1= MoE(input_size=expert_num*self.adapt_H*self.adapt_W, num_experts=expert_num, k=self.k, noisy_gating=True,moe_num=element1_moe_num).to(self.DEVICE)
        MOE_layer2_element1= MoE(input_size=expert_num*self.adapt_H*self.adapt_W, num_experts=expert_num, k=self.k, noisy_gating=True,moe_num=element1_moe_num).to(self.DEVICE)
        MOE_layer3_element1= MoE(input_size=expert_num*self.adapt_H*self.adapt_W, num_experts=expert_num, k=self.k, noisy_gating=True,moe_num=element1_moe_num).to(self.DEVICE)
        MOE_layer4_element1= MoE(input_size=expert_num*self.adapt_H*self.adapt_W, num_experts=expert_num, k=self.k, noisy_gating=True,moe_num=element1_moe_num).to(self.DEVICE)

        element2_moe_num = element1_moe_num//2

        MOE_layer1_element2= MoE(input_size=expert_num*self.adapt_H*self.adapt_W, num_experts=expert_num, k=self.k, noisy_gating=True,moe_num=element2_moe_num).to(self.DEVICE)
        MOE_layer2_element2= MoE(input_size=expert_num*self.adapt_H*self.adapt_W, num_experts=expert_num, k=self.k, noisy_gating=True,moe_num=element2_moe_num).to(self.DEVICE)
        MOE_layer3_element2= MoE(input_size=expert_num*self.adapt_H*self.adapt_W, num_experts=expert_num, k=self.k, noisy_gating=True,moe_num=element2_moe_num).to(self.DEVICE)
        MOE_layer4_element2= MoE(input_size=expert_num*self.adapt_H*self.adapt_W, num_experts=expert_num, k=self.k, noisy_gating=True,moe_num=element2_moe_num).to(self.DEVICE)
        
        element3_moe_num = element2_moe_num//2

        MOE_layer1_element3= MoE(input_size=expert_num*self.adapt_H*self.adapt_W, num_experts=expert_num, k=self.k, noisy_gating=True,moe_num=element3_moe_num).to(self.DEVICE)
        MOE_layer2_element3= MoE(input_size=expert_num*self.adapt_H*self.adapt_W, num_experts=expert_num, k=self.k, noisy_gating=True,moe_num=element3_moe_num).to(self.DEVICE)
        MOE_layer3_element3= MoE(input_size=expert_num*self.adapt_H*self.adapt_W, num_experts=expert_num, k=self.k, noisy_gating=True,moe_num=element3_moe_num).to(self.DEVICE)
        MOE_layer4_element3= MoE(input_size=expert_num*self.adapt_H*self.adapt_W, num_experts=expert_num, k=self.k, noisy_gating=True,moe_num=element3_moe_num).to(self.DEVICE)
        
        '''
        element4_moe_num = element3_moe_num//2

        MOE_layer1_element4= MoE(input_size=expert_num*self.adapt_H*self.adapt_W, num_experts=expert_num, k=self.k, noisy_gating=True,moe_num=element4_moe_num).to(self.DEVICE)
        MOE_layer2_element4= MoE(input_size=expert_num*self.adapt_H*self.adapt_W, num_experts=expert_num, k=self.k, noisy_gating=True,moe_num=element4_moe_num).to(self.DEVICE)
        MOE_layer3_element4= MoE(input_size=expert_num*self.adapt_H*self.adapt_W, num_experts=expert_num, k=self.k, noisy_gating=True,moe_num=element4_moe_num).to(self.DEVICE)
        MOE_layer4_element4= MoE(input_size=expert_num*self.adapt_H*self.adapt_W, num_experts=expert_num, k=self.k, noisy_gating=True,moe_num=element4_moe_num).to(self.DEVICE)
        '''
        MOE_layer1 = nn.ModuleList([MOE_layer1_element1,MOE_layer1_element2,MOE_layer1_element3])
        MOE_layer2 = nn.ModuleList([MOE_layer2_element1,MOE_layer2_element2,MOE_layer2_element3])
        MOE_layer3 = nn.ModuleList([MOE_layer3_element1,MOE_layer3_element2,MOE_layer3_element3])
        MOE_layer4 = nn.ModuleList([MOE_layer4_element1,MOE_layer4_element2,MOE_layer4_element3])
        
        self.MOE_layers = nn.ModuleList([MOE_layer1,MOE_layer2,MOE_layer3,MOE_layer4])


        features = 64
        #double_out_channels=[512, 1024, 2048, 2048]
        self.scratch = _make_scratch(
            selected_out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        head_features_1 = features
        head_features_2 = 32
        
        if nclass > 1:
            self.scratch.output_conv = nn.Sequential(
                nn.Conv2d(head_features_1, head_features_1, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_1, nclass, kernel_size=1, stride=1, padding=0),
            )
        else:
            self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
            
            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
                nn.ReLU(True),
                nn.Identity(),
            )

            
    def forward(self, depth_intermediate_features, depth_patch_h, depth_patch_w, seg_intermediate_features, seg_patch_h, seg_patch_w ):
        depth_out = []
        for i, x in enumerate(depth_intermediate_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], depth_patch_h, depth_patch_w))
            

            # x = self.depth_projects[i](x)
            # x = self.depth_resize_layers[i](x)
            
            depth_out.append(x)

        # seg_out = []
        # for i, x in enumerate(seg_intermediate_features):
        #     print(f'seg size : {x.shape}')
        #     x = self.seg_projects[i](x)
        #     x = self.seg_resize_layers[i](x)
            
        #     seg_out.append(x)
            
        seg_resize_out = []
        for idx, seg_feature in enumerate(seg_intermediate_features):
            seg_feature = F.interpolate(seg_feature, size=(depth_out[idx].shape[2]*2, depth_out[idx].shape[3]*2), mode='bilinear', align_corners=False)
            seg_feature = F.max_pool2d(seg_feature, kernel_size=2)
            seg_resize_out.append(seg_feature)


        feature_fusion_out = [] 
        for depth_feature, seg_feature in zip(depth_out, seg_resize_out):
            concatenated_tensor = torch.cat((depth_feature, seg_feature), dim=1)
            feature_fusion_out.append(concatenated_tensor)
            # print(f'feature_fusion_out : {concatenated_tensor.shape}')

        '''Adaptive Pooling 코드'''
        adaptive_pooing_features = [] 
        for feature_fusion in feature_fusion_out:
            output_tensor_max = self.adaptive_max_pool(feature_fusion)
            adaptive_pooing_features.append(output_tensor_max)
            # print(f'adaptive_pooing_feature : {output_tensor_max.shape}')



        '''MOE Select layer 코드'''
        self.MoE_Loss = None
        MoE_weighted_top_k_absolute_selected_feature = []
        temp_MoE_selected_feature = []
        B, C, H, W = adaptive_pooing_features[0].shape
        for MOE_layer, adaptive_pooing_feature, absolute_feature in zip(self.MOE_layers, adaptive_pooing_features, feature_fusion_out):
            temp_adaptive_pooing_feature = adaptive_pooing_feature
            temp_absolute_feature = absolute_feature
            for MOE in MOE_layer:
                loss, weighted_top_k_absolute_channel,weighted_top_k_adaptive_channel = MOE(temp_adaptive_pooing_feature, temp_absolute_feature)
                if self.MoE_Loss is not None:
                    self.MoE_Loss = self.MoE_Loss + loss
                else:
                    self.MoE_Loss = loss
                temp_adaptive_pooing_feature = weighted_top_k_adaptive_channel
                temp_absolute_feature = weighted_top_k_absolute_channel 

                
            temp_MoE_selected_feature.append(temp_adaptive_pooing_feature)
            MoE_weighted_top_k_absolute_selected_feature.append(temp_absolute_feature)

        # MoE_selected_features = []
        # select_channel_num = C//(2**4) #128
        # _,_,H,W = feature_fusion_out[0].shape
        # for layer,fusion_feature in zip(MoE_selected_feature_index,feature_fusion_out):
        #     MoE_selected_feature = torch.zeros((B, select_channel_num, H, W)).to(self.DEVICE)
        #     for b in range(B):
        #         for c in range(select_channel_num):
        #             MoE_selected_feature[b, c] = fusion_feature[b, layer[b,c].item()]

        #     MoE_selected_features.append(MoE_selected_feature)
        #     # print(f'MoE_selected_feature{MoE_selected_feature.shape}')
        
        return MoE_weighted_top_k_absolute_selected_feature, self.MoE_Loss

        '''decoder에 넣기전 Select layer conv 연산'''
        conv_selected_layer = []
        for i, x in enumerate(MoE_weighted_top_k_absolute_selected_feature):
            # print(f'seg size : {x.shape}')
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            conv_selected_layer.append(x)
        
        


        layer_1, layer_2, layer_3, layer_4 = conv_selected_layer
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        


        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        depth_out = self.scratch.output_conv1(path_1)
        depth_out = F.interpolate(depth_out, (int(depth_patch_h * 14), int(depth_patch_w * 14)), mode="bilinear", align_corners=True)
        depth_out = self.scratch.output_conv2(depth_out)
        
        return depth_out, self.MoE_Loss
        
        
class AsymKD_DepthAnything(nn.Module):
    def __init__(self, ImageEncoderViT, features=256, out_channels=[256, 512, 1024, 1024], use_bn=False, use_clstoken=False, localhub=True):
        super(AsymKD_DepthAnything, self).__init__()
        
        
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        encoder = 'vitl' # can also be 'vitb' or 'vitl'
        self.depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(encoder)).to(self.DEVICE).eval()
        

        for param in self.depth_anything.parameters():
            param.requires_grad = False

        

        self.ImageEncoderViT = ImageEncoderViT

        for i, (name, param) in enumerate(self.ImageEncoderViT.named_parameters()):
            param.requires_grad = False

        #dim = 768 #= self.pretrained.blocks[0].attn.qkv.in_features
        dim = 1024

        self.depth_head = AsymKD_DPTHead(1, dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)
        
    def forward(self, depth_image,seg_image):


        depth_image_h, depth_image_w = depth_image.shape[-2:]
        seg_image_h, seg_image_w = seg_image.shape[-2:]
        self.ImageEncoderViT.eval()
        self.depth_anything.eval()

        
        depth_intermediate_features = self.depth_anything(depth_image)



        seg_intermediate_features = self.ImageEncoderViT(seg_image)



        depth_patch_h, depth_patch_w = depth_image_h // 14, depth_image_w // 14
        seg_patch_h, seg_patch_w = seg_image_h // 16, seg_image_w // 16


        depth, MoE_Loss = self.depth_head(depth_intermediate_features, depth_patch_h, depth_patch_w, seg_intermediate_features, seg_patch_h, seg_patch_w )
        # depth = F.interpolate(depth, size=(depth_patch_h*14, depth_patch_w*14), mode="bilinear", align_corners=True)
        # depth = F.relu(depth)

        return depth, MoE_Loss




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder",
        default="vits",
        type=str,
        choices=["vits", "vitb", "vitl"],
    )
    args = parser.parse_args()
    
    model = DepthAnything.from_pretrained("LiheYoung/depth_anything_{:}14".format(args.encoder))
    
    print(model)
    