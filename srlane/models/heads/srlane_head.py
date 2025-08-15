from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import deform_conv2d
import math

from .multi_segment_attention import MultiSegmentAttention
from srlane.ops import nms
from srlane.utils.lane import Lane
from srlane.models.losses.focal_loss import FocalLoss
from srlane.models.utils.dynamic_assign import assign
from srlane.models.losses.lineiou_loss import liou_loss
from srlane.models.registry import HEADS


class EfficientDeformablePooling(nn.Module):
    def __init__(self, in_channels, sample_points):
        super().__init__()
        self.sample_points = sample_points
        
        if in_channels % sample_points != 0:
            self.adjusted_channels = (in_channels // sample_points + 1) * sample_points
            self.channel_adjust = nn.Conv2d(in_channels, self.adjusted_channels, 1)
        else:
            self.adjusted_channels = in_channels
            self.channel_adjust = nn.Identity()
        
        self.offset_conv = nn.Conv2d(self.adjusted_channels, 2 * sample_points, kernel_size=1)
        
        self.weight = nn.Parameter(torch.zeros(self.adjusted_channels, self.adjusted_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(self.adjusted_channels))
        
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.bias)
        
    def forward(self, features: List[Tensor], grid: Tensor) -> Tensor:
        batch_size, num_priors, S, _ = grid.shape
        device = features[0].device
        
        adjusted_features = [self.channel_adjust(feat) for feat in features]
        

        high_res_feature = adjusted_features[0]
        offsets = self.offset_conv(high_res_feature)  # (B, 2*S, H, W)
        
        sampled_features = []
        
        for i, feat in enumerate(adjusted_features):
            if i > 0:
                scale_factor = feat.shape[2] / high_res_feature.shape[2]
                resized_offsets = F.interpolate(offsets, scale_factor=scale_factor, 
                                               mode='bilinear', align_corners=True)
            else:
                resized_offsets = offsets
            
  
            grid_flat = grid.view(batch_size * num_priors * S, 1, 1, 2)
            

            feat_expanded = feat.unsqueeze(1).repeat(1, num_priors, 1, 1, 1)
            feat_flat = feat_expanded.reshape(batch_size * num_priors, self.adjusted_channels, *feat.shape[2:])
            
            offset_expanded = resized_offsets.unsqueeze(1).repeat(1, num_priors, 1, 1, 1)
            offset_flat = offset_expanded.reshape(batch_size * num_priors, 2 * self.sample_points, *feat.shape[2:])

            sampled = deform_conv2d(
                input=feat_flat,
                offset=offset_flat,
                weight=self.weight,
                bias=self.bias,
                padding=0,
            )
            
            # 重塑为统一形状 (B, N, S, C)
            sampled = sampled.view(batch_size, num_priors, self.sample_points, -1)
            sampled_features.append(sampled)
        
        return torch.stack(sampled_features, dim=1)  # (B, L, N, S, C)


class RegularizedPolynomialRegression(nn.Module):

    def __init__(self, fc_hidden_dim, n_offsets, degree=3):
        super().__init__()
        self.degree = degree
        self.n_offsets = n_offsets
        

        self.fc = nn.Sequential(
            nn.Linear(fc_hidden_dim, fc_hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(fc_hidden_dim * 2, degree + 1)
        )
        

        self.boundary_bias = nn.Parameter(torch.zeros(1))
        self.boundary_scale = nn.Parameter(torch.ones(1))
        

        self.register_buffer("basis_vectors", self._create_basis_vectors())
    
    def _create_basis_vectors(self):
        ys = torch.linspace(0, 1, self.n_offsets)
        basis = torch.stack([ys ** i for i in range(self.degree + 1)], dim=1)
        return basis  # (n_offsets, degree+1)
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        coeffs = self.fc(x)  # (B, N, degree+1)
        

        coeffs[:, :, 2:] = coeffs[:, :, 2:] * self.boundary_scale + self.boundary_bias
        

        pred_xs = torch.einsum('bnd,od->bno', coeffs, self.basis_vectors)
        

        pred_xs = torch.sigmoid(pred_xs)
        
        return coeffs, pred_xs


class RefineHead(nn.Module):
    def __init__(self,
                 stage: int,
                 num_points: int,
                 prior_feat_channels: int,
                 fc_hidden_dim: int,
                 refine_layers: int,
                 sample_points: int,
                 num_groups: int,
                 cfg=None):
        super(RefineHead, self).__init__()
        self.stage = stage
        self.cfg = cfg
        self.img_w = self.cfg.img_w
        self.img_h = self.cfg.img_h
        self.n_strips = num_points - 1
        self.n_offsets = num_points
        self.sample_points = sample_points
        self.fc_hidden_dim = fc_hidden_dim
        self.num_groups = num_groups
        self.num_level = cfg.n_fpn
        self.last_stage = stage == refine_layers - 1
        self.poly_degree = cfg.get('poly_degree', 3)  # 多项式次数

        # 采样点索引和Y坐标
        self.register_buffer("sample_x_indexs", tensor=(
                torch.linspace(0, 1,
                               steps=self.sample_points,
                               dtype=torch.float32) * self.n_strips).long())
        self.register_buffer("prior_feat_ys", tensor=torch.flip(
            (1 - self.sample_x_indexs.float() / self.n_strips), dims=[-1]))

        self.prior_feat_channels = prior_feat_channels
        self.z_embeddings = nn.Parameter(torch.zeros(self.sample_points),
                                         requires_grad=True)

        self.deform_pool = EfficientDeformablePooling(prior_feat_channels, sample_points)
        
        self.gather_fc = nn.Conv1d(sample_points, fc_hidden_dim,
                                   kernel_size=self.deform_pool.adjusted_channels,
                                   groups=self.num_groups)
        self.shuffle_fc = nn.Linear(fc_hidden_dim, fc_hidden_dim)
        self.channel_fc = nn.ModuleList()
        self.segment_attn = nn.ModuleList()
        for i in range(1):
            self.segment_attn.append(
                MultiSegmentAttention(fc_hidden_dim, num_groups=num_groups))
            self.channel_fc.append(
                nn.Sequential(nn.Linear(fc_hidden_dim, 2 * fc_hidden_dim),
                              nn.ReLU(),
                              nn.Linear(2 * fc_hidden_dim, fc_hidden_dim)))
        
        reg_modules = list()
        cls_modules = list()
        for _ in range(1):
            reg_modules += [nn.Linear(self.fc_hidden_dim, self.fc_hidden_dim),
                            nn.ReLU()]
            cls_modules += [nn.Linear(self.fc_hidden_dim, self.fc_hidden_dim),
                            nn.ReLU()]

        self.reg_modules = nn.ModuleList(reg_modules)
        self.cls_modules = nn.ModuleList(cls_modules)
        
        if self.last_stage:
            self.reg_pred = RegularizedPolynomialRegression(
                fc_hidden_dim, self.n_offsets, self.poly_degree)
        else:
            self.reg_layers = nn.Linear(
                fc_hidden_dim,
                self.n_offsets + 1 + 1)  # x坐标 + start_y + length
        
        self.cls_layers = nn.Linear(fc_hidden_dim, 2)
        self.init_weights()

    def init_weights(self):
        for m in self.cls_layers.parameters():
            nn.init.normal_(m, mean=0., std=1e-3)
        if not self.last_stage:
            for m in self.reg_layers.parameters():
                nn.init.normal_(m, mean=0., std=1e-3)
        nn.init.normal_(self.z_embeddings, mean=self.cfg.z_mean[self.stage],
                        std=self.cfg.z_std[self.stage])

    def translate_to_linear_weight(self,
                                   ref: Tensor,
                                   num_total: int = 3,
                                   tau: int = 2.0):
        """优化版的线性权重计算"""
        grid = torch.arange(num_total, device=ref.device,
                            dtype=ref.dtype).view(
            *[len(ref.shape) * [1, ] + [-1, ]])
        ref = ref.unsqueeze(-1).clone()
        l2 = (ref - grid).pow(2.0).div(tau).neg()
        weight = torch.softmax(l2, dim=-1)
        return weight

    def pool_prior_features(self,
                            batch_features: List[Tensor],
                            num_priors: int,
                            prior_feat_xs: Tensor):
        """使用可变形采样的特征池化"""
        batch_size = batch_features[0].shape[0]

        norm_ys = self.prior_feat_ys.expand(batch_size, num_priors, -1).unsqueeze(-1)
        norm_xs = (prior_feat_xs / self.n_strips).clamp(0, 1) 

        grid = torch.cat((norm_xs.unsqueeze(-1), norm_ys), dim=-1) * 2 - 1

        feature = self.deform_pool(batch_features, grid)  # (B, L, N, S, C)

        if self.training or not hasattr(self, "z_weight"):
            z_weight = self.translate_to_linear_weight(self.z_embeddings)
            z_weight = z_weight.view(1, 1, self.sample_points, -1).permute(0, 3, 1, 2)
        else:
            z_weight = self.z_weight.view(1, 1, self.sample_points, -1).permute(0, 3, 1, 2)
        

        feature = (feature * z_weight.unsqueeze(-1)).sum(dim=1)  # (B, N, S, C)
        

        feature = feature.reshape(batch_size * num_priors, -1,
                                  self.deform_pool.adjusted_channels)
        feature = self.gather_fc(feature).reshape(batch_size, num_priors, -1)

        for i in range(len(self.segment_attn)):
            res_feature, attn = self.segment_attn[i](feature, attn_mask=None)
            feature = feature + self.channel_fc[i](res_feature)
            
        return feature, attn

    def forward(self, batch_features, priors, pre_feature=None):
        batch_size = batch_features[0].shape[0]
        num_priors = priors.shape[1]

        prior_feat_xs = (priors[..., 4 + self.sample_x_indexs]).flip(dims=[2])
        

        batch_prior_features, attn = self.pool_prior_features(
            batch_features, num_priors, prior_feat_xs)

        fc_features = batch_prior_features
        fc_features = fc_features.reshape(batch_size * num_priors, self.fc_hidden_dim)


        if pre_feature is not None:
            fc_features = fc_features + pre_feature.view(*fc_features.shape)


        cls_features = fc_features
        for cls_layer in self.cls_modules:
            cls_features = cls_layer(cls_features)
        cls_logits = self.cls_layers(cls_features)
        cls_logits = cls_logits.reshape(batch_size, num_priors, -1)  # (B, num_priors, 2)
        

        predictions = priors.clone()
        predictions[:, :, :2] = cls_logits
        
        reg_features = fc_features
        for reg_layer in self.reg_modules:
            reg_features = reg_layer(reg_features)
            
        if self.last_stage:

            coeffs, xs_pred = self.reg_pred(reg_features.view(batch_size, num_priors, -1))
            

            if xs_pred.size(2) != self.n_offsets:

                xs_pred = F.interpolate(
                    xs_pred.permute(0, 2, 1), 
                    size=self.n_offsets, 
                    mode='linear', 
                    align_corners=True
                ).permute(0, 2, 1)
            

            coeffs_size = self.poly_degree + 1
            xs_start_idx = 3 + coeffs_size  
            total_size_required = xs_start_idx + self.n_offsets
 
            if predictions.size(2) < total_size_required:

                extended_predictions = torch.zeros(
                    predictions.size(0), 
                    predictions.size(1), 
                    total_size_required,
                    device=predictions.device,
                    dtype=predictions.dtype
                )
                extended_predictions[:, :, :predictions.size(2)] = predictions
                predictions = extended_predictions
            

            predictions[:, :, :2] = cls_logits  
            predictions[:, :, 2:3] = coeffs[:, :, 0:1]  
            predictions[:, :, 3:3+coeffs_size] = coeffs  
            

            predictions[:, :, xs_start_idx:xs_start_idx+self.n_offsets] = xs_pred
        else:

            reg = self.reg_layers(reg_features).reshape(batch_size, num_priors, -1)

            predictions[:, :, 2:] += reg

        return predictions, fc_features, attn


@HEADS.register_module
class CascadeRefineHead(nn.Module):
    def __init__(self,
                 num_points: int = 72,
                 prior_feat_channels: int = 64,
                 fc_hidden_dim: int = 64,
                 refine_layers: int = 1,
                 sample_points: int = 36,
                 num_groups: int = 6,
                 cfg=None):
        super(CascadeRefineHead, self).__init__()
        self.cfg = cfg
        self.img_w = self.cfg.img_w
        self.img_h = self.cfg.img_h
        self.n_strips = num_points - 1
        self.n_offsets = num_points
        self.sample_points = sample_points
        self.refine_layers = refine_layers
        self.fc_hidden_dim = fc_hidden_dim
        self.num_groups = num_groups
        self.prior_feat_channels = prior_feat_channels
        self.poly_degree = cfg.get('poly_degree', 3)  
        self.register_buffer(name="prior_ys",
                             tensor=torch.linspace(1, 0, steps=self.n_offsets,
                                                   dtype=torch.float32))

        self.stage_heads = nn.ModuleList()
        for i in range(refine_layers):
            self.stage_heads.append(
                RefineHead(stage=i,
                           num_points=num_points,
                           prior_feat_channels=prior_feat_channels,
                           fc_hidden_dim=fc_hidden_dim,
                           refine_layers=refine_layers,
                           sample_points=sample_points,
                           num_groups=num_groups,
                           cfg=cfg))

        self.cls_criterion = FocalLoss(alpha=0.25, gamma=2.)
        

        ys = torch.linspace(0, 1, self.n_offsets)
        basis = torch.stack([ys ** i for i in range(self.poly_degree + 1)], dim=1)
        self.register_buffer("poly_basis", basis)
        self.register_buffer("poly_pinv", torch.pinverse(basis))

    def forward(self, x, **kwargs):
        batch_features = list(x)
        batch_features.reverse()
        priors = kwargs["priors"]
        pre_feature = None
        predictions_lists = []
        attn_lists = []

        # 迭代细化
        for stage in range(self.refine_layers):
            predictions, pre_feature, attn = self.stage_heads[stage](
                batch_features, priors,
                pre_feature)
            predictions_lists.append(predictions)
            attn_lists.append(attn)

            if stage != self.refine_layers - 1:
                priors = predictions.clone().detach()

        if self.training:
            output = {"predictions_lists": predictions_lists,
                      "attn_lists": attn_lists}
            return output
        return predictions_lists[-1]

    def loss(self,
             output,
             batch):
        predictions_lists = output["predictions_lists"]
        attn_lists = output["attn_lists"]
        targets = batch["gt_lane"].clone()


        w_cls = 1.0
        w_l1 = 0.5 if self.refine_layers > 1 else 1.0  
        w_iou = 1.0
        w_attn = 0.1
        w_poly = 0.5 
        w_curv = 0.2 

        cls_loss = torch.tensor(0., device=targets.device)
        l1_loss = torch.tensor(0., device=targets.device)
        iou_loss = torch.tensor(0., device=targets.device)
        attn_loss = torch.tensor(0., device=targets.device)
        poly_loss = torch.tensor(0., device=targets.device)
        curv_loss = torch.tensor(0., device=targets.device)
        
        batch_size = len(targets)
        num_valid_samples = 0

        for stage in range(0, self.refine_layers):
            predictions_list = predictions_lists[stage]
            attn_list = attn_lists[stage]
            
            for idx in range(batch_size):
                predictions = predictions_list[idx]
                target = targets[idx]
                attn = attn_list[idx] if attn_list is not None else None
                

                valid_target = target[target[:, 1] == 1]
                if len(valid_target) == 0:

                    cls_target = predictions.new_zeros(predictions.shape[0]).long()
                    cls_pred = predictions[:, :2]
                    cls_loss += self.cls_criterion(cls_pred, cls_target).sum()
                    num_valid_samples += 1
                    continue
                    

                is_final_stage = (stage == self.refine_layers - 1)
                
                if is_final_stage:
   
                    poly_coeffs = predictions[:, 3:3+self.poly_degree+1]
                    

                    ys = self.prior_ys.expand_as(predictions[:, 4:4+self.n_offsets])
                    pred_xs = torch.zeros_like(ys)
                    for i in range(self.poly_degree+1):
                        pred_xs += poly_coeffs[:, i:i+1] * (ys ** i)
                    

                    predictions = torch.cat((
                        predictions[:, :4], 
                        pred_xs * self.img_w
                    ), dim=-1)

                    target_poly_coeffs = []
                    for t in valid_target:
                        xs = t[4:].float() / self.img_w

                        coeffs = torch.einsum('o,od->d', xs, self.poly_pinv)
                        target_poly_coeffs.append(coeffs)
                    target_poly_coeffs = torch.stack(target_poly_coeffs, dim=0)
                    
                else:

                    predictions = torch.cat((
                        predictions[:, :2],
                        predictions[:, 2:4] * self.n_strips,
                        predictions[:, 4:] * self.img_w
                    ), dim=-1)


                with torch.no_grad():
                    matched_row_inds, matched_col_inds = assign(
                        predictions, valid_target, self.img_w,
                        k=self.cfg.angle_map_size[0])
                    
                    num_matches = len(matched_row_inds)
                    if num_matches == 0:
                        continue
                

                cls_target = predictions.new_zeros(predictions.shape[0]).long()
                cls_target[matched_row_inds] = 1
                cls_pred = predictions[:, :2]
                cls_loss += self.cls_criterion(cls_pred, cls_target).sum() / max(1, num_matches)
                

                if stage > 0 and attn is not None:
                    attn_loss += MultiSegmentAttention.loss(
                        predictions[matched_row_inds, 4:] / self.img_w,
                        valid_target[matched_col_inds, 4:] / self.img_w,
                        attn[matched_row_inds])
                
                if is_final_stage:

                    pred_coeffs = predictions[matched_row_inds, 3:3+self.poly_degree+1]
                    poly_loss += w_poly * F.smooth_l1_loss(
                        pred_coeffs, 
                        target_poly_coeffs[matched_col_inds],
                        reduction='mean'
                    )
                    

                    d2_pred = pred_coeffs[:, 2] * 2 + pred_coeffs[:, 3] * 6 * self.prior_ys.mean()
                    d2_target = target_poly_coeffs[matched_col_inds, 2] * 2 + \
                               target_poly_coeffs[matched_col_inds, 3] * 6 * self.prior_ys.mean()
                    curv_loss += w_curv * F.smooth_l1_loss(d2_pred, d2_target)
                    
                else:

                    reg_yl = predictions[matched_row_inds, 2:4]
                    target_yl = valid_target[matched_col_inds, 2:4].clone()
                    with torch.no_grad():
                        reg_start_y = torch.clamp(
                            (reg_yl[:, 0]).round().long(), 0,
                            self.n_strips)
                        target_start_y = target_yl[:, 0].round().long()
         
                        len_diff = reg_start_y - target_start_y
                        target_yl[:, 1] = torch.clamp(target_yl[:, 1] - len_diff, 0, self.n_strips)
                    
                    l1_loss += w_l1 * F.smooth_l1_loss(reg_yl, target_yl, reduction="mean")
                    

                    reg_pred = predictions[matched_row_inds, 4:]
                    reg_targets = valid_target[matched_col_inds, 4:].clone()
                    iou_loss += w_iou * liou_loss(reg_pred, reg_targets, self.img_w)
                    
                num_valid_samples += 1


        if num_valid_samples == 0:
            num_valid_samples = 1  
        
        cls_loss /= (num_valid_samples * self.refine_layers)
        l1_loss /= (num_valid_samples * self.refine_layers)
        iou_loss /= (num_valid_samples * self.refine_layers)
        attn_loss /= (num_valid_samples * (self.refine_layers - 1) if self.refine_layers > 1 else 1)
        poly_loss /= num_valid_samples
        curv_loss /= num_valid_samples


        total_loss = (w_cls * cls_loss + w_l1 * l1_loss + w_iou * iou_loss + 
                      w_attn * attn_loss + w_poly * poly_loss + w_curv * curv_loss)
        
        return_value = {
            "loss": total_loss,
            "cls_loss": cls_loss,
            "l1_loss": l1_loss,
            "iou_loss": iou_loss,
            "attn_loss": attn_loss,
            "poly_loss": poly_loss,
            "curv_loss": curv_loss
        }

        return return_value

    def predictions_to_pred(self, predictions, img_meta):
        """处理最后阶段的多项式参数输出"""
        prior_ys = self.prior_ys.to(predictions.device)
        prior_ys = prior_ys.double()
        lanes = []
        

        cls_scores = F.softmax(predictions[:, :2], dim=1)[:, 1]
        

        for i, pred in enumerate(predictions):
            if cls_scores[i] < 0.5: 
                continue

            poly_coeffs = pred[3:3+self.poly_degree+1]
            

            lane_xs = torch.zeros(self.n_offsets, device=predictions.device)
            for j in range(self.poly_degree+1):
                lane_xs += poly_coeffs[j] * (prior_ys ** j)
            

            lane_ys = prior_ys * (self.img_h - 0) + 0  
            if "img_cut_height" in img_meta:
                cut_height = img_meta["img_cut_height"]
                ori_img_h = img_meta["img_size"][0]
                lane_ys = (lane_ys * (ori_img_h - cut_height) + cut_height) / ori_img_h
            

            lane_xs = lane_xs * self.img_w
            

            valid_mask = (lane_xs >= 0) & (lane_xs <= self.img_w)
            if valid_mask.sum() <= 1:  
                continue
                
            points = torch.stack((lane_xs[valid_mask], lane_ys[valid_mask]), dim=1)
            lane = Lane(
                points=points.cpu().numpy(), 
                metadata={
                    'confidence': cls_scores[i].item(),
                    'coeffs': poly_coeffs.cpu().numpy()
                }
            )
            lanes.append(lane)
        
        return lanes

    def get_lanes(self, output, img_metas, as_lanes=True):
     
        softmax = nn.Softmax(dim=1)

        decoded = []
        img_metas = [item for img_meta in img_metas.data for item in img_meta]

        predictions = output if torch.is_tensor(output) else output[-1]
        
        for i, (pred, img_meta) in enumerate(zip(predictions, img_metas)):

            scores = softmax(pred[:, :2])[:, 1]
            keep_inds = scores >= self.cfg.test_parameters.conf_threshold
            pred = pred[keep_inds]
            scores = scores[keep_inds]

            if pred.shape[0] == 0:
                decoded.append([])
                continue


            if self.stage_heads[-1].last_stage:

                poly_coeffs = pred[:, 3:3+self.poly_degree+1]
                

                ys = torch.linspace(0, 1, self.n_offsets, device=pred.device)
                basis = torch.stack([ys ** i for i in range(self.poly_degree + 1)], dim=1)
                pred_xs = torch.einsum('nd,od->no', poly_coeffs, basis)

                nms_preds = pred.clone()
                nms_preds[:, 4:4+self.n_offsets] = pred_xs * self.img_w
            else:
                nms_preds = pred.clone()

            nms_preds[:, 2:4] *= self.n_strips
            nms_preds[:, 3] = nms_preds[:, 2] + nms_preds[:, 3] - 1

            keep, num_to_keep, _ = nms(
                nms_preds,
                scores,
                overlap=self.cfg.test_parameters.nms_thres,
                top_k=self.cfg.max_lanes)
            keep = keep[:num_to_keep]
            pred = pred[keep]

            if pred.shape[0] == 0:
                decoded.append([])
                continue

            if as_lanes:
                decoded_pred = self.predictions_to_pred(pred, img_meta)
            else:
                decoded_pred = pred
            decoded.append(decoded_pred)

        return decoded

    def __repr__(self):
        num_params = sum(map(lambda x: x.numel(), self.parameters()))
        return f"#Params of {self._get_name()}: {num_params / 10 ** 3:<.2f}[K]"
