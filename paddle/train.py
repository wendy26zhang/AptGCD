import argparse
import math
import os
import random
import numpy as np
import pandas as pd
import paddle
import paddle.nn as nn
import paddle.vision.transforms as T
from paddle.io import DataLoader, Dataset
from PIL import Image
import cv2
import matplotlib.pyplot as plt

import respmvit as vit_prompt
from prompt_model_glu import *

from sklearn.cluster import KMeans

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from util.general_utils import AverageMeter, init_experiment
from util.cluster_and_log_utils import log_accs_from_preds
from config import exp_root
from model import DINOHead, info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator, get_params_groups, initial_qhat, update_qhat, causal_inference, WeightedEntropyLoss 
import warnings
import paddle.nn.functional as F

from paddle.optimizer import SGD
from paddle.optimizer.lr import CosineAnnealingDecay
from visualdl import LogWriter
warnings.filterwarnings("ignore", category=UserWarning)


############################## 相关utils函数，如下 ##############################

class PaddleFlag:
    cudnn_enabled = True
    cudnn_benchmark = False
    matmul_allow_tf32 = False
    cudnn_allow_tf32 = True
    cudnn_deterministic = False

# def reshape(self, *args, **kwargs):
#     if args:
#         if len(args) == 1 and isinstance(args[0], (tuple, list)):
#             return paddle.reshape(self, args[0])
#         else:
#             return paddle.reshape(self, list(args))
#     elif kwargs:
#         assert "shape" in kwargs
#         return paddle.reshape(self, shape=kwargs["shape"])

# setattr(paddle.Tensor, "reshape", reshape)

def min_class_func(self, *args, **kwargs):
    if "other" in kwargs:
        kwargs["y"] = kwargs.pop("other")
        ret = paddle.minimum(self, *args, **kwargs)
    elif len(args) == 1 and isinstance(args[0], paddle.Tensor):
        ret = paddle.minimum(self, *args, **kwargs)
    else:
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")

        if "axis" in kwargs or len(args) >= 1:
            ret = paddle.min(self, *args, **kwargs), paddle.argmin(self, *args, **kwargs)
        else:
            ret = paddle.min(self, *args, **kwargs)

    return ret

setattr(paddle.Tensor, "min_func", min_class_func)

def max_class_func(self, *args, **kwargs):
    if "other" in kwargs:
        kwargs["y"] = kwargs.pop("other")
        ret = paddle.maximum(self, *args, **kwargs)
    elif len(args) == 1 and isinstance(args[0], paddle.Tensor):
        ret = paddle.maximum(self, *args, **kwargs)
    else:
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")

        if "axis" in kwargs or len(args) >= 1:
            ret = paddle.max(self, *args, **kwargs), paddle.argmax(self, *args, **kwargs)
        else:
            ret = paddle.max(self, *args, **kwargs)

    return ret

setattr(paddle.Tensor, "max_func", max_class_func)
############################## 相关utils函数，如上 ##############################

# 固定随机种子
seed = 1
np.random.seed(seed)
random.seed(seed)
paddle.seed(seed)



import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class PatchPrompter(nn.Layer):
    def __init__(self, args):
        super(PatchPrompter, self).__init__()
        self.patch_size = args.patch_size
        self.prompt_size = args.prompt_size
        self.fg_size = self.patch_size - args.prompt_size * 2

        # Paddle创建可训练参数，初始化同torch.randn
        self.patch = self.create_parameter(
            shape=[1, 3, args.image_size, args.image_size],
            default_initializer=paddle.nn.initializer.Normal()
        )
    
    def forward(self, x):
        _, _, h, w = x.shape

        # 创建全0 tensor，注意Paddle中默认创建在CPU，需移到输入x同设备
        fg_in_patch = paddle.zeros([1, 3, self.fg_size, self.fg_size], dtype=x.dtype).to(device)
        fg_in_patch = F.pad(fg_in_patch, 
                           pad=[self.prompt_size, self.prompt_size, self.prompt_size, self.prompt_size], 
                           mode='constant', value=1.0)
        
        # Tile重复mask，h和w必须是patch_size的倍数
        mask = fg_in_patch.tile([1, 1, h // self.patch_size, w // self.patch_size])
        
        self.prompt = self.patch * mask

        return x + self.prompt

    
def unfreeze(backbone):
    backbone.train()
    for m in backbone.parameters():
        m.stop_gradient = False
    return backbone


def train(student, backbone, projector, train_loader, test_loader, unlabelled_train_loader, args, prompt_model):
    # 设置优化器参数组
    params_groups = get_params_groups(student)
    optimizer = paddle.optimizer.Momentum(
        learning_rate=args.lr,
        parameters=params_groups,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    fp16_scaler = None
    if args.fp16:
        fp16_scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
    tmp_lr = paddle.optimizer.lr.CosineAnnealingDecay(
        T_max=args.epochs, eta_min=args.lr * 0.001, learning_rate=optimizer.get_lr()
    )
    optimizer.set_lr_scheduler(tmp_lr)
    exp_lr_scheduler = tmp_lr

    # 定义聚类损失（你需自己定义DistillLoss对应的paddle版本）
    cluster_criterion = DistillLoss(
        args.warmup_teacher_temp_epochs,
        args.epochs,
        args.n_views,
        args.warmup_teacher_temp,
        args.teacher_temp,
    )

    # 初始化 qhat (你需定义 initial_qhat 的paddle版本)
    qhat = initial_qhat(class_num=args.num_labeled_classes + args.num_unlabeled_classes)

    loss_record = AverageMeter()  # 你需要改写 AverageMeter 或直接用列表记录

    for epoch in range(args.epochs):
        student.train()
        backbone.train()
        projector.train()
        prompt_model.train()
        prompter.eval()
        unfreeze(prompt_model)  # 你需要写 unfreeze 的 paddle 版本，类似设置参数 requires_grad = True

        for batch_idx, batch in enumerate(train_loader):
            images, class_labels, uq_idxs, mask_lab = batch

            mask_lab = mask_lab[:, 0]
            # 转移到GPU (Paddle自动管理，无需手动cuda，但要转成Tensor且放入GPU)
            class_labels = paddle.to_tensor(class_labels).to(device,
                blocking=not True
            )
            mask_lab = paddle.to_tensor(mask_lab).to(device,
                blocking=not True
            ).astype('bool')
            images = paddle.concat([paddle.to_tensor(im) for im in images], axis=0)

            # AMP自动混合精度上下文
            with paddle.amp.auto_cast(enable=fp16_scaler is not None):
                feat, _ = backbone(images, prompt_model)
                student_proj, student_out = projector(feat)

                # teacher输出detach
                teacher_out = student_out.detach()

                # * clustering supervised loss
                sup_logits = paddle.concat([f[mask_lab] / 0.1 for f in paddle.chunk(student_out, chunks=2, axis=0)], axis=0)
                sup_labels = paddle.concat([class_labels[mask_lab] for _ in range(2)], axis=0)
                cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

                # * clustering unsupervised loss
                cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
                avg_probs = F.softmax(student_out / 0.1, axis=1).mean(axis=0)
                me_max_loss = -paddle.sum(paddle.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
                cluster_loss += args.memax_weight * me_max_loss

                # * representation learning unsupervised loss (InfoNCE)
                contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj)
                contrastive_loss = nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

                # * representation learning supervised loss (SupCon)
                # chunk后按mask_lab索引，然后unsqueeze维度1，再拼接
                student_proj_chunks = paddle.chunk(student_proj, chunks=2, axis=0)
                student_proj = paddle.concat([f[mask_lab].unsqueeze(1) for f in student_proj_chunks], axis=1)
                student_proj = F.normalize(student_proj, axis=-1)
                sup_con_labels = class_labels[mask_lab]
                sup_con_loss = SupConLoss()(student_proj, labels=sup_con_labels)

                # * Soft loss part
                unsup_logits = F.softmax(student_out / 0.1, axis=-1)
                max_probs_, idx = paddle.topk(unsup_logits, k=1, axis=-1)
                max_probs_ = max_probs_.squeeze(-1)
                idx = idx.squeeze(-1)
                mask_all = max_probs_ >= args.thr

                mask_lab_p = paddle.concat([mask_lab, mask_lab], axis=0)
                labels_p = paddle.concat([class_labels, class_labels], axis=0)
                mask_p_true = (~mask_lab_p) & mask_all
                mask_p = mask_p_true & (labels_p == idx)

                mask_old = paddle.zeros_like(mask_p_true, dtype='bool')
                mask_old = mask_old.numpy()
                idx_lt = (idx < args.num_labeled_classes).numpy()
                mask_p_true_np = mask_p_true.numpy()
                mask_old[(idx_lt) & (mask_p_true_np)] = True
                mask_old = paddle.to_tensor(mask_old)

                mask_old_num = paddle.sum(mask_old.astype('int32')).item()

                # * Unlabeled true masks
                mask_condidate_unlabeled = (~mask_lab_p) & mask_all
                mask_p_unlabeled_true = mask_condidate_unlabeled & (labels_p == idx)

                mask_p_unlabeled_true_np = mask_p_unlabeled_true.numpy()
                idx_np = idx.numpy()
                labels_p_np = labels_p.numpy()

                mask_p_unlabeled_true_old = paddle.to_tensor((~mask_lab_p.numpy()) & mask_all.numpy() & (labels_p_np == idx_np) & (idx_np < args.num_labeled_classes))
                mask_p_unlabeled_true_novel = paddle.to_tensor((~mask_lab_p.numpy()) & mask_all.numpy() & (labels_p_np == idx_np) & (idx_np >= args.num_labeled_classes))

                Unlabeled_true_to_old = paddle.sum(mask_p_unlabeled_true_old.astype('int32')).item()
                Unlabeled_true_to_novel = paddle.sum(mask_p_unlabeled_true_novel.astype('int32')).item()

                mask_p_unlabeled_wrong = mask_condidate_unlabeled & (labels_p != idx)
                Unlabeled_condiate = paddle.sum(mask_condidate_unlabeled.astype('int32')).item()
                Unlabeled_true = paddle.sum(mask_p_unlabeled_true.astype('int32')).item()
                Unlabeled_wrong = paddle.sum(mask_p_unlabeled_wrong.astype('int32')).item()

                # # Wrong classification analysis
                # idx_ = labels_p[mask_p_unlabeled_wrong]
                # idx_pre = idx[mask_p_unlabeled_wrong]

                # idx_num_old = idx_ < args.num_labeled_classes
                # idx_num_novel = idx_ >= args.num_labeled_classes

                # idx_num_novel_to_old = (idx_ >= args.num_labeled_classes) & (idx_pre < args.num_labeled_classes)
                # idx_num_novel_to_novel = (idx_ >= args.num_labeled_classes) & (idx_pre >= args.num_labeled_classes)

                # num_wrong_old = paddle.sum(idx_num_old.astype('int32')).item()
                # num_wrong_novel = paddle.sum(idx_num_novel.astype('int32')).item()

                # num_wrong_novel_to_old = paddle.sum(idx_num_novel_to_old.astype('int32')).item()
                # num_wrong_novel_to_novel = paddle.sum(idx_num_novel_to_novel.astype('int32')).item()

                # 计算nll loss
                pseudo_label = F.softmax(student_out / 0.05, axis=-1)
                delta_logits = paddle.log(qhat)
                logits_u = student_out / 0.05 + 0.4 * delta_logits
                log_pred = F.log_softmax(logits_u, axis=-1)
                nll_loss = paddle.sum(-pseudo_label * log_pred, axis=1) * mask_old.astype('float32')

                # 更新qhat
                qhat = update_qhat(F.softmax(student_out.detach(), axis=-1), qhat, momentum=args.qhat_m)

                # 汇总日志字符串
                pstr = f'cls_loss: {cls_loss.item():.4f} cluster_loss: {cluster_loss.item():.4f} sup_con_loss: {sup_con_loss.item():.4f} contrastive_loss: {contrastive_loss.item():.4f} nll_loss: {nll_loss.mean().item():.4f}'

                # 计算总损失
                loss = 0
                loss += (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss
                loss += (1 - args.sup_weight) * contrastive_loss + args.sup_weight * sup_con_loss
                loss += 2 * nll_loss.mean()

            loss_record.update(loss.item(), class_labels.shape[0])

            optimizer.clear_gradients()
            if fp16_scaler is None:
                loss.backward()
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                fp16_scaler.step(optimizer)
                fp16_scaler.update()

            if batch_idx % args.print_freq == 0:
                args.logger.info(
                    "Epoch: [{}][{}/{}]\t loss {:.5f}\t {} Unlabeled condidate {} Unlabeled pred Old {} Unabeled true {} Unlabeled true_to_old {} Unlabeled true_to_novel {} Unabeled wrong {}".format(
                        epoch,
                        batch_idx,
                        len(train_loader),
                        loss.item(),
                        pstr,
                        Unlabeled_condiate,
                        mask_old_num,
                        Unlabeled_true,
                        Unlabeled_true_to_old,
                        Unlabeled_true_to_novel,
                        Unlabeled_wrong,
                        # num_wrong_old,
                        # num_wrong_novel,
                        # num_wrong_novel_to_old,
                        # num_wrong_novel_to_novel,
                    )
                )

        args.logger.info(f'Train Epoch: {epoch} Avg Loss: {loss_record.avg:.4f}')
        args.logger.info('Testing on unlabelled examples in the training data...')
        all_acc, old_acc, new_acc = test(student, backbone, projector, unlabelled_train_loader, epoch=epoch, save_name='Train ACC Unlabelled', args=args, prompt_model=prompt_model)
        args.logger.info(f'Train Accuracies: All {all_acc:.4f} | Old {old_acc:.4f} | New {new_acc:.4f}')

        # 更新学习率
        exp_lr_scheduler.step()

        # 保存模型
        save_dict = {
            'model': student.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
        }
        paddle.save(save_dict, args.model_path)
        args.logger.info(f"model saved to {args.model_path}.")

from tqdm import tqdm

def test(model, backbone, projector, test_loader, epoch, save_name, args, prompt_model):
    model.eval()
    backbone.eval()
    projector.eval()
    prompt_model.eval()

    preds, targets = [], []
    mask = np.array([])

    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        # Paddle中无需cuda调用，确保device设置正确即可
        images = paddle.to_tensor(images)
        label = paddle.to_tensor(label).astype('int64')

        with paddle.no_grad():
            feat, _ = backbone(images, prompt_model)
            _, logits = projector(feat)

            pred_labels = logits.argmax(axis=1).cpu().numpy()
            preds.append(pred_labels)
            targets.append(label.cpu().numpy())

            # mask用于标识label是否属于训练类别范围内
            mask_batch = np.array([True if x in range(len(args.train_classes)) else False for x in label.numpy()])
            mask = np.append(mask, mask_batch)

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    all_acc, old_acc, new_acc = log_accs_from_preds(
        y_true=targets,
        y_pred=preds,
        mask=mask,
        T=epoch,
        eval_funcs=args.eval_funcs,
        save_name=save_name,
        args=args
    )

    # 如果有可视化需求，可以在这里调用 visualize_attention(epoch, backbone, prompt_model)

    return all_acc, old_acc, new_acc


def check_weight_compatibility(model, pretrain_path):
    # 加载预训练权重
    pretrain_dict = paddle.load(pretrain_path)
    
    # 获取模型参数
    model_params = dict(model.named_parameters())
    
    # 键集合
    model_keys = set(model_params.keys())
    pretrain_keys = set(pretrain_dict.keys())
    
    # 打印检查结果
    print("="*50)
    print("Weight Compatibility Check")
    print("="*50)
    
    # 1. 检查缺失键
    missing_keys = model_keys - pretrain_keys
    print(f"\n[Missing keys in pretrain] ({len(missing_keys)})")
    for key in sorted(missing_keys):
        print(f"- {key}: {model_params[key].shape}")
    
    # 2. 检查多余键
    unexpected_keys = pretrain_keys - model_keys
    print(f"\n[Unexpected keys in pretrain] ({len(unexpected_keys)})")
    for key in sorted(unexpected_keys):
        print(f"- {key}: {pretrain_dict[key].shape}")
    
    # 3. 检查形状匹配
    matched_keys = model_keys & pretrain_keys
    shape_mismatch = []
    
    for key in sorted(matched_keys):
        model_shape = model_params[key].shape
        pretrain_shape = pretrain_dict[key].shape
        
        if model_shape != pretrain_shape:
            shape_mismatch.append((key, model_shape, pretrain_shape))
    
    print(f"\n[Shape mismatches] ({len(shape_mismatch)})")
    for key, model_shape, pretrain_shape in shape_mismatch:
        print(f"- {key}")
        print(f"  MODEL: {model_shape}")
        print(f"  PRETRAIN: {pretrain_shape}")
    
    # 4. 统计匹配情况
    print("\n" + "="*50)
    print(f"Total model parameters: {len(model_keys)}")
    print(f"Total pretrain parameters: {len(pretrain_keys)}")
    print(f"Perfectly matched parameters: {len(matched_keys) - len(shape_mismatch)}")
    print("="*50)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="cluster", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument(
        "--eval_funcs",
        nargs="+",
        help="Which eval functions to use",
        default=["v2", "v2p"],
    )
    parser.add_argument("--warmup_model_dir", type=str, default=None)
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="scars",
        help="options: cifar10, cifar100, imagenet_100, cub, scars, fgvc_aricraft, herbarium_19",
    )
    parser.add_argument("--prop_train_labels", type=float, default=0.5)
    parser.add_argument("--use_ssb_splits", action="store_true", default=True)
    parser.add_argument("--grad_from_block", type=int, default=11)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--thr", type=float, default=0.7)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--exp_root", type=str, default=exp_root)
    parser.add_argument("--transform", type=str, default="imagenet")
    parser.add_argument("--sup_weight", type=float, default=0.35)
    parser.add_argument("--n_views", default=2, type=int)
    parser.add_argument("--memax_weight", type=float, default=2)
    parser.add_argument(
        "--warmup_teacher_temp",
        default=0.07,
        type=float,
        help="Initial value for the teacher temperature.",
    )
    parser.add_argument(
        "--teacher_temp",
        default=0.04,
        type=float,
        help="Final value (after linear warmup)of the teacher temperature.",
    )
    parser.add_argument(
        "--warmup_teacher_temp_epochs",
        default=30,
        type=int,
        help="Number of warmup epochs for the teacher temperature.",
    )
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--print_freq", default=10, type=int)
    parser.add_argument("--exp_name", default=None, type=str)
    parser.add_argument(
        "--masked-qhat",
        action="store_true",
        default=False,
        help="update qhat with instances passing a threshold",
    )
    parser.add_argument(
        "--qhat_m", default=0.999, type=float, help="momentum for updating q_hat"
    )
    parser.add_argument("--e_cutoff", default=-5.4, type=int)
    parser.add_argument("--use_marginal_loss", default=False)
    parser.add_argument("--tau", default=0.4, type=float)
    parser.add_argument('--log_dir', type=str, default='./logs')
    args = parser.parse_args()
    # device = paddle.CUDAPlace(int("cuda:4".replace("cuda:", "")))
    writer = LogWriter(logdir=args.log_dir)
    args.writer = writer
    device = "gpu:4"
    paddle.device.set_device(
        device="gpu:4"
        if isinstance(device, int)
        else str(device).replace("cuda", "gpu")
    )
    args = get_class_splits(args)
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)
    init_experiment(args, runner_name=["legogcd"])
    args.logger.info(f"Using evaluation function {args.eval_funcs[0]} to print results")
    PaddleFlag.cudnn_benchmark = True
    ##插值方式不接受整数
    args.interpolation = 'bicubic'
    args.crop_pct = 0.875
    backbone = vit_prompt.__dict__["vit_base"]().to(device)

    # if args.warmup_model_dir is not None:
    #     args.logger.info(f"Loading weights from {args.warmup_model_dir}")
    #     backbone.set_state_dict(state_dict=paddle.load(path=str(args.warmup_model_dir)))
    backbone.set_state_dict(
        state_dict=paddle.load(
            path=str("/data/aptgcd_paddle/dino_vitbase16_pretrain.pdparams")
        )
    )
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes
    prompt_model = PromptResNet().to(device)
    args.patch_size = 16
    args.prompt_size = 1
    prompter = PatchPrompter(args)
    for m in backbone.parameters():
        m.stop_gradient = not False
    for name, m in backbone.named_parameters():
        if "block" in name:
            block_num = int(name.split(".")[1])
            if block_num >= args.grad_from_block:
                m.stop_gradient = not True
    args.logger.info("model build")
    train_transform, test_transform = get_transform(
        args.transform, image_size=args.image_size, args=args
    )
    train_transform = ContrastiveLearningViewGenerator(
        base_transform=train_transform, n_views=args.n_views
    )

    print(f"prop_train_labels = {args.prop_train_labels}")


    (
        train_dataset,
        test_dataset,
        unlabelled_train_examples_test,
        train_labelled,
    ) = get_datasets(args.dataset_name, train_transform, test_transform, args)
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [
        (1 if i < label_len else label_len / unlabelled_len)
        for i in range(len(train_dataset))
    ]
    sample_weights = paddle.to_tensor(data=sample_weights, dtype="float64")
    sampler = paddle.io.WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_dataset), replacement=True
    )
    batch_sampler = paddle.io.BatchSampler(sampler, batch_size=args.batch_size, drop_last=True, shuffle=True)
    
    
    # train_loader = paddle.io.DataLoader(
    #     train_dataset,
    #     num_workers=args.num_workers,
    #     batch_sampler=sampler,
    #     use_shared_memory=args.num_workers > 0,
    # )
    train_loader = paddle.io.DataLoader(
        train_dataset,
        num_workers=args.num_workers,
        batch_sampler=batch_sampler,  # ✅ 修复了这里，sampler改成了batch_sampler
        use_shared_memory=args.num_workers > 0
    )

    
    test_loader_unlabelled = paddle.io.DataLoader(
        dataset=unlabelled_train_examples_test,
        num_workers=args.num_workers,
        batch_size=256,
        shuffle=False,
    )
    test_loader_labelled = paddle.io.DataLoader(
        dataset=test_dataset,
        num_workers=args.num_workers,
        batch_size=256,
        shuffle=False,
    )
    projector = DINOHead(
        in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers
    )
    classifier = paddle.nn.Sequential(backbone, projector).to(device)
    model = paddle.nn.Sequential(prompter, classifier).to(device)


    backbone.set_state_dict(
        state_dict=paddle.load(
            path=str("/data/zbp05/ZW/aptgcd_paddle/dino_vitbase16_pretrain.pdparams")
        )
    )


    check_weight_compatibility(backbone, "/data/aptgcd_paddle/dino_vitbase16_pretrain.pdparams")

    backbone.to(device)
    projector.to(device)
    train(
        model,
        backbone,
        projector,
        train_loader,
        test_loader_labelled,
        test_loader_unlabelled,
        args,
        prompt_model
    )
