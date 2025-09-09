import numpy as np
import paddle
import paddle.nn.functional as F
############################## 相关utils函数，如下 ##############################

def view(self, *args, **kwargs):
    if args:
        if len(args)==1 and isinstance(args[0], (tuple, list, str)):
            return paddle.view(self, args[0])
        else:
            return paddle.view(self, list(args))
    elif kwargs:
        return paddle.view(self, shape_or_dtype = list(kwargs.values())[0])

setattr(paddle.Tensor, 'view', view)
############################## 相关utils函数，如上 ##############################



class DINOHead(paddle.nn.Layer):
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        norm_last_layer=True,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = paddle.nn.Linear(in_features=in_dim, out_features=bottleneck_dim)
        elif nlayers != 0:
            layers = [paddle.nn.Linear(in_features=in_dim, out_features=hidden_dim)]
            if use_bn:
                layers.append(paddle.nn.BatchNorm1D(num_features=hidden_dim))
            layers.append(paddle.nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(
                    paddle.nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
                )
                if use_bn:
                    layers.append(paddle.nn.BatchNorm1D(num_features=hidden_dim))
                layers.append(paddle.nn.GELU())
            layers.append(
                paddle.nn.Linear(in_features=hidden_dim, out_features=bottleneck_dim)
            )
            self.mlp = paddle.nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = paddle.nn.utils.weight_norm(
            layer=paddle.nn.Linear(
                in_features=in_dim, out_features=out_dim, bias_attr=False
            )
        )
        # self.last_layer.weight_g.data.fill_(value=1)
        new_tensor = paddle.full_like(self.last_layer.weight_g, 1.0)
        self.last_layer.weight_g.set_value(new_tensor)
        
        if norm_last_layer:
            self.last_layer.weight_g.stop_gradient = not False

    def _init_weights(self, m):
        if isinstance(m, paddle.nn.Linear):
            init_TruncatedNormal = paddle.nn.initializer.TruncatedNormal(std=0.02)
            init_TruncatedNormal(m.weight)
            if isinstance(m, paddle.nn.Linear) and m.bias is not None:
                init_Constant = paddle.nn.initializer.Constant(value=0)
                init_Constant(m.bias)

    def forward(self, x):
        x_proj = self.mlp(x)
        x = paddle.nn.functional.normalize(x=x, axis=-1, p=2)
        logits = self.last_layer(x)
        return x_proj, logits


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        if not isinstance(self.base_transform, list):
            return [self.base_transform(x) for i in range(self.n_views)]
        else:
            return [self.base_transform[i](x) for i in range(self.n_views)]


import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class SupConLoss(nn.Layer):
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        Args:
            features: [batch_size, n_views, ...]
            labels: [batch_size]
            mask: [batch_size, batch_size]
        Returns:
            loss scalar
        """
        if len(features.shape) < 3:
            raise ValueError("`features` needs to be [bsz, n_views, ...], at least 3 dimensions are required")

        if len(features.shape) > 3:
            features = paddle.reshape(features, [features.shape[0], features.shape[1], -1])

        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")

        # device info from features
        place = features.place

        if labels is None and mask is None:
            mask = paddle.eye(batch_size, dtype='float32')
        elif labels is not None:
            labels = paddle.reshape(labels, [-1, 1])
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = paddle.equal(labels, labels.transpose([1, 0])).astype('float32')
        else:
            mask = mask.astype('float32')

        contrast_count = features.shape[1]
        contrast_feature = paddle.concat(x=paddle.unstack(features, axis=1), axis=0)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = paddle.matmul(anchor_feature, contrast_feature.T) / self.temperature

        # for numerical stability
        logits_max = paddle.max(anchor_dot_contrast, axis=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = paddle.tile(mask, repeat_times=[anchor_count, contrast_count])

        # mask-out self-contrast cases
        logits_mask = 1.0 - paddle.eye(batch_size * anchor_count, dtype='float32')

        mask = mask * logits_mask

        # compute log_prob
        exp_logits = paddle.exp(logits) * logits_mask
        log_prob = logits - paddle.log(paddle.sum(exp_logits, axis=1, keepdim=True) + 1e-12)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = paddle.sum(mask * log_prob, axis=1) / (paddle.sum(mask, axis=1) + 1e-12)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = paddle.reshape(loss, [anchor_count, batch_size])
        return paddle.mean(loss)



def info_nce_logits(features, n_views=2, temperature=1.0, device='gpu:4'):
    b_ = 0.5 * int(features.shape[0])

    # 创建标签
    labels = paddle.concat([paddle.arange(b_) for _ in range(n_views)], axis=0)
    labels = paddle.cast(paddle.unsqueeze(labels, axis=0) == paddle.unsqueeze(labels, axis=1), 'float32')
    labels = paddle.to_tensor(labels, place=paddle.CUDAPlace(int(device.split(':')[-1])))

    # L2 normalize features
    features = F.normalize(features, axis=1)

    # 相似度矩阵
    similarity_matrix = paddle.matmul(features, features.T)

    # 构造 mask，去除对角线
    mask = paddle.cast(paddle.eye(labels.shape[0], dtype='int32'), 'bool')


    # 去掉对角线元素
    labels = paddle.masked_select(labels, ~mask)
    labels = paddle.reshape(labels, [labels.shape[0] // (labels.shape[0] // n_views - 1), -1])

    similarity_matrix = paddle.masked_select(similarity_matrix, ~mask)
    similarity_matrix = paddle.reshape(similarity_matrix, [labels.shape[0], -1])

    # positives
    positives = paddle.masked_select(similarity_matrix, paddle.cast(labels, 'bool'))
    positives = paddle.reshape(positives, [labels.shape[0], -1])

    # negatives
    negatives = paddle.masked_select(similarity_matrix, ~paddle.cast(labels, 'bool'))
    negatives = paddle.reshape(negatives, [labels.shape[0], -1])

    # logits 和 labels
    logits = paddle.concat([positives, negatives], axis=1)
    labels = paddle.zeros([logits.shape[0]], dtype='int64')
    labels = paddle.to_tensor(labels, place=paddle.CUDAPlace(int(device.split(':')[-1])))

    # 温度缩放
    logits = logits / temperature
    return logits, labels



def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if param.stop_gradient:  
            continue
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]



class DistillLoss(paddle.nn.Layer):
    def __init__(
        self,
        warmup_teacher_temp_epochs,
        nepochs,
        ncrops=2,
        warmup_teacher_temp=0.07,
        teacher_temp=0.04,
        student_temp=0.1,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.ncrops = ncrops
        self.teacher_temp_schedule = np.concatenate(
            (
                np.linspace(
                    warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs
                ),
                np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp,
            )
        )

    def _gaussian(self, p):
        return paddle.exp(x=-((p - 0.5) ** 2) / (2 * 0.1**2)) + 0.0

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(chunks=self.ncrops)
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = paddle.nn.functional.softmax(x=teacher_output / temp, axis=-1)
        teacher_out = teacher_out.detach().chunk(chunks=2)
        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue
                loss = paddle.sum(
                    x=-q * paddle.nn.functional.log_softmax(x=student_out[v], axis=-1),
                    axis=-1,
                )
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        student_out_1, student_out_2 = student_out
        student_out_1 = paddle.nn.functional.softmax(x=student_out_1, axis=-1)
        student_out_2 = paddle.nn.functional.softmax(x=student_out_2, axis=-1)
        loss_entropy = paddle.sum(
            x=student_out_1
            * (
                paddle.nn.functional.log_softmax(x=student_out_1, axis=-1)
                - paddle.nn.functional.log_softmax(x=student_out_2, axis=-1)
            ),
            axis=-1,
        )
        total_loss += loss_entropy.mean()
        return total_loss


class WeightedEntropyLoss(paddle.nn.Layer):
    def __init__(self, miu=0.5, sigma=0.1, beta=0.0):
        super(WeightedEntropyLoss, self).__init__()
        self.miu = miu
        self.sigma = sigma
        self.eps = paddle.finfo(dtype="float32").eps
        self.beta = beta

    def _gaussian(self, p):
        return paddle.exp(x=-((p - self.miu) ** 2) / (2 * self.sigma**2)) + self.beta

    def forward(self, p):
        return -paddle.sum(
            x=(
                p * paddle.log(x=p + self.eps)
                + (1 - p) * paddle.log(x=1 - p + self.eps)
            )
            * self._gaussian(p)
        ) / (tuple(p.shape)[0] * tuple(p.shape)[1])


def causal_inference(current_logit, qhat, exp_idx, tau=0.5):
    debiased_prob = current_logit - tau * paddle.log(x=qhat)
    return debiased_prob


def update_qhat(probs, qhat, momentum, qhat_mask=None):
    if qhat_mask is not None:
        mean_prob = probs.detach() * qhat_mask.detach().unsqueeze(axis=-1)
    else:
        mean_prob = probs.detach().mean(axis=0)
    qhat = momentum * qhat + (1 - momentum) * mean_prob
    return qhat


def initial_qhat(class_num=1000,device='gpu:4'):
    qhat = (paddle.ones(shape=[1, class_num], dtype="float32") / class_num).to(device,
        blocking=True
    )
    return qhat