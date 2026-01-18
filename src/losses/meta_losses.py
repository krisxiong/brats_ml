"""
元学习损失函数
"""

import torch
from torch.autograd import grad
from torch.func import functional_call

class MAMLLoss:
    """MAML损失函数（支持一阶和二阶近似）"""

    def __init__(self, l2_reg=0.001, first_order=True):
        self.l2_reg = l2_reg
        self.first_order = first_order

    def compute_adapted_loss(self, model, support_x, support_y, adapted_params, loss_fn):
        """
        计算适应后的损失（包含L2正则化）
        """
        # 使用适应后的参数进行前向传播
        logits = self._forward_with_params(model, support_x, adapted_params)

        # 计算任务损失
        task_loss = loss_fn(logits, support_y)

        # L2正则化（防止过拟合support）
        l2_loss = 0
        for param in adapted_params.values():
            if param.requires_grad:
                l2_loss += torch.norm(param, p=2)

        total_loss = task_loss
        return total_loss

    def compute_gradients(self, loss, adapted_params, create_graph=False):
        """计算梯度（支持一阶近似）"""
        params_with_grad = [p for p in adapted_params.values() if p.requires_grad]

        if len(params_with_grad) == 0:
            return None

        grads = grad(
            loss,
            params_with_grad,
            create_graph=create_graph and not self.first_order,
            allow_unused=True,
            retain_graph=False
        )

        return grads

    def _forward_with_params(self, model, x, params):
        """使用给定参数前向传播（保持计算图）"""
        # 将参数转换为字典格式
        params_dict = {name: param for name, param in params.items()}

        # 使用 functional_call 进行前向传播
        # 这会保持完整的计算图连接
        output = functional_call(model, params_dict, x)

        return output