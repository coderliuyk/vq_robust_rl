# pylint: disable=too-many-ancestors

from typing import Optional, Sequence, Tuple, Any

import torch
import numpy as np

from ...gpu import Device
from ...models.encoders import EncoderFactory
from ...models.optimizers import OptimizerFactory
from ...models.q_functions import QFunctionFactory
from ...preprocessing import ActionScaler, RewardScaler, Scaler
from ...torch_utility import TorchMiniBatch, torch_api, train_api
from .td3_impl import TD3Impl
from ...adversarial_training import ENV_OBS_RANGE

# pylint: disable=too-many-ancestors

class TD3PlusBC_VQImpl(TD3Impl):
    """
    TD3+BC算法的实现，带有向量量化（VQ）模块。

    该类继承自TD3Impl，并在其基础上增加了行为克隆损失和向量量化支持。

    Attributes:
        _alpha (float): 损失平衡系数，用于调节行为克隆损失和策略损失的比重。
    """

    def __init__(...):
        """
        初始化TD3PlusBC_VQImpl类。

        Args:
            observation_shape (Sequence[int]): 环境观测值的形状。
            action_size (int): 动作空间的维度。
            actor_learning_rate (float): Actor网络的学习率。
            critic_learning_rate (float): Critic网络的学习率。
            actor_optim_factory (OptimizerFactory): Actor网络的优化器工厂。
            critic_optim_factory (OptimizerFactory): Critic网络的优化器工厂。
            actor_encoder_factory (EncoderFactory): Actor网络的编码器工厂。
            critic_encoder_factory (EncoderFactory): Critic网络的编码器工厂。
            q_func_factory (QFunctionFactory): Q函数工厂。
            gamma (float): 折扣因子。
            tau (float): 软更新系数。
            n_critics (int): Critic网络的数量。
            target_smoothing_sigma (float): 目标平滑噪声的标准差。
            target_smoothing_clip (float): 目标平滑噪声的剪切范围。
            alpha (float): 损失平衡系数。
            use_gpu (Optional[Device]): GPU设备（如果可用）。
            scaler (Optional[Scaler]): 状态归一化工具。
            action_scaler (Optional[ActionScaler]): 动作归一化工具。
            reward_scaler (Optional[RewardScaler]): 奖励归一化工具。
            env_name (str): 环境名称。
        """
        super().__init__(...)
        self._alpha = alpha

        # 初始化环境名称和观测范围
        env_name_ = env_name.split('-')
        self.env_name = env_name_[0] + '-' + env_name_[-1]
        self._obs_max = torch.Tensor(ENV_OBS_RANGE[self.env_name]['max']).to(
            'cuda:{}'.format(self._use_gpu.get_id()))
        self._obs_min = torch.Tensor(ENV_OBS_RANGE[self.env_name]['min']).to(
            'cuda:{}'.format(self._use_gpu.get_id()))

        self._obs_max_norm = self._obs_min_norm = None

    def init_range_of_norm_obs(self):
        """
        初始化归一化后的观测值范围。

        根据Scaler工具，将环境的最大值和最小值映射到归一化空间。
        """
        self._obs_max_norm = self.scaler.transform(
            torch.Tensor(ENV_OBS_RANGE[self.env_name]['max']).to('cuda:{}'.format(
                self._use_gpu.get_id()))
        )
        self._obs_min_norm = self.scaler.transform(
            torch.Tensor(ENV_OBS_RANGE[self.env_name]['min']).to('cuda:{}'.format(
                self._use_gpu.get_id()))
        )

    def sync_codebook_from_policy(self):
        """
        从当前策略同步编码字典到目标策略。

        确保向量量化的编码字典（Codebooks）在策略和目标策略之间保持一致。
        """
        assert self._policy.vq_input is not None, "VQ模块未初始化。"
        with torch.no_grad():
            self._targ_policy.vq_input.codebooks.data.copy_(
                self._policy.vq_input.codebooks.data
            )

    def compute_actor_loss(self, batch: TorchMiniBatch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """
        计算Actor网络的损失，包括策略损失和行为克隆损失。

        Args:
            batch (TorchMiniBatch): 包含观测值和动作的小批量数据。

        Returns:
            Tuple: 总损失、策略损失、行为克隆损失以及额外的输出信息。
        """
        assert self._policy is not None, "Actor网络未初始化。"
        assert self._q_func is not None, "Q函数未初始化。"

        # 获取策略生成的动作和额外输出
        action, extra_outs = self._policy(batch.observations)

        # 计算策略损失：目标为最大化Q值，取负号变为最小化
        q_t = self._q_func(batch.observations, action, "none")[0]
        lam = self._alpha / (q_t.abs().mean()).detach()
        actor_loss = lam * -q_t.mean()

        # 计算行为克隆损失：最小化策略动作和真实动作之间的均方误差
        bc_loss = ((batch.actions - action) ** 2).mean()

        # 总损失
        total_loss = actor_loss + bc_loss
        return total_loss, actor_loss, bc_loss, extra_outs

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        """
        计算目标Q值，用于Critic网络的更新。

        Args:
            batch (TorchMiniBatch): 包含下一步观测值的小批量数据。

        Returns:
            torch.Tensor: 目标Q值。
        """
        assert self._targ_policy is not None, "目标策略未初始化。"
        assert self._targ_q_func is not None, "目标Q函数未初始化。"

        with torch.no_grad():
            # 使用目标策略生成下一步的动作
            action, _ = self._targ_policy(batch.next_observations)

            # 添加噪声以平滑目标动作
            noise = torch.randn(action.shape, device=batch.device)
            scaled_noise = self._target_smoothing_sigma * noise
            clipped_noise = scaled_noise.clamp(
                -self._target_smoothing_clip, self._target_smoothing_clip
            )
            smoothed_action = action + clipped_noise
            clipped_action = smoothed_action.clamp(-1.0, 1.0)

            # 使用目标Q函数计算目标Q值
            return self._targ_q_func.compute_target(
                batch.next_observations,
                clipped_action,
                reduction="min",
            )

    def compute_critic_loss(self, batch: TorchMiniBatch, q_tpn: torch.Tensor) -> torch.Tensor:
        """
        计算Critic网络的损失。

        Args:
            batch (TorchMiniBatch): 包含观测值、动作、奖励等数据的小批量。
            q_tpn (torch.Tensor): 目标Q值。

        Returns:
            torch.Tensor: Critic网络的损失。
        """
        assert self._q_func is not None, "Q函数未初始化。"

        return self._q_func.compute_error(
            observations=batch.observations,
            actions=batch.actions,
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.n_steps,
        )

    @train_api
    @torch_api()
    def update_critic(self, batch: TorchMiniBatch) -> Tuple[np.ndarray, Tuple]:
        """
        更新Critic网络。

        Args:
            batch (TorchMiniBatch): 包含训练数据的小批量。

        Returns:
            Tuple: Critic总损失和额外日志。
        """
        assert self._critic_optim is not None, "Critic优化器未初始化。"

        with torch.no_grad():
            # 预测当前Q值
            q_prediction = self._q_func(batch.observations, batch.actions, reduction="none")
            q1_pred = q_prediction[0].cpu().detach().numpy().mean()
            q2_pred = q_prediction[1].cpu().detach().numpy().mean()

        # 清零梯度
        self._critic_optim.zero_grad()

        # 计算目标Q值和损失
        q_tpn = self.compute_target(batch)
        loss = self.compute_critic_loss(batch, q_tpn)

        # 反向传播和优化
        loss.backward()
        self._critic_optim.step()

        # 记录日志
        extra_logs = (q_tpn.cpu().detach().numpy().mean(), q1_pred, q2_pred, loss.item())
        return loss.cpu().detach().numpy(), extra_logs

    @train_api
    @torch_api()
    def update_actor(self, batch: TorchMiniBatch) -> Tuple[np.ndarray, Tuple]:
        """
        更新Actor网络。

        Args:
            batch (TorchMiniBatch): 包含训练数据的小批量。

        Returns:
            Tuple: Actor总损失和额外日志。
        """
        assert self._q_func is not None, "Q函数未初始化。"
        assert self._actor_optim is not None, "Actor优化器未初始化。"

        # 设置Q函数为评估模式，提升稳定性
        self._q_func.eval()

        # 清零梯度
        self._actor_optim.zero_grad()

        # 计算Actor损失
        loss, actor_loss, bc_loss, extra_outs = self.compute_actor_loss(batch)

        # 反向传播和优化
        loss.backward()
        self._actor_optim.step()

        # 获取向量量化损失并记录日志
        vq_loss = extra_outs.get("
