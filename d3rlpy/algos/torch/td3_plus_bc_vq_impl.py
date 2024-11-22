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

class TD3PlusBC_VQImpl(TD3Impl):

    _alpha: float

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
        actor_optim_factory: OptimizerFactory,
        critic_optim_factory: OptimizerFactory,
        actor_encoder_factory: EncoderFactory,
        critic_encoder_factory: EncoderFactory,
        q_func_factory: QFunctionFactory,
        gamma: float,
        tau: float,
        n_critics: int,
        target_smoothing_sigma: float,
        target_smoothing_clip: float,
        alpha: float,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        action_scaler: Optional[ActionScaler],
        reward_scaler: Optional[RewardScaler],
        env_name: str = '',
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            actor_optim_factory=actor_optim_factory,
            critic_optim_factory=critic_optim_factory,
            actor_encoder_factory=actor_encoder_factory,
            critic_encoder_factory=critic_encoder_factory,
            q_func_factory=q_func_factory,
            gamma=gamma,
            tau=tau,
            n_critics=n_critics,
            target_smoothing_sigma=target_smoothing_sigma,
            target_smoothing_clip=target_smoothing_clip,
            use_gpu=use_gpu,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
        )
        self._alpha = alpha

        env_name_ = env_name.split('-')
        self.env_name = env_name_[0] + '-' + env_name_[-1]
        self._obs_max = torch.Tensor(ENV_OBS_RANGE[self.env_name]['max']).to(
            'cuda:{}'.format(self._use_gpu.get_id()))
        self._obs_min = torch.Tensor(ENV_OBS_RANGE[self.env_name]['min']).to(
            'cuda:{}'.format(self._use_gpu.get_id()))

        self._obs_max_norm = self._obs_min_norm = None

    def init_range_of_norm_obs(self):
        """
        初始化归一化观测值的范围。
    
        根据环境的观测值最大最小范围，通过标量变换将其映射到归一化空间。
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
    
        在更新目标策略的参数后，确保编码字典（codebooks）保持同步。
        """
        assert self._policy.vq_input is not None, "当前策略的向量量化输入不存在。"
        with torch.no_grad():
            self._targ_policy.vq_input.codebooks.data.copy_(
                self._policy.vq_input.codebooks.data
            )
    
    
    def compute_actor_loss(self, batch: TorchMiniBatch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """
        计算Actor网络的损失，包括策略损失（Actor Loss）和行为克隆损失（BC Loss）。
    
        Args:
            batch (TorchMiniBatch): 包含当前小批量的观测值、动作、奖励等数据。
    
        Returns:
            Tuple: 总损失、Actor损失、BC损失和额外的输出信息。
        """
        assert self._policy is not None, "Actor网络尚未初始化。"
        assert self._q_func is not None, "Q函数尚未初始化。"
    
        # 从策略生成动作
        action, extra_outs = self._policy(batch.observations)
        # 使用Q函数计算Q值
        q_t = self._q_func(batch.observations, action, "none")[0]
    
        # 动态调整的系数
        lam = self._alpha / (q_t.abs().mean()).detach()
    
        # 策略损失：目标为最大化Q值，因此负号转换为最小化
        actor_loss = lam * -q_t.mean()
    
        # 行为克隆损失：将当前动作与目标动作进行均方误差计算
        bc_loss = ((batch.actions - action) ** 2).mean()
    
        # 总损失为策略损失和行为克隆损失的和
        total_loss = actor_loss + bc_loss
    
        return total_loss, actor_loss, bc_loss, extra_outs
    
    
    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        """
        计算目标值（Target Q值），用于更新Critic网络。
    
        Args:
            batch (TorchMiniBatch): 包含下一步观测值、奖励等数据的小批量。
    
        Returns:
            torch.Tensor: 目标值张量。
        """
        assert self._targ_policy is not None, "目标策略尚未初始化。"
        assert self._targ_q_func is not None, "目标Q函数尚未初始化。"
    
        with torch.no_grad():
            # 使用目标策略生成下一步的动作
            action, _ = self._targ_policy(batch.next_observations)
    
            # 添加噪声实现目标动作的平滑
            noise = torch.randn(action.shape, device=batch.device)
            scaled_noise = self._target_smoothing_sigma * noise
            clipped_noise = scaled_noise.clamp(
                -self._target_smoothing_clip, self._target_smoothing_clip
            )
            smoothed_action = action + clipped_noise
            clipped_action = smoothed_action.clamp(-1.0, 1.0)
    
            # 使用目标Q函数计算目标值，取最小值以提高稳定性
            return self._targ_q_func.compute_target(
                batch.next_observations,
                clipped_action,
                reduction="min",
            )
    
    
    def compute_critic_loss(self, batch: TorchMiniBatch, q_tpn: torch.Tensor) -> torch.Tensor:
        """
        计算Critic网络的损失。
    
        Args:
            batch (TorchMiniBatch): 包含观测值、动作、奖励、终止标志等数据的小批量。
            q_tpn (torch.Tensor): 目标Q值。
    
        Returns:
            torch.Tensor: Critic损失值。
        """
        assert self._q_func is not None, "Q函数尚未初始化。"
    
        return self._q_func.compute_error(
            observations=batch.observations,
            actions=batch.actions,
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.n_steps,  # 考虑n步奖励的折扣因子
        )
    
    
    @train_api
    @torch_api()
    def update_critic(self, batch: TorchMiniBatch) -> Tuple[np.ndarray, Tuple]:
        """
        更新Critic网络。
    
        Args:
            batch (TorchMiniBatch): 当前小批量的训练数据。
    
        Returns:
            Tuple: Critic总损失值以及额外的日志信息。
        """
        assert self._critic_optim is not None, "Critic优化器尚未初始化。"
    
        with torch.no_grad():
            # 预测当前Q值
            q_prediction = self._q_func(batch.observations, batch.actions, reduction="none")
            q1_pred = q_prediction[0].cpu().detach().numpy().mean()
            q2_pred = q_prediction[1].cpu().detach().numpy().mean()
    
        # 清零优化器梯度
        self._critic_optim.zero_grad()
    
        # 计算目标Q值
        q_tpn = self.compute_target(batch)
    
        # 计算Critic损失
        loss = self.compute_critic_loss(batch, q_tpn)
    
        # 反向传播并更新参数
        loss.backward()
        self._critic_optim.step()
    
        # 记录额外日志
        extra_logs = (q_tpn.cpu().detach().numpy().mean(), q1_pred, q2_pred, loss.item())
        return loss.cpu().detach().numpy(), extra_logs
    
    
    @train_api
    @torch_api()
    def update_actor(self, batch: TorchMiniBatch) -> Tuple[np.ndarray, Tuple]:
        """
        更新Actor网络。
    
        Args:
            batch (TorchMiniBatch): 当前小批量的训练数据。
    
        Returns:
            Tuple: Actor总损失值以及额外的日志信息。
        """
        assert self._q_func is not None, "Q函数尚未初始化。"
        assert self._actor_optim is not None, "Actor优化器尚未初始化。"
    
        # 将Q函数设置为推断模式以提高稳定性
        self._q_func.eval()
    
        # 清零优化器梯度
        self._actor_optim.zero_grad()
    
        # 计算Actor损失
        loss, actor_loss, bc_loss, extra_outs = self.compute_actor_loss(batch)
    
        # 反向传播并更新参数
        loss.backward()
        self._actor_optim.step()
    
        # 获取向量量化损失（如果有）
        vq_loss = extra_outs.get("vq_loss", -1.0)  # -1为无效值，仅作日志记录
        extra_logs = (
            actor_loss.cpu().detach().numpy(),
            bc_loss.cpu().detach().numpy(),
            vq_loss.item(),
        )
        return loss.cpu().detach().numpy(), extra_logs
