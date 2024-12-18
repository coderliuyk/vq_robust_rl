from typing import Any, Dict, Optional, Sequence

from ..argument_utility import (
    ActionScalerArg,
    EncoderArg,
    QFuncArg,
    RewardScalerArg,
    ScalerArg,
    UseGPUArg,
    check_encoder,
    check_q_func,
    check_use_gpu,
)
from ..constants import IMPL_NOT_INITIALIZED_ERROR, ActionSpace
from ..dataset import TransitionMiniBatch
from ..gpu import Device
from ..models.encoders import EncoderFactory
from ..models.optimizers import AdamFactory, OptimizerFactory
from ..models.q_functions import QFunctionFactory
from .base import AlgoBase
from .torch.td3_plus_bc_vq_impl import TD3PlusBC_VQImpl


class TD3PlusBC_VQ(AlgoBase):
    r"""TD3+BC algorithm.

    TD3+BC is an simple offline RL algorithm built on top of TD3.
    TD3+BC introduces BC-reguralized policy objective function.

    .. math::

        J(\phi) = \mathbb{E}_{s,a \sim D}
            [\lambda Q(s, \pi(s)) - (a - \pi(s))^2]

    where

    .. math::

        \lambda = \frac{\alpha}{\frac{1}{N} \sum_(s_i, a_i) |Q(s_i, a_i)|}

    References:
        * `Fujimoto et al., A Minimalist Approach to Offline Reinforcement
          Learning. <https://arxiv.org/abs/2106.06860>`_

    Args:
        actor_learning_rate (float): learning rate for a policy function.
        critic_learning_rate (float): learning rate for Q functions.
        actor_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the actor.
        critic_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the critic.
        actor_encoder_factory (d3rlpy.models.encoders.EncoderFactory or str):
            encoder factory for the actor.
        critic_encoder_factory (d3rlpy.models.encoders.EncoderFactory or str):
            encoder factory for the critic.
        q_func_factory (d3rlpy.models.q_functions.QFunctionFactory or str):
            Q function factory.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        n_steps (int): N-step TD calculation.
        gamma (float): discount factor.
        tau (float): target network synchronization coefficiency.
        n_critics (int): the number of Q functions for ensemble.
        target_smoothing_sigma (float): standard deviation for target noise.
        target_smoothing_clip (float): clipping range for target noise.
        alpha (float): :math:`\alpha` value.
        update_actor_interval (int): interval to update policy function
            described as `delayed policy update` in the paper.
        use_gpu (bool, int or d3rlpy.gpu.Device):
            flag to use GPU, device ID or device.
        scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
            The available options are `['pixel', 'min_max', 'standard']`.
        action_scaler (d3rlpy.preprocessing.ActionScaler or str):
            action preprocessor. The available options are ``['min_max']``.
        reward_scaler (d3rlpy.preprocessing.RewardScaler or str):
            reward preprocessor. The available options are
            ``['clip', 'min_max', 'standard']``.
        impl (d3rlpy.algos.torch.td3_impl.TD3Impl): algorithm implementation.

    """

    _actor_learning_rate: float
    _critic_learning_rate: float
    _actor_optim_factory: OptimizerFactory
    _critic_optim_factory: OptimizerFactory
    _actor_encoder_factory: EncoderFactory
    _critic_encoder_factory: EncoderFactory
    _q_func_factory: QFunctionFactory
    _tau: float
    _n_critics: int
    _target_smoothing_sigma: float
    _target_smoothing_clip: float
    _alpha: float
    _update_actor_interval: int
    _use_gpu: Optional[Device]
    _impl: Optional[TD3PlusBC_VQImpl]

    def __init__(
        self,
        *,
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        actor_optim_factory: OptimizerFactory = AdamFactory(),
        critic_optim_factory: OptimizerFactory = AdamFactory(),
        actor_encoder_factory: EncoderArg = "default",
        critic_encoder_factory: EncoderArg = "default",
        q_func_factory: QFuncArg = "mean",
        batch_size: int = 256,
        n_frames: int = 1,
        n_steps: int = 1,
        gamma: float = 0.99,
        tau: float = 0.005,
        n_critics: int = 2,
        target_smoothing_sigma: float = 0.2,
        target_smoothing_clip: float = 0.5,
        alpha: float = 2.5,
        update_actor_interval: int = 2,
        use_gpu: UseGPUArg = False,
        scaler: ScalerArg = "standard",
        action_scaler: ActionScalerArg = None,
        reward_scaler: RewardScalerArg = None,
        impl: Optional[TD3PlusBC_VQImpl] = None,
        env_name: str = '',
        use_vq_in: bool = False,
        number_embeddings: int = 128, embedding_dim: int = 1,
        **kwargs: Any
    ):
        super().__init__(
            batch_size=batch_size,
            n_frames=n_frames,
            n_steps=n_steps,
            gamma=gamma,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
            kwargs=kwargs,
        )
        self._actor_learning_rate = actor_learning_rate
        self._critic_learning_rate = critic_learning_rate
        self._actor_optim_factory = actor_optim_factory
        self._critic_optim_factory = critic_optim_factory
        self._actor_encoder_factory = check_encoder(actor_encoder_factory)
        self._critic_encoder_factory = check_encoder(critic_encoder_factory)
        self._q_func_factory = check_q_func(q_func_factory)
        self._tau = tau
        self._n_critics = n_critics
        self._target_smoothing_sigma = target_smoothing_sigma
        self._target_smoothing_clip = target_smoothing_clip
        self._alpha = alpha
        self._update_actor_interval = update_actor_interval
        self._use_gpu = check_use_gpu(use_gpu)
        self._impl = impl
        self._env_name = env_name

        self._use_vq_in = use_vq_in
        self._number_embeddings = number_embeddings
        self._embedding_dim = embedding_dim

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._impl = TD3PlusBC_VQImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=self._actor_learning_rate,
            critic_learning_rate=self._critic_learning_rate,
            actor_optim_factory=self._actor_optim_factory,
            critic_optim_factory=self._critic_optim_factory,
            actor_encoder_factory=self._actor_encoder_factory,
            critic_encoder_factory=self._critic_encoder_factory,
            q_func_factory=self._q_func_factory,
            gamma=self._gamma,
            tau=self._tau,
            n_critics=self._n_critics,
            target_smoothing_sigma=self._target_smoothing_sigma,
            target_smoothing_clip=self._target_smoothing_clip,
            alpha=self._alpha,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
            action_scaler=self._action_scaler,
            reward_scaler=self._reward_scaler,
            env_name=self._env_name,
        )
        policy_args = dict(
            use_vq_in=self._use_vq_in,
            number_embeddings=self._number_embeddings,
            embedding_dim=self._embedding_dim
        )
        self._impl.build(policy_args)
        assert self.scaler._mean is not None and self.scaler._std is not None
        self._impl.init_range_of_norm_obs()

        if self._use_vq_in:
            self._impl._targ_policy.vq_input.disable_update_codebook()  # No need to update

    def _update(self, batch: TransitionMiniBatch) -> Dict[str, float]:
        """
        更新模型参数，包括Critic和Actor的优化步骤。
    
        Args:
            batch (TransitionMiniBatch): 包含采样的状态、动作、奖励、下一状态等数据的小批量样本。
    
        Returns:
            Dict[str, float]: 包含损失值和预测值的指标字典。
        
        Raises:
            ValueError: 如果返回的日志长度与预期不符。
        """
        # 确保实现已经初始化
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        # 确保编码字典（codebooks）在策略和目标策略之间保持同步
        assert (self._impl._policy.vq_input.codebooks == self._impl._targ_policy.vq_input.codebooks).all(), "Codebooks have not sync yet."
    
        metrics = {}
    
        # 如果使用向量量化输入，则禁用更新编码字典的能力
        if self._use_vq_in:
            self._impl._policy.vq_input.disable_update_codebook()
    
        # 更新Critic网络并获取总损失和额外日志
        critic_total_loss, extra_logs = self._impl.update_critic(batch)
        if len(extra_logs) == 4:
            q_target, current_q1, current_q2, critic_loss = extra_logs
        else:
            raise ValueError("Critic extra logs should contain exactly 4 elements.")
    
        # 将Critic相关的指标记录到metrics中
        metrics.update({
            "critic_total_loss": critic_total_loss,  # Critic总损失
            "critic_loss": critic_loss,              # Critic单独损失
            "q_target": q_target,                    # Q值的目标值
            "q1_prediction": current_q1,             # 第一个Critic网络的Q值预测
            "q2_prediction": current_q2,             # 第二个Critic网络的Q值预测
        })
    
        # 延迟策略更新，每隔固定步长更新一次Actor
        if self._grad_step % self._update_actor_interval == 0:
            # 如果使用向量量化输入，则重新启用更新编码字典的能力
            if self._use_vq_in:
                self._impl._policy.vq_input.enable_update_codebook()
    
            # 更新Actor网络并获取总损失和额外日志
            actor_total_loss, extra_logs = self._impl.update_actor(batch)
            if len(extra_logs) == 3:
                actor_loss, bc_loss, vq_loss = extra_logs
            else:
                raise ValueError("Actor extra logs should contain exactly 3 elements.")
    
            # 将Actor相关的指标记录到metrics中
            metrics.update({
                "actor_total_loss": actor_total_loss,  # Actor总损失
                "actor_loss": actor_loss,              # Actor单独损失
                "bc_loss": bc_loss,                    # 行为克隆损失（Behavior Cloning Loss）
                "vq_loss": vq_loss,                    # 向量量化损失（Vector Quantization Loss）
            })
    
            # 更新Critic目标网络
            self._impl.update_critic_target()
            # 更新Actor目标网络
            self._impl.update_actor_target()
    
            # 如果使用向量量化输入，则同步编码字典
            if self._use_vq_in:
                self._impl.sync_codebook_from_policy()  # 在软更新Critic/Actor后执行同步
    
        return metrics  # 返回所有记录的指标
    
    
    def get_action_type(self) -> ActionSpace:
        """
        获取动作空间类型。
    
        Returns:
            ActionSpace: 动作空间类型（此处为连续动作空间）。
        """
        return ActionSpace.CONTINUOUS

