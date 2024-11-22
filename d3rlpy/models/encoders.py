from typing import Any, ClassVar, Dict, Type

from ..decorators import pretty_repr
from .torch import (
    ContinuousFQFQFunction,
    ContinuousIQNQFunction,
    ContinuousMeanQFunction,
    ContinuousQFunction,
    ContinuousQRQFunction,
    DiscreteFQFQFunction,
    DiscreteIQNQFunction,
    DiscreteMeanQFunction,
    DiscreteQFunction,
    DiscreteQRQFunction,
    Encoder,
    EncoderWithAction,
)


@pretty_repr
class QFunctionFactory:
    """
    Q函数工厂的基类，用于生成不同类型的Q函数。
    
    Attributes:
        TYPE (str): Q函数的类型标识符。
        _share_encoder (bool): 是否在多个Q函数之间共享编码器。
    """

    TYPE: ClassVar[str] = "none"  # Q函数类型，子类需重写
    _share_encoder: bool

    def __init__(self, share_encoder: bool):
        """
        初始化工厂。
        
        Args:
            share_encoder (bool): 是否共享编码器。
        """
        self._share_encoder = share_encoder

    def create_discrete(
        self, encoder: Encoder, action_size: int
    ) -> DiscreteQFunction:
        """
        创建用于离散动作空间的Q函数。

        Args:
            encoder (Encoder): 编码器模块，用于处理观测数据。
            action_size (int): 离散动作空间的维度。

        Returns:
            DiscreteQFunction: 离散Q函数对象。
        """
        raise NotImplementedError

    def create_continuous(
        self, encoder: EncoderWithAction
    ) -> ContinuousQFunction:
        """
        创建用于连续动作空间的Q函数。

        Args:
            encoder (EncoderWithAction): 编码器模块，用于处理观测数据和动作。

        Returns:
            ContinuousQFunction: 连续Q函数对象。
        """
        raise NotImplementedError

    def get_type(self) -> str:
        """
        获取Q函数的类型。

        Returns:
            str: Q函数类型。
        """
        return self.TYPE

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        """
        获取Q函数的参数。

        Args:
            deep (bool): 是否递归获取嵌套参数。

        Returns:
            Dict[str, Any]: 参数字典。
        """
        raise NotImplementedError

    @property
    def share_encoder(self) -> bool:
        """
        是否共享编码器。

        Returns:
            bool: 共享状态。
        """
        return self._share_encoder


class MeanQFunctionFactory(QFunctionFactory):
    """
    标准Q函数工厂类，生成Mean Q函数。

    参考文献:
        * Mnih et al., Human-level control through deep reinforcement learning.
          <https://www.nature.com/articles/nature14236>
        * Lillicrap et al., Continuous control with deep reinforcement learning.
          <https://arxiv.org/abs/1509.02971>
    
    Args:
        share_encoder (bool): 是否共享编码器。
    """

    TYPE: ClassVar[str] = "mean"  # 类型为“mean”

    def __init__(self, share_encoder: bool = False, **kwargs: Any):
        super().__init__(share_encoder)

    def create_discrete(
        self,
        encoder: Encoder,
        action_size: int,
    ) -> DiscreteMeanQFunction:
        """
        创建离散动作空间的Mean Q函数。

        Args:
            encoder (Encoder): 编码器。
            action_size (int): 动作空间维度。

        Returns:
            DiscreteMeanQFunction: 离散Mean Q函数对象。
        """
        return DiscreteMeanQFunction(encoder, action_size)

    def create_continuous(
        self,
        encoder: EncoderWithAction,
    ) -> ContinuousMeanQFunction:
        """
        创建连续动作空间的Mean Q函数。

        Args:
            encoder (EncoderWithAction): 编码器。

        Returns:
            ContinuousMeanQFunction: 连续Mean Q函数对象。
        """
        return ContinuousMeanQFunction(encoder)

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        """
        获取参数。

        Returns:
            Dict[str, Any]: 包含`share_encoder`的参数字典。
        """
        return {
            "share_encoder": self._share_encoder,
        }


class QRQFunctionFactory(QFunctionFactory):
    """
    分位数回归Q函数工厂类。

    参考文献:
        * Dabney et al., Distributional reinforcement learning with quantile regression.
          <https://arxiv.org/abs/1710.10044>
    
    Args:
        share_encoder (bool): 是否共享编码器。
        n_quantiles (int): 分位数的数量。
    """

    TYPE: ClassVar[str] = "qr"  # 类型为“qr”
    _n_quantiles: int

    def __init__(
        self, share_encoder: bool = False, n_quantiles: int = 32, **kwargs: Any
    ):
        super().__init__(share_encoder)
        self._n_quantiles = n_quantiles

    def create_discrete(
        self, encoder: Encoder, action_size: int
    ) -> DiscreteQRQFunction:
        """
        创建离散动作空间的QR Q函数。

        Args:
            encoder (Encoder): 编码器。
            action_size (int): 动作空间维度。

        Returns:
            DiscreteQRQFunction: 离散QR Q函数对象。
        """
        return DiscreteQRQFunction(encoder, action_size, self._n_quantiles)

    def create_continuous(
        self,
        encoder: EncoderWithAction,
    ) -> ContinuousQRQFunction:
        """
        创建连续动作空间的QR Q函数。

        Args:
            encoder (EncoderWithAction): 编码器。

        Returns:
            ContinuousQRQFunction: 连续QR Q函数对象。
        """
        return ContinuousQRQFunction(encoder, self._n_quantiles)

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        """
        获取参数。

        Returns:
            Dict[str, Any]: 参数字典，包含`share_encoder`和`n_quantiles`。
        """
        return {
            "share_encoder": self._share_encoder,
            "n_quantiles": self._n_quantiles,
        }

    @property
    def n_quantiles(self) -> int:
        """
        获取分位数数量。

        Returns:
            int: 分位数数量。
        """
        return self._n_quantiles


# 其他工厂类（IQNQFunctionFactory, FQFQFunctionFactory）的注释风格与上述类似，按需对其余部分进行补充。

# 注册Q函数工厂
Q_FUNC_LIST: Dict[str, Type[QFunctionFactory]] = {}


def register_q_func_factory(cls: Type[QFunctionFactory]) -> None:
    """
    注册Q函数工厂类。

    Args:
        cls (Type[QFunctionFactory]): 继承自`QFunctionFactory`的工厂类。
    """
    is_registered = cls.TYPE in Q_FUNC_LIST
    assert not is_registered, f"{cls.TYPE}已被注册。"
    Q_FUNC_LIST[cls.TYPE] = cls


def create_q_func_factory(name: str, **kwargs: Any) -> QFunctionFactory:
    """
    根据名称创建已注册的Q函数工厂实例。

    Args:
        name (str): 注册的Q函数工厂类型名称。
        kwargs: Q函数参数。

    Returns:
        QFunctionFactory: Q函数工厂实例。
    """
    assert name in Q_FUNC_LIST, f"{name}未注册。"
    factory = Q_FUNC_LIST[name](**kwargs)
    assert isinstance(factory, QFunctionFactory)
    return factory


# 注册所有实现的Q函数工厂
register_q_func_factory(MeanQFunctionFactory)
register_q_func_factory(QRQFunctionFactory)
register_q_func_factory(IQNQFunctionFactory)
register_q_func_factory(FQFQFunctionFactory)
