import pytest

# 从 d3rlpy 库中导入必要的类
from d3rlpy.models.encoders import VectorEncoderFactory
from d3rlpy.models.q_functions import (
    FQFQFunctionFactory,
    IQNQFunctionFactory,
    MeanQFunctionFactory,
    QRQFunctionFactory,
    create_q_func_factory,
)
from d3rlpy.models.torch import (
    ContinuousFQFQFunction,
    ContinuousIQNQFunction,
    ContinuousMeanQFunction,
    ContinuousQRQFunction,
    DiscreteFQFQFunction,
    DiscreteIQNQFunction,
    DiscreteMeanQFunction,
    DiscreteQRQFunction,
)

# 辅助函数用于创建编码器
def _create_encoder(observation_shape, action_size):
    # 实例化一个 VectorEncoderFactory
    factory = VectorEncoderFactory()
    # 如果 action_size 为 None，创建一个用于连续动作的编码器
    if action_size is None:
        encoder = factory.create_with_action(observation_shape, 2)
    else:
        # 否则创建一个用于离散动作的编码器
        encoder = factory.create(observation_shape)
    return encoder

# 测试 MeanQFunctionFactory
@pytest.mark.parametrize("observation_shape", [(100,)])  # 使用形状为 100 的观察
@pytest.mark.parametrize("action_size", [None, 2])  # 测试连续和离散动作空间
def test_mean_q_function_factory(observation_shape, action_size):
    # 创建编码器
    encoder = _create_encoder(observation_shape, action_size)

    # 实例化 MeanQFunctionFactory
    factory = MeanQFunctionFactory()
    if action_size is None:
        # 创建并测试连续 Q 函数
        q_func = factory.create_continuous(encoder)
        assert isinstance(q_func, ContinuousMeanQFunction)
    else:
        # 创建并测试离散 Q 函数
        q_func = factory.create_discrete(encoder, action_size)
        assert isinstance(q_func, DiscreteMeanQFunction)

    # 检查工厂的类型和参数
    assert factory.get_type() == "mean"
    params = factory.get_params()
    new_factory = MeanQFunctionFactory(**params)
    assert new_factory.get_params() == params

# 测试 QRQFunctionFactory
@pytest.mark.parametrize("observation_shape", [(100,)])  # 使用形状为 100 的观察
@pytest.mark.parametrize("action_size", [None, 2])  # 测试连续和离散动作空间
def test_qr_q_function_factory(observation_shape, action_size):
    # 创建编码器
    encoder = _create_encoder(observation_shape, action_size)

    # 实例化 QRQFunctionFactory
    factory = QRQFunctionFactory()
    if action_size is None:
        # 创建并测试连续 Q 函数
        q_func = factory.create_continuous(encoder)
        assert isinstance(q_func, ContinuousQRQFunction)
    else:
        # 创建并测试离散 Q 函数
        q_func = factory.create_discrete(encoder, action_size)
        assert isinstance(q_func, DiscreteQRQFunction)

    # 检查工厂的类型和参数
    assert factory.get_type() == "qr"
    params = factory.get_params()
    new_factory = QRQFunctionFactory(**params)
    assert new_factory.get_params() == params

# 测试 IQNQFunctionFactory
@pytest.mark.parametrize("observation_shape", [(100,)])  # 使用形状为 100 的观察
@pytest.mark.parametrize("action_size", [None, 2])  # 测试连续和离散动作空间
def test_iqn_q_function_factory(observation_shape, action_size):
    # 创建编码器
    encoder = _create_encoder(observation_shape, action_size)

    # 实例化 IQNQFunctionFactory
    factory = IQNQFunctionFactory()
    if action_size is None:
        # 创建并测试连续 Q 函数
        q_func = factory.create_continuous(encoder)
        assert isinstance(q_func, ContinuousIQNQFunction)
    else:
        # 创建并测试离散 Q 函数
        q_func = factory.create_discrete(encoder, action_size)
        assert isinstance(q_func, DiscreteIQNQFunction)

    # 检查工厂的类型和参数
    assert factory.get_type() == "iqn"
    params = factory.get_params()
    new_factory = IQNQFunctionFactory(**params)
    assert new_factory.get_params() == params

# 测试 FQFQFunctionFactory
@pytest.mark.parametrize("observation_shape", [(100,)])  # 使用形状为 100 的观察
@pytest.mark.parametrize("action_size", [None, 2])  # 测试连续和离散动作空间
def test_fqf_q_function_factory(observation_shape, action_size):
    # 创建编码器
    encoder = _create_encoder(observation_shape, action_size)

    # 实例化 FQFQFunctionFactory
    factory = FQFQFunctionFactory()
    if action_size is None:
        # 创建并测试连续 Q 函数
        q_func = factory.create_continuous(encoder)
        assert isinstance(q_func, ContinuousFQFQFunction)
    else:
        # 创建并测试离散 Q 函数
        q_func = factory.create_discrete(encoder, action_size)
        assert isinstance(q_func, DiscreteFQFQFunction)

    # 检查工厂的类型和参数
    assert factory.get_type() == "fqf"
    params = factory.get_params()
    new_factory = FQFQFunctionFactory(**params)
    assert new_factory.get_params() == params

# 测试工厂创建函数
@pytest.mark.parametrize("name", ["mean", "qr", "iqn", "fqf"])  # 使用不同工厂名称进行测试
def test_create_q_func_factory(name):
    # 使用工厂创建函数创建工厂
    factory = create_q_func_factory(name)
    # 检查是否创建了正确类型的工厂
    if name == "mean":
        assert isinstance(factory, MeanQFunctionFactory)
    elif name == "qr":
        assert isinstance(factory, QRQFunctionFactory)
    elif name == "iqn":
        assert isinstance(factory, IQNQFunctionFactory)
    elif name == "fqf":
        assert isinstance(factory, FQFQFunctionFactory)
