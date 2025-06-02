"""
数值稳定性管理模块

为NSGA-II海上通信优化系统提供全面的数值稳定性保护
包含单例模式的管理器访问接口

@author: Improved NSGA-II Team
@date: 2024
"""

import numpy as np
import logging
from typing import Union, Any, List, Dict, Tuple, Optional
import warnings
import threading

logger = logging.getLogger(__name__)

class NumericalStabilityManager:
    """
    数值稳定性管理器
    提供各种数值计算的稳定性保护措施，专门为NSGA-II优化算法设计
    """
    
    def __init__(self):
        # 数值常数
        self.EPSILON = 1e-12              # 防止除零的小值
        self.MAX_FLOAT = 1e15             # 最大浮点数
        self.MIN_FLOAT = -1e15            # 最小浮点数
        self.MAX_EXP_ARG = 700            # exp函数的最大参数
        self.MIN_EXP_ARG = -700           # exp函数的最小参数
        self.MAX_LOG_ARG = 1e15           # log函数的最大参数
        self.MIN_LOG_ARG = 1e-15          # log函数的最小参数
        
        # 目标函数值范围 - 针对通信优化问题
        self.OBJECTIVE_MIN = -1000.0      # 目标函数最小值
        self.OBJECTIVE_MAX = 1000.0       # 目标函数最大值
        self.RELIABILITY_MIN = 0.0        # 可靠性最小值
        self.RELIABILITY_MAX = 1.0        # 可靠性最大值
        self.SNR_MIN = -20.0              # SNR最小值(dB)
        self.SNR_MAX = 50.0               # SNR最大值(dB)
        
        # 约束值范围
        self.CONSTRAINT_MIN = -100.0      # 约束最小值
        self.CONSTRAINT_MAX = 100.0       # 约束最大值
        
        # 频率相关参数
        self.FREQ_MIN = 100e6             # 最小频率
        self.FREQ_MAX = 10e9              # 最大频率
        self.POWER_MIN = 1.0              # 最小功率
        self.POWER_MAX = 1000.0           # 最大功率
        self.BW_MIN = 1e6                 # 最小带宽
        self.BW_MAX = 100e6               # 最大带宽
        
        # 统计信息
        self.correction_count = 0
        self.warning_count = 0
        self.is_initialized = True
        
        logger.debug("数值稳定性管理器初始化完成")

    def safe_divide(self, numerator: float, denominator: float, 
                   default: float = 0.0) -> float:
        """
        安全除法，防止除零错误
        
        参数:
        numerator: 分子
        denominator: 分母
        default: 除零时的默认值
        
        返回:
        安全的除法结果
        """
        try:
            if abs(denominator) < self.EPSILON:
                self.warning_count += 1
                logger.debug(f"检测到除零风险：{numerator}/{denominator}，返回默认值 {default}")
                return default
            
            result = numerator / denominator
            return self.clamp_float(result)
            
        except Exception as e:
            self.warning_count += 1
            logger.warning(f"除法计算异常：{str(e)}，返回默认值")
            return default

    def safe_log(self, x: float, base: float = np.e, default: float = 0.0) -> float:
        """
        安全对数计算
        
        参数:
        x: 输入值
        base: 底数
        default: 异常时的默认值
        
        返回:
        安全的对数结果
        """
        try:
            if x <= 0:
                self.warning_count += 1
                logger.debug(f"对数参数非正数：{x}，返回默认值 {default}")
                return default
                
            # 限制输入范围
            x = max(self.MIN_LOG_ARG, min(self.MAX_LOG_ARG, x))
            
            if base == np.e:
                result = np.log(x)
            else:
                result = np.log(x) / np.log(base)
                
            return self.clamp_float(result)
            
        except Exception as e:
            self.warning_count += 1
            logger.warning(f"对数计算异常：{str(e)}，返回默认值")
            return default

    def safe_log2(self, x: float, default: float = 0.0) -> float:
        """Shannon容量计算中的安全log2"""
        return self.safe_log(x, 2.0, default)

    def safe_log10(self, x: float, default: float = 0.0) -> float:
        """dB计算中的安全log10"""
        return self.safe_log(x, 10.0, default)

    def safe_exp(self, x: float, default: float = 1.0) -> float:
        """
        安全指数计算
        
        参数:
        x: 指数
        default: 异常时的默认值
        
        返回:
        安全的指数结果
        """
        try:
            # 限制指数范围防止溢出
            x = max(self.MIN_EXP_ARG, min(self.MAX_EXP_ARG, x))
            
            result = np.exp(x)
            return self.clamp_float(result)
            
        except Exception as e:
            self.warning_count += 1
            logger.warning(f"指数计算异常：{str(e)}，返回默认值")
            return default

    def safe_power(self, base: float, exponent: float, default: float = 1.0) -> float:
        """
        安全幂运算
        
        参数:
        base: 底数
        exponent: 指数
        default: 异常时的默认值
        
        返回:
        安全的幂运算结果
        """
        try:
            if base < 0 and not self._is_integer(exponent):
                self.warning_count += 1
                logger.debug(f"负数的非整数次幂：{base}^{exponent}，返回默认值")
                return default
                
            if base == 0 and exponent < 0:
                self.warning_count += 1
                logger.debug(f"零的负数次幂：{base}^{exponent}，返回默认值")
                return default
                
            # 限制结果范围
            if abs(base) > 1 and abs(exponent) > 100:
                self.warning_count += 1
                logger.debug(f"幂运算可能溢出：{base}^{exponent}，使用限制值")
                result = np.sign(base) * self.MAX_FLOAT if base != 0 else 0
            else:
                result = np.power(base, exponent)
                
            return self.clamp_float(result)
            
        except Exception as e:
            self.warning_count += 1
            logger.warning(f"幂运算异常：{str(e)}，返回默认值")
            return default

    def safe_sqrt(self, x: float, default: float = 0.0) -> float:
        """
        安全平方根计算
        
        参数:
        x: 输入值
        default: 异常时的默认值
        
        返回:
        安全的平方根结果
        """
        try:
            if x < 0:
                self.warning_count += 1
                logger.debug(f"负数开方：sqrt({x})，返回默认值")
                return default
                
            result = np.sqrt(x)
            return self.clamp_float(result)
            
        except Exception as e:
            self.warning_count += 1
            logger.warning(f"平方根计算异常：{str(e)}，返回默认值")
            return default

    def clamp_float(self, value: float, min_val: Optional[float] = None, 
                   max_val: Optional[float] = None) -> float:
        """
        限制浮点数范围
        
        参数:
        value: 输入值
        min_val: 最小值（可选）
        max_val: 最大值（可选）
        
        返回:
        限制后的值
        """
        try:
            if np.isnan(value):
                self.correction_count += 1
                logger.debug("检测到NaN值，返回0")
                return 0.0
                
            if np.isinf(value):
                self.correction_count += 1
                logger.debug(f"检测到无穷值：{value}")
                return self.MAX_FLOAT if value > 0 else self.MIN_FLOAT
                
            min_bound = min_val if min_val is not None else self.MIN_FLOAT
            max_bound = max_val if max_val is not None else self.MAX_FLOAT
            
            clamped = max(min_bound, min(max_bound, value))
            if abs(clamped - value) > self.EPSILON:
                self.correction_count += 1
                
            return clamped
            
        except Exception as e:
            self.correction_count += 1
            logger.warning(f"数值限制异常：{str(e)}，返回0")
            return 0.0

    def validate_objective_value(self, value: float, obj_type: str = "general") -> float:
        """
        验证和修正目标函数值
        
        参数:
        value: 目标函数值
        obj_type: 目标类型（reliability, spectral, energy, interference, adaptability）
        
        返回:
        验证后的目标函数值
        """
        if np.isnan(value) or np.isinf(value):
            self.correction_count += 1
            logger.warning(f"{obj_type}目标函数值异常：{value}，使用默认值")
            
            # 根据目标类型返回合适的默认值
            defaults = {
                "reliability": -1.0,      # 可靠性目标（最小化负值）
                "spectral": -1.0,         # 频谱效率目标
                "energy": 100.0,          # 能量效率目标（最小化）
                "interference": -1.0,     # 抗干扰目标
                "adaptability": -1.0      # 适应性目标
            }
            return defaults.get(obj_type, self.OBJECTIVE_MAX)
            
        return self.clamp_float(value, self.OBJECTIVE_MIN, self.OBJECTIVE_MAX)

    def validate_constraint_value(self, value: float) -> float:
        """
        验证和修正约束值
        
        参数:
        value: 约束值
        
        返回:
        验证后的约束值
        """
        if np.isnan(value) or np.isinf(value):
            self.correction_count += 1
            logger.warning(f"约束值异常：{value}，使用违反约束的值")
            return 1.0  # 违反约束
            
        return self.clamp_float(value, self.CONSTRAINT_MIN, self.CONSTRAINT_MAX)

    def validate_parameter_vector(self, params: np.ndarray, 
                                bounds: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        验证和修正参数向量
        
        参数:
        params: 参数向量
        bounds: (下界, 上界)
        
        返回:
        验证后的参数向量
        """
        try:
            lower_bounds, upper_bounds = bounds
            
            # 确保输入是numpy数组
            if not isinstance(params, np.ndarray):
                params = np.array(params)
            
            # 检查异常值
            valid_mask = np.isfinite(params)
            if not np.all(valid_mask):
                self.correction_count += 1
                logger.warning(f"参数向量包含异常值，位置：{np.where(~valid_mask)[0]}")
                params = params.copy()
                
                # 用边界中点替换异常值
                invalid_indices = np.where(~valid_mask)[0]
                for idx in invalid_indices:
                    if idx < len(lower_bounds) and idx < len(upper_bounds):
                        params[idx] = (lower_bounds[idx] + upper_bounds[idx]) / 2
                    else:
                        params[idx] = 0.0
            
            # 应用边界约束
            params = np.clip(params, lower_bounds, upper_bounds)
            
            return params
            
        except Exception as e:
            self.correction_count += 1
            logger.error(f"参数向量验证出错：{e}")
            # 返回边界中点作为安全值
            try:
                return (lower_bounds + upper_bounds) / 2
            except:
                return np.zeros_like(params) if isinstance(params, np.ndarray) else np.array([0.0])

    def validate_snr(self, snr: float) -> float:
        """验证SNR值"""
        if np.isnan(snr) or np.isinf(snr):
            self.correction_count += 1
            return 0.0  # 默认SNR
        return self.clamp_float(snr, self.SNR_MIN, self.SNR_MAX)

    def validate_frequency(self, freq: float) -> float:
        """验证频率值"""
        if freq <= 0 or np.isnan(freq) or np.isinf(freq):
            self.correction_count += 1
            return (self.FREQ_MIN + self.FREQ_MAX) / 2
        return self.clamp_float(freq, self.FREQ_MIN, self.FREQ_MAX)

    def validate_power(self, power: float) -> float:
        """验证功率值"""
        if power <= 0 or np.isnan(power) or np.isinf(power):
            self.correction_count += 1
            return (self.POWER_MIN + self.POWER_MAX) / 2
        return self.clamp_float(power, self.POWER_MIN, self.POWER_MAX)

    def validate_bandwidth(self, bw: float) -> float:
        """验证带宽值"""
        if bw <= 0 or np.isnan(bw) or np.isinf(bw):
            self.correction_count += 1
            return (self.BW_MIN + self.BW_MAX) / 2
        return self.clamp_float(bw, self.BW_MIN, self.BW_MAX)

    def validate_reliability(self, reliability: float) -> float:
        """验证可靠性值"""
        if np.isnan(reliability) or np.isinf(reliability):
            self.correction_count += 1
            return 0.5  # 中等可靠性
        return self.clamp_float(reliability, self.RELIABILITY_MIN, self.RELIABILITY_MAX)

    def safe_ber_calculation(self, snr_db: float, modulation: str = "BPSK") -> float:
        """
        安全的误码率计算
        
        参数:
        snr_db: 信噪比(dB)
        modulation: 调制方式
        
        返回:
        误码率
        """
        try:
            # 验证SNR
            snr_db = self.validate_snr(snr_db)
            
            # 转换到线性值
            snr_linear = self.safe_power(10, snr_db / 10.0, 1.0)
            
            # 根据调制方式计算BER
            if modulation == 'BPSK':
                # BER = 0.5 * exp(-snr_linear/2)
                arg = -snr_linear / 2.0
                ber = 0.5 * self.safe_exp(arg, 0.5)
            elif modulation == 'QPSK':
                arg = -snr_linear / 4.0
                ber = 0.5 * self.safe_exp(arg, 0.5)
            elif modulation == 'QAM16':
                arg = -snr_linear / 10.0
                ber = 0.2 * self.safe_exp(arg, 0.2)
            elif modulation == 'QAM64':
                arg = -snr_linear / 20.0
                ber = 0.1 * self.safe_exp(arg, 0.1)
            else:
                ber = 0.1  # 默认BER
            
            # 限制BER在合理范围内
            return self.clamp_float(ber, 1e-10, 0.5)
            
        except Exception as e:
            self.warning_count += 1
            logger.warning(f"BER计算异常：{str(e)}")
            return 0.1  # 保守的BER值

    def safe_shannon_capacity(self, snr_linear: float, bandwidth: float) -> float:
        """
        安全的Shannon容量计算
        
        参数:
        snr_linear: 线性SNR
        bandwidth: 带宽
        
        返回:
        信道容量
        """
        try:
            # 验证输入
            snr_linear = max(0.1, snr_linear)  # 避免零或负值
            bandwidth = self.validate_bandwidth(bandwidth)
            
            # C = B * log2(1 + SNR)
            capacity = bandwidth * self.safe_log2(1 + snr_linear, 0.0)
            
            return self.clamp_float(capacity, 0.0, bandwidth * 20)  # 理论上限约束
            
        except Exception as e:
            self.warning_count += 1
            logger.warning(f"Shannon容量计算异常：{str(e)}")
            return bandwidth  # 返回带宽作为保守估计

    def safe_path_loss_calculation(self, freq: float, distance: float) -> float:
        """
        安全的路径损耗计算
        
        参数:
        freq: 频率(Hz)
        distance: 距离(km)
        
        返回:
        路径损耗(dB)
        """
        try:
            freq = self.validate_frequency(freq)
            distance = max(0.1, distance)  # 最小距离
            
            # 自由空间路径损耗: 20*log10(4*pi*d*f/c)
            c = 3e8  # 光速
            
            term1 = 20 * self.safe_log10(4 * np.pi)
            term2 = 20 * self.safe_log10(distance * 1000)  # 转换为米
            term3 = 20 * self.safe_log10(freq)
            term4 = -20 * self.safe_log10(c)
            
            path_loss = term1 + term2 + term3 + term4
            
            return self.clamp_float(path_loss, 30.0, 200.0)  # 合理的路径损耗范围
            
        except Exception as e:
            self.warning_count += 1
            logger.warning(f"路径损耗计算异常：{str(e)}")
            return 100.0  # 默认路径损耗

    def validate_array(self, arr: np.ndarray, default_value: float = 0.0,
                      min_val: Optional[float] = None, max_val: Optional[float] = None) -> np.ndarray:
        """
        验证和修正数组中的所有元素
        
        参数:
        arr: 输入数组
        default_value: NaN/Inf时的默认值
        min_val: 最小值限制
        max_val: 最大值限制
        
        返回:
        验证后的数组
        """
        try:
            if not isinstance(arr, np.ndarray):
                arr = np.array(arr)
            
            # 创建输出数组
            validated_arr = arr.copy()
            
            # 检查和修正NaN值
            nan_mask = np.isnan(validated_arr)
            if np.any(nan_mask):
                self.correction_count += np.sum(nan_mask)
                validated_arr[nan_mask] = default_value
                logger.debug(f"数组中发现 {np.sum(nan_mask)} 个NaN值，已替换为 {default_value}")
            
            # 检查和修正无穷值
            inf_mask = np.isinf(validated_arr)
            if np.any(inf_mask):
                self.correction_count += np.sum(inf_mask)
                pos_inf_mask = validated_arr == np.inf
                neg_inf_mask = validated_arr == -np.inf
                
                # 处理正无穷
                if np.any(pos_inf_mask):
                    validated_arr[pos_inf_mask] = max_val if max_val is not None else self.MAX_FLOAT
                
                # 处理负无穷
                if np.any(neg_inf_mask):
                    validated_arr[neg_inf_mask] = min_val if min_val is not None else self.MIN_FLOAT
                
                logger.debug(f"数组中发现 {np.sum(inf_mask)} 个无穷值，已修正")
            
            # 应用范围限制
            if min_val is not None or max_val is not None:
                min_bound = min_val if min_val is not None else self.MIN_FLOAT
                max_bound = max_val if max_val is not None else self.MAX_FLOAT
                
                # 检查超出范围的值
                out_of_range_mask = (validated_arr < min_bound) | (validated_arr > max_bound)
                if np.any(out_of_range_mask):
                    self.correction_count += np.sum(out_of_range_mask)
                    validated_arr = np.clip(validated_arr, min_bound, max_bound)
                    logger.debug(f"数组中 {np.sum(out_of_range_mask)} 个值超出范围，已限制")
            
            return validated_arr
            
        except Exception as e:
            self.correction_count += 1
            logger.error(f"数组验证异常：{str(e)}")
            # 返回安全的默认数组
            if isinstance(arr, np.ndarray):
                return np.full_like(arr, default_value, dtype=float)
            else:
                return np.array([default_value])

    def validate_objective_array(self, objectives: np.ndarray) -> np.ndarray:
        """
        验证目标函数值数组
        
        参数:
        objectives: 目标函数值数组 (n_solutions, n_objectives)
        
        返回:
        验证后的目标函数值数组
        """
        try:
            if not isinstance(objectives, np.ndarray):
                objectives = np.array(objectives)
            
            # 确保是2D数组
            if objectives.ndim == 1:
                objectives = objectives.reshape(1, -1)
            
            validated_objectives = np.zeros_like(objectives)
            obj_types = ["reliability", "spectral", "energy", "interference", "adaptability"]
            
            # 逐列验证，每列对应一个目标函数
            for col in range(objectives.shape[1]):
                obj_type = obj_types[col] if col < len(obj_types) else "general"
                
                for row in range(objectives.shape[0]):
                    validated_objectives[row, col] = self.validate_objective_value(
                        objectives[row, col], obj_type)
            
            return validated_objectives
            
        except Exception as e:
            self.correction_count += 1
            logger.error(f"目标函数数组验证异常：{str(e)}")
            # 返回默认的安全值
            if objectives.ndim == 2:
                n_solutions, n_objectives = objectives.shape
                default_values = [-1.0, -1.0, 100.0, -1.0, -1.0]
                safe_objectives = np.tile(default_values[:n_objectives], (n_solutions, 1))
                return safe_objectives
            else:
                return np.array([[-1.0, -1.0, 100.0, -1.0, -1.0]])

    def validate_constraint_array(self, constraints: np.ndarray) -> np.ndarray:
        """
        验证约束值数组
        
        参数:
        constraints: 约束值数组
        
        返回:
        验证后的约束值数组
        """
        try:
            # 使用通用数组验证方法
            validated_constraints = self.validate_array(
                constraints, 
                default_value=1.0,  # 违反约束的默认值
                min_val=self.CONSTRAINT_MIN,
                max_val=self.CONSTRAINT_MAX
            )
            
            return validated_constraints
            
        except Exception as e:
            self.correction_count += 1
            logger.error(f"约束数组验证异常：{str(e)}")
            # 返回表示违反约束的安全值
            if isinstance(constraints, np.ndarray):
                return np.ones_like(constraints)
            else:
                return np.array([1.0])

    def validate_population_array(self, population: np.ndarray, 
                                 bounds: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        验证种群数组（批量验证参数向量）
        
        参数:
        population: 种群数组 (n_individuals, n_variables)
        bounds: (下界, 上界)
        
        返回:
        验证后的种群数组
        """
        try:
            lower_bounds, upper_bounds = bounds
            
            if not isinstance(population, np.ndarray):
                population = np.array(population)
            
            # 确保是2D数组
            if population.ndim == 1:
                population = population.reshape(1, -1)
            
            validated_population = np.zeros_like(population)
            
            # 逐行验证每个个体
            for i in range(population.shape[0]):
                validated_population[i] = self.validate_parameter_vector(
                    population[i], (lower_bounds, upper_bounds))
            
            return validated_population
            
        except Exception as e:
            self.correction_count += 1
            logger.error(f"种群数组验证异常：{str(e)}")
            # 返回边界中点作为安全值
            try:
                n_individuals = population.shape[0] if population.ndim == 2 else 1
                n_variables = len(lower_bounds)
                safe_population = np.tile((lower_bounds + upper_bounds) / 2, (n_individuals, 1))
                return safe_population
            except:
                return np.array([[0.0]])

    def validate_optimization_results(self, objectives: np.ndarray, 
                                    constraints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        验证优化结果
        
        参数:
        objectives: 目标函数值数组
        constraints: 约束值数组
        
        返回:
        验证后的(目标函数值, 约束值)
        """
        try:
            # 使用专门的数组验证方法
            validated_objectives = self.validate_objective_array(objectives)
            validated_constraints = self.validate_constraint_array(constraints)
            
            return validated_objectives, validated_constraints
            
        except Exception as e:
            self.correction_count += 1
            logger.error(f"优化结果验证异常：{str(e)}")
            # 返回默认的安全值
            safe_objectives = np.array([[-1.0, -1.0, 100.0, -1.0, -1.0]])
            safe_constraints = np.array([[0.0]])
            return safe_objectives, safe_constraints

    def get_statistics(self) -> Dict[str, int]:
        """获取数值稳定性统计信息"""
        return {
            "corrections_made": self.correction_count,
            "warnings_issued": self.warning_count
        }

    def reset_statistics(self):
        """重置统计信息"""
        self.correction_count = 0
        self.warning_count = 0

    def _is_integer(self, value: float, tolerance: float = 1e-9) -> bool:
        """检查浮点数是否接近整数"""
        return abs(value - round(value)) < tolerance

    def enable_warnings(self, enable: bool = True):
        """启用或禁用numpy警告"""
        if enable:
            warnings.resetwarnings()
        else:
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            warnings.filterwarnings('ignore', category=UserWarning)

    def __enter__(self):
        """上下文管理器入口"""
        self.reset_statistics()
        self.enable_warnings(False)  # 禁用警告以避免日志噪音
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        stats = self.get_statistics()
        if stats["corrections_made"] > 0 or stats["warnings_issued"] > 0:
            logger.info(f"数值稳定性管理器统计: 修正{stats['corrections_made']}次，"
                       f"警告{stats['warnings_issued']}次")
        self.enable_warnings(True)  # 恢复警告


# 单例模式实现
_stability_manager_instance = None
_lock = threading.Lock()


def get_stability_manager() -> NumericalStabilityManager:
    """
    获取数值稳定性管理器实例（单例模式）
    
    返回:
    NumericalStabilityManager: 管理器实例
    """
    global _stability_manager_instance
    
    if _stability_manager_instance is None:
        with _lock:
            # 双重检查锁定
            if _stability_manager_instance is None:
                _stability_manager_instance = NumericalStabilityManager()
                logger.info("创建数值稳定性管理器单例实例")
    
    return _stability_manager_instance


def reset_stability_manager():
    """
    重置数值稳定性管理器实例
    主要用于测试或重新初始化
    """
    global _stability_manager_instance
    
    with _lock:
        if _stability_manager_instance is not None:
            logger.info("重置数值稳定性管理器实例")
        _stability_manager_instance = None

class SafeCalculator:
    """
    安全计算器类
    
    为目标函数计算提供数值稳定的计算方法
    专门为 objectives_improved 模块设计
    """
    
    def __init__(self, stability_manager=None):
        """初始化安全计算器"""
        self.stability_manager = stability_manager or get_stability_manager()
        
        # 通信系统相关常数
        self.LIGHT_SPEED = 3e8  # 光速 (m/s)
        self.BOLTZMANN_CONSTANT = 1.38e-23  # 玻尔兹曼常数 (J/K)
        self.TEMPERATURE = 290  # 标准温度 (K)
        
        # 计算范围限制
        self.MIN_SNR_DB = -30.0
        self.MAX_SNR_DB = 60.0
        self.MIN_FREQUENCY = 100e6  # 100 MHz
        self.MAX_FREQUENCY = 10e9   # 10 GHz
        self.MIN_POWER = 0.1        # 0.1 W
        self.MAX_POWER = 1000.0     # 1000 W
        self.MIN_BANDWIDTH = 1e3    # 1 kHz
        self.MAX_BANDWIDTH = 1e9    # 1 GHz
        
        logger.debug("SafeCalculator 初始化完成")

    def safe_snr_calculation(self, signal_power: float, noise_power: float, 
                           return_db: bool = True) -> float:
        """
        安全的信噪比计算
        
        参数:
        signal_power: 信号功率 (W)
        noise_power: 噪声功率 (W)
        return_db: 是否返回dB值
        
        返回:
        信噪比 (线性值或dB)
        """
        try:
            # 验证输入
            signal_power = self.stability_manager.validate_power(signal_power)
            noise_power = self.stability_manager.validate_power(noise_power)
            
            # 计算线性SNR
            snr_linear = self.stability_manager.safe_divide(signal_power, noise_power, 0.001)
            
            if return_db:
                # 转换为dB
                snr_db = 10 * self.stability_manager.safe_log10(snr_linear, 0.001)
                return self.stability_manager.clamp_float(snr_db, self.MIN_SNR_DB, self.MAX_SNR_DB)
            else:
                return snr_linear
                
        except Exception as e:
            logger.warning(f"SNR计算异常：{str(e)}")
            return 0.0 if not return_db else -20.0

    def safe_capacity_calculation(self, snr_linear: float, bandwidth: float) -> float:
        """
        安全的Shannon容量计算
        
        参数:
        snr_linear: 线性信噪比
        bandwidth: 带宽 (Hz)
        
        返回:
        信道容量 (bps)
        """
        try:
            # 验证输入
            snr_linear = max(0.001, snr_linear)
            bandwidth = self.stability_manager.validate_bandwidth(bandwidth)
            
            # Shannon公式: C = B * log2(1 + SNR)
            capacity = bandwidth * self.stability_manager.safe_log2(1 + snr_linear, 0.1)
            
            # 限制在合理范围内
            max_capacity = bandwidth * 20  # 假设最大频谱效率为20 bps/Hz
            return self.stability_manager.clamp_float(capacity, 0.0, max_capacity)
            
        except Exception as e:
            logger.warning(f"容量计算异常：{str(e)}")
            return bandwidth

    def safe_ber_calculation(self, snr_db: float, modulation_type: str) -> float:
        """
        安全的误码率计算
        
        参数:
        snr_db: 信噪比 (dB)
        modulation_type: 调制类型
        
        返回:
        误码率
        """
        try:
            # 验证SNR
            snr_db = self.stability_manager.validate_snr(snr_db)
            snr_linear = self.stability_manager.safe_power(10, snr_db / 10.0, 0.001)
            
            # 根据调制类型计算BER
            if modulation_type.upper() == 'BPSK':
                arg = -snr_linear
                ber = 0.5 * self.stability_manager.safe_exp(arg, 0.5)
            elif modulation_type.upper() == 'QPSK':
                arg = -snr_linear / 2.0
                ber = 0.5 * self.stability_manager.safe_exp(arg, 0.5)
            elif modulation_type.upper() in ['QAM16', '16QAM']:
                arg = -snr_linear / 5.0
                ber = 0.2 * self.stability_manager.safe_exp(arg, 0.2)
            elif modulation_type.upper() in ['QAM64', '64QAM']:
                arg = -snr_linear / 10.0
                ber = 0.1 * self.stability_manager.safe_exp(arg, 0.1)
            else:
                # 默认使用BPSK
                arg = -snr_linear
                ber = 0.5 * self.stability_manager.safe_exp(arg, 0.5)
            
            # 限制BER在合理范围内
            return self.stability_manager.clamp_float(ber, 1e-12, 0.5)
            
        except Exception as e:
            logger.warning(f"BER计算异常：{str(e)}")
            return 0.1

    def safe_reliability_calculation(self, ber: float, packet_size_bits: int = 1024) -> float:
        """
        安全的可靠性计算
        
        参数:
        ber: 误码率
        packet_size_bits: 数据包大小 (比特)
        
        返回:
        可靠性 (包成功率)
        """
        try:
            # 验证输入
            ber = self.stability_manager.clamp_float(ber, 1e-12, 0.5)
            packet_size_bits = max(1, min(65536, packet_size_bits))
            
            # 计算包成功率: P_success = (1 - BER)^packet_size
            if ber < 1e-10:
                reliability = 1.0 - packet_size_bits * ber
            else:
                log_reliability = packet_size_bits * self.stability_manager.safe_log(1 - ber, default=-1000)
                reliability = self.stability_manager.safe_exp(log_reliability, 0.0)
            
            return self.stability_manager.clamp_float(reliability, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"可靠性计算异常：{str(e)}")
            return 0.5

    def safe_spectral_efficiency_calculation(self, data_rate: float, bandwidth: float) -> float:
        """
        安全的频谱效率计算
        
        参数:
        data_rate: 数据速率 (bps)
        bandwidth: 带宽 (Hz)
        
        返回:
        频谱效率 (bps/Hz)
        """
        try:
            data_rate = max(0.0, data_rate)
            bandwidth = self.stability_manager.validate_bandwidth(bandwidth)
            
            spectral_efficiency = self.stability_manager.safe_divide(data_rate, bandwidth, 0.0)
            return self.stability_manager.clamp_float(spectral_efficiency, 0.0, 20.0)
            
        except Exception as e:
            logger.warning(f"频谱效率计算异常：{str(e)}")
            return 1.0

    def safe_energy_efficiency_calculation(self, data_rate: float, total_power: float) -> float:
        """
        安全的能量效率计算
        
        参数:
        data_rate: 数据速率 (bps)
        total_power: 总功率消耗 (W)
        
        返回:
        能量效率 (bps/W)
        """
        try:
            data_rate = max(0.0, data_rate)
            total_power = self.stability_manager.validate_power(total_power)
            
            energy_efficiency = self.stability_manager.safe_divide(data_rate, total_power, 0.0)
            return self.stability_manager.clamp_float(energy_efficiency, 0.0, 1e6)
            
        except Exception as e:
            logger.warning(f"能量效率计算异常：{str(e)}")
            return 1000.0

    def safe_interference_calculation(self, signal_power: float, interference_powers: List[float]) -> float:
        """
        安全的干扰计算
        
        参数:
        signal_power: 期望信号功率 (W)
        interference_powers: 干扰信号功率列表 (W)
        
        返回:
        信干比 (SIR) (dB)
        """
        try:
            signal_power = self.stability_manager.validate_power(signal_power)
            
            total_interference = 0.0
            for interference in interference_powers:
                validated_interference = self.stability_manager.validate_power(interference)
                total_interference += validated_interference
            
            total_interference = max(total_interference, 1e-12)
            
            sir_linear = self.stability_manager.safe_divide(signal_power, total_interference, 1000.0)
            sir_db = 10 * self.stability_manager.safe_log10(sir_linear, 30.0)
            
            return self.stability_manager.clamp_float(sir_db, -50.0, 100.0)
            
        except Exception as e:
            logger.warning(f"干扰计算异常：{str(e)}")
            return 20.0

    def safe_path_loss_calculation(self, frequency: float, distance: float) -> float:
        """
        安全的路径损耗计算
        
        参数:
        frequency: 频率 (Hz)
        distance: 距离 (km)
        
        返回:
        路径损耗 (dB)
        """
        try:
            frequency = self.stability_manager.validate_frequency(frequency)
            distance = max(0.01, abs(distance))
            
            # Friis自由空间路径损耗
            wavelength = self.stability_manager.safe_divide(self.LIGHT_SPEED, frequency, 1.0)
            
            term1 = 20 * self.stability_manager.safe_log10(4 * np.pi)
            term2 = 20 * self.stability_manager.safe_log10(distance * 1000)
            term3 = -20 * self.stability_manager.safe_log10(wavelength)
            
            path_loss = term1 + term2 + term3
            return self.stability_manager.clamp_float(path_loss, 30.0, 200.0)
            
        except Exception as e:
            logger.warning(f"路径损耗计算异常：{str(e)}")
            return 120.0

    def validate_communication_parameters(self, params: Dict) -> Dict:
        """
        验证和修正通信参数字典
        
        参数:
        params: 通信参数字典
        
        返回:
        验证后的参数字典
        """
        try:
            validated_params = params.copy()
            
            if 'frequency' in validated_params:
                validated_params['frequency'] = self.stability_manager.validate_frequency(
                    validated_params['frequency'])
            
            if 'bandwidth' in validated_params:
                validated_params['bandwidth'] = self.stability_manager.validate_bandwidth(
                    validated_params['bandwidth'])
            
            if 'power' in validated_params:
                validated_params['power'] = self.stability_manager.validate_power(
                    validated_params['power'])
            
            if 'snr' in validated_params:
                validated_params['snr'] = self.stability_manager.validate_snr(
                    validated_params['snr'])
            
            # 验证调制方式
            if 'modulation' in validated_params:
                valid_modulations = ['BPSK', 'QPSK', 'QAM16', 'QAM64']
                if validated_params['modulation'] not in valid_modulations:
                    validated_params['modulation'] = 'BPSK'
            
            # 验证极化方式
            if 'polarization' in validated_params:
                valid_polarizations = ['LINEAR', 'CIRCULAR', 'DUAL', 'ADAPTIVE']
                if validated_params['polarization'] not in valid_polarizations:
                    validated_params['polarization'] = 'LINEAR'
            
            return validated_params
            
        except Exception as e:
            logger.error(f"参数验证异常：{str(e)}")
            return {
                'frequency': 2.4e9,
                'bandwidth': 20e6,
                'power': 10.0,
                'modulation': 'BPSK',
                'polarization': 'LINEAR'
            }


# 全局SafeCalculator实例
_safe_calculator_instance = None

def get_safe_calculator() -> SafeCalculator:
    """
    获取SafeCalculator实例（单例模式）
    
    返回:
    SafeCalculator实例
    """
    global _safe_calculator_instance
    
    if _safe_calculator_instance is None:
        _safe_calculator_instance = SafeCalculator()
        logger.debug("创建SafeCalculator单例实例")
    
    return _safe_calculator_instance


# 便利函数，直接使用单例实例的方法
def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """便利函数：安全除法"""
    return get_stability_manager().safe_divide(numerator, denominator, default)


def safe_log(x: float, base: float = np.e, default: float = 0.0) -> float:
    """便利函数：安全对数"""
    return get_stability_manager().safe_log(x, base, default)


def safe_exp(x: float, default: float = 1.0) -> float:
    """便利函数：安全指数"""
    return get_stability_manager().safe_exp(x, default)


def validate_objective_value(value: float, obj_type: str = "general") -> float:
    """便利函数：验证目标函数值"""
    return get_stability_manager().validate_objective_value(value, obj_type)


def validate_constraint_value(value: float) -> float:
    """便利函数：验证约束值"""
    return get_stability_manager().validate_constraint_value(value)


def clamp_float(value: float, min_val: Optional[float] = None, 
               max_val: Optional[float] = None) -> float:
    """便利函数：限制浮点数范围"""
    return get_stability_manager().clamp_float(value, min_val, max_val)


# 专门的通信系统验证函数
def validate_frequency(freq: float) -> float:
    """便利函数：验证频率值"""
    return get_stability_manager().validate_frequency(freq)


def validate_power(power: float) -> float:
    """便利函数：验证功率值"""
    return get_stability_manager().validate_power(power)


def validate_bandwidth(bw: float) -> float:
    """便利函数：验证带宽值"""
    return get_stability_manager().validate_bandwidth(bw)


def validate_snr(snr: float) -> float:
    """便利函数：验证SNR值"""
    return get_stability_manager().validate_snr(snr)


def safe_ber_calculation(snr_db: float, modulation: str = "BPSK") -> float:
    """便利函数：安全BER计算"""
    return get_stability_manager().safe_ber_calculation(snr_db, modulation)


def validate_array(arr: np.ndarray, default_value: float = 0.0,
                  min_val: Optional[float] = None, max_val: Optional[float] = None) -> np.ndarray:
    """便利函数：验证数组"""
    return get_stability_manager().validate_array(arr, default_value, min_val, max_val)


def validate_objective_array(objectives: np.ndarray) -> np.ndarray:
    """便利函数：验证目标函数值数组"""
    return get_stability_manager().validate_objective_array(objectives)


def validate_constraint_array(constraints: np.ndarray) -> np.ndarray:
    """便利函数：验证约束值数组"""
    return get_stability_manager().validate_constraint_array(constraints)


def validate_population_array(population: np.ndarray, 
                             bounds: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """便利函数：验证种群数组"""
    return get_stability_manager().validate_population_array(population, bounds)


def validate_optimization_results_array(objectives: np.ndarray, 
                                       constraints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """便利函数：验证优化结果数组"""
    return get_stability_manager().validate_optimization_results(objectives, constraints)


# SafeCalculator 便利函数
def get_calculator() -> SafeCalculator:
    """便利函数：获取SafeCalculator实例"""
    return get_safe_calculator()


def safe_snr_calculation(signal_power: float, noise_power: float, return_db: bool = True) -> float:
    """便利函数：安全SNR计算"""
    return get_safe_calculator().safe_snr_calculation(signal_power, noise_power, return_db)


def safe_capacity_calculation(snr_linear: float, bandwidth: float) -> float:
    """便利函数：安全容量计算"""
    return get_safe_calculator().safe_capacity_calculation(snr_linear, bandwidth)


def safe_ber_calculation_extended(snr_db: float, modulation_type: str) -> float:
    """便利函数：安全BER计算（扩展版）"""
    return get_safe_calculator().safe_ber_calculation(snr_db, modulation_type)


def safe_reliability_calculation(ber: float, packet_size_bits: int = 1024) -> float:
    """便利函数：安全可靠性计算"""
    return get_safe_calculator().safe_reliability_calculation(ber, packet_size_bits)


def safe_spectral_efficiency_calculation(data_rate: float, bandwidth: float) -> float:
    """便利函数：安全频谱效率计算"""
    return get_safe_calculator().safe_spectral_efficiency_calculation(data_rate, bandwidth)


def safe_energy_efficiency_calculation(data_rate: float, total_power: float) -> float:
    """便利函数：安全能量效率计算"""
    return get_safe_calculator().safe_energy_efficiency_calculation(data_rate, total_power)


def safe_interference_calculation(signal_power: float, interference_powers: List[float]) -> float:
    """便利函数：安全干扰计算"""
    return get_safe_calculator().safe_interference_calculation(signal_power, interference_powers)


def safe_path_loss_calculation_extended(frequency: float, distance: float) -> float:
    """便利函数：安全路径损耗计算（扩展版）"""
    return get_safe_calculator().safe_path_loss_calculation(frequency, distance)


def validate_communication_parameters(params: Dict) -> Dict:
    """便利函数：验证通信参数"""
    return get_safe_calculator().validate_communication_parameters(params)


# 模块初始化检查
def check_module_integrity():
    """检查模块完整性"""
    try:
        # 测试单例模式
        mgr1 = get_stability_manager()
        mgr2 = get_stability_manager()
        
        if mgr1 is not mgr2:
            logger.error("单例模式实现失败")
            return False
        
        # 测试基本功能
        test_result = mgr1.safe_divide(10, 2, 0)
        if abs(test_result - 5.0) > 1e-9:
            logger.error("基本计算功能测试失败")
            return False
        
        logger.debug("数值稳定性模块完整性检查通过")
        return True
        
    except Exception as e:
        logger.error(f"模块完整性检查失败：{str(e)}")
        return False


# 模块级别的异常处理装饰器
def with_numerical_stability(func):
    """
    装饰器：为函数添加数值稳定性保护
    
    用法:
    @with_numerical_stability
    def my_calculation_function(x, y):
        return x / y
    """
    def wrapper(*args, **kwargs):
        try:
            with get_stability_manager():
                return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"函数 {func.__name__} 执行异常：{str(e)}")
            raise
    
    return wrapper


# 模块加载时执行的初始化
if __name__ != "__main__":
    # 只在作为模块导入时进行检查
    if not check_module_integrity():
        logger.warning("数值稳定性模块初始化检查失败，部分功能可能不可用")


# 测试代码（仅在直接运行此文件时执行）
if __name__ == "__main__":
    # 设置测试日志
    logging.basicConfig(level=logging.DEBUG)
    
    print("数值稳定性管理器测试")
    print("=" * 40)
    
    # 测试单例模式
    mgr1 = get_stability_manager()
    mgr2 = get_stability_manager()
    print(f"单例测试: {mgr1 is mgr2}")
    
    # 测试SafeCalculator功能
    print("\nSafeCalculator测试:")
    print("-" * 20)
    
    calc = get_safe_calculator()
    
    # 测试SNR计算
    snr_linear = safe_snr_calculation(100.0, 1.0, return_db=False)
    snr_db = safe_snr_calculation(100.0, 1.0, return_db=True)
    print(f"SNR计算 (100W/1W): 线性={snr_linear:.2f}, dB={snr_db:.2f}")
    
    # 测试容量计算
    capacity = safe_capacity_calculation(100.0, 20e6)
    print(f"Shannon容量 (SNR=100, BW=20MHz): {capacity/1e6:.2f} Mbps")
    
    # 测试BER计算
    ber_bpsk = safe_ber_calculation_extended(20.0, 'BPSK')
    ber_qam16 = safe_ber_calculation_extended(20.0, 'QAM16')
    print(f"BER计算 (SNR=20dB): BPSK={ber_bpsk:.2e}, QAM16={ber_qam16:.2e}")
    
    # 测试可靠性计算
    reliability = safe_reliability_calculation(1e-6, 1024)
    print(f"可靠性计算 (BER=1e-6, 1024bits): {reliability:.6f}")
    
    # 测试频谱效率
    spectral_eff = safe_spectral_efficiency_calculation(100e6, 20e6)
    print(f"频谱效率 (100Mbps/20MHz): {spectral_eff:.2f} bps/Hz")
    
    # 测试能量效率
    energy_eff = safe_energy_efficiency_calculation(100e6, 50.0)
    print(f"能量效率 (100Mbps/50W): {energy_eff:.0f} bps/W")
    
    # 测试干扰计算
    sir = safe_interference_calculation(100.0, [1.0, 2.0, 0.5])
    print(f"信干比 (100W vs [1,2,0.5]W): {sir:.2f} dB")
    
    # 测试路径损耗
    path_loss = safe_path_loss_calculation_extended(2.4e9, 10.0)
    print(f"路径损耗 (2.4GHz, 10km): {path_loss:.2f} dB")
    
    # 测试参数验证
    test_params = {
        'frequency': 2.4e9,
        'bandwidth': 20e6,
        'power': 25.0,
        'modulation': 'QPSK',
        'polarization': 'CIRCULAR'
    }
    validated_params = validate_communication_parameters(test_params)
    print(f"参数验证测试: {validated_params['modulation']}, {validated_params['polarization']}")
    
    # 测试异常参数
    bad_params = {
        'frequency': -1000,
        'bandwidth': np.inf,
        'power': np.nan,
        'modulation': 'INVALID',
        'polarization': 'UNKNOWN'
    }
    fixed_params = validate_communication_parameters(bad_params)
    print(f"异常参数修正: freq={fixed_params['frequency']:.0f}, "
          f"mod={fixed_params['modulation']}, pol={fixed_params['polarization']}")
    
    # 测试数组验证功能
    print("\n数组验证测试:")
    print("-" * 20)
    
    # 测试包含异常值的数组
    test_array = np.array([1.0, np.nan, np.inf, -np.inf, 5.0, 1000.0])
    print(f"原始数组: {test_array}")
    validated = validate_array(test_array, default_value=0.0, min_val=-100, max_val=100)
    print(f"验证后: {validated}")
    
    # 测试目标函数数组
    test_objectives = np.array([
        [-100000.0, -100000.0, 100000.0, -100000.0, -100000.0],  # 异常值
        [0.8, 2.1, 50.0, 0.5, 0.9],  # 正常值
        [np.nan, np.inf, -np.inf, 1.5, 0.7]  # 包含异常值
    ])
    print(f"\n原始目标函数数组:\n{test_objectives}")
    validated_obj = validate_objective_array(test_objectives)
    print(f"验证后目标函数数组:\n{validated_obj}")
    
    # 测试约束数组
    test_constraints = np.array([
        [np.nan, 1.0, -5.0],
        [np.inf, 0.0, 2.0],
        [-np.inf, -1.0, 10000.0]
    ])
    print(f"\n原始约束数组:\n{test_constraints}")
    validated_const = validate_constraint_array(test_constraints)
    print(f"验证后约束数组:\n{validated_const}")
    
    # 测试种群数组
    test_population = np.array([
        [1e9, 20e6, 25.0, 1.0, 0.0],  # 正常个体
        [np.nan, np.inf, -100.0, 5.0, 2.0],  # 异常个体
        [5e9, 15e6, 30.0, 2.0, 1.0]   # 正常个体
    ])
    test_bounds = (
        np.array([100e6, 5e6, 5.0, 0.0, 0.0]),  # 下界
        np.array([10e9, 50e6, 50.0, 3.0, 3.0])  # 上界
    )
    print(f"\n原始种群数组:\n{test_population}")
    validated_pop = validate_population_array(test_population, test_bounds)
    print(f"验证后种群数组:\n{validated_pop}")
    
    # 测试基本计算
    print(f"安全除法 10/2: {safe_divide(10, 2)}")
    print(f"安全除法 10/0: {safe_divide(10, 0, -1)}")
    print(f"安全对数 ln(10): {safe_log(10)}")
    print(f"安全对数 ln(-1): {safe_log(-1, default=-999)}")
    print(f"安全指数 exp(1): {safe_exp(1)}")
    print(f"安全指数 exp(1000): {safe_exp(1000)}")
    
    # 测试通信参数验证
    print(f"频率验证 (5GHz): {validate_frequency(5e9)}")
    print(f"频率验证 (无效值): {validate_frequency(-100)}")
    print(f"功率验证 (20W): {validate_power(20)}")
    print(f"带宽验证 (10MHz): {validate_bandwidth(10e6)}")
    
    # 测试BER计算
    print(f"BER计算 (SNR=20dB, BPSK): {safe_ber_calculation(20, 'BPSK')}")
    print(f"BER计算 (SNR=10dB, QAM16): {safe_ber_calculation(10, 'QAM16')}")
    
    # 测试统计信息
    stats = mgr1.get_statistics()
    calc_stats = calc.stability_manager.get_statistics()
    print(f"管理器统计: {stats}")
    print(f"计算器统计: {calc_stats}")
    
    print("=" * 40)
    print("测试完成)=20dB, BPSK): {safe_ber_calculation(20, 'BPSK')}")
    print(f"BER计算 (SNR=10dB, QAM16): {safe_ber_calculation(10, 'QAM16')}")
    
    # 测试统计信息
    stats = mgr1.get_statistics()
    calc_stats = calc.stability_manager.get_statistics()
    print(f"管理器统计: {stats}")
    print(f"计算器统计: {calc_stats}")
    
    print("=" * 40)
    print("测试完成)=20dB, BPSK): {safe_ber_calculation(20, 'BPSK')}")
    print(f"BER计算 (SNR=10dB, QAM16): {safe_ber_calculation(10, 'QAM16')}")
    
    # 测试统计信息
    stats = mgr1.get_statistics()
    print(f"统计信息: {stats}")
    
    print("=" * 40)
    print("测试完成")


