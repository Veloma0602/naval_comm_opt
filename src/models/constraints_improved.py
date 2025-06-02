import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ImprovedConstraints:
    """
    改进的约束处理类，解决数值异常和约束不匹配问题
    主要改进：
    1. 数值稳定性保护
    2. 动态约束数量调整
    3. 软约束机制
    4. 约束权重自适应
    """
    
    def __init__(self, config, enable_soft_constraints=True):
        """
        初始化改进的约束条件处理器
        
        参数:
        config: 优化配置
        enable_soft_constraints: 是否启用软约束机制
        """
        self.config = config
        self.enable_soft_constraints = enable_soft_constraints
        
        # 数值稳定性保护常数
        self.EPSILON = 1e-10
        self.MAX_CONSTRAINT_VALUE = 1e6
        self.MIN_CONSTRAINT_VALUE = -1e6
        
        # 约束权重 - 根据重要性调整
        self.constraint_weights = {
            'frequency_bounds': 1.0,      # 频率边界约束最重要
            'power_bounds': 1.0,          # 功率边界约束
            'bandwidth_bounds': 0.8,      # 带宽边界约束
            'frequency_spacing': 0.9,     # 频率间隔约束
            'snr_requirement': 0.7,       # SNR要求（软约束）
            'delay_requirement': 0.6      # 时延要求（软约束）
        }
        
        # 软约束参数
        self.soft_constraint_alpha = 2.0  # 软约束的惩罚因子
        
        logger.info(f"初始化改进约束处理器，软约束: {enable_soft_constraints}")

    def evaluate_constraints(self, params_list: List[Dict]) -> np.ndarray:
        """
        评估所有约束条件 - 改进版
        
        参数:
        params_list: 参数字典列表
        
        返回:
        约束值数组，负值表示满足约束，正值表示违反约束
        """
        if not params_list:
            logger.warning("参数列表为空，返回默认约束")
            return np.array([1.0])  # 返回违反约束的默认值
        
        try:
            constraints = []
            n_links = len(params_list)
            
            # 1. 硬约束 - 频率边界约束（最严格）
            freq_constraints = self._evaluate_frequency_bounds(params_list)
            constraints.extend(freq_constraints)
            
            # 2. 硬约束 - 功率边界约束
            power_constraints = self._evaluate_power_bounds(params_list)
            constraints.extend(power_constraints)
            
            # 3. 硬约束 - 带宽边界约束
            bandwidth_constraints = self._evaluate_bandwidth_bounds(params_list)
            constraints.extend(bandwidth_constraints)
            
            # 4. 软约束 - 频率间隔约束
            spacing_constraints = self._evaluate_frequency_spacing(params_list)
            constraints.extend(spacing_constraints)
            
            if self.enable_soft_constraints:
                # 5. 软约束 - SNR要求
                snr_constraints = self._evaluate_snr_requirements(params_list)
                constraints.extend(snr_constraints)
                
                # 6. 软约束 - 时延要求
                delay_constraints = self._evaluate_delay_requirements(params_list)
                constraints.extend(delay_constraints)
            
            # 数值稳定性检查和处理
            constraints = self._apply_numerical_stability(constraints)
            
            # 动态调整约束数量以匹配期望
            expected_constraints = self._calculate_expected_constraints(n_links)
            constraints = self._adjust_constraint_count(constraints, expected_constraints)
            
            logger.debug(f"约束评估完成: {len(constraints)} 个约束，违反数量: {sum(1 for c in constraints if c > 0)}")
            
            return np.array(constraints)
            
        except Exception as e:
            logger.error(f"约束评估过程出错: {str(e)}")
            # 返回安全的默认约束值
            n_links = max(1, len(params_list))
            default_constraints = np.ones(n_links * 4) * 0.5  # 轻微违反约束
            return default_constraints

    def _evaluate_frequency_bounds(self, params_list: List[Dict]) -> List[float]:
        """评估频率边界约束 - 改进版"""
        constraints = []
        weight = self.constraint_weights['frequency_bounds']
        
        for i, params in enumerate(params_list):
            try:
                freq = self._safe_get_numeric(params, 'frequency', 4e9)
                
                # 下界约束 - 使用归一化违反度
                if freq < self.config.freq_min:
                    violation = (self.config.freq_min - freq) / self.config.freq_min
                    constraints.append(min(violation * weight, 10.0))  # 限制最大惩罚
                else:
                    constraints.append(-0.1)  # 满足约束
                
                # 上界约束
                if freq > self.config.freq_max:
                    violation = (freq - self.config.freq_max) / self.config.freq_max
                    constraints.append(min(violation * weight, 10.0))
                else:
                    constraints.append(-0.1)
                    
            except Exception as e:
                logger.warning(f"频率约束评估出错（链路{i}）: {str(e)}")
                constraints.extend([1.0, 1.0])  # 默认违反约束
        
        return constraints

    def _evaluate_power_bounds(self, params_list: List[Dict]) -> List[float]:
        """评估功率边界约束 - 改进版"""
        constraints = []
        weight = self.constraint_weights['power_bounds']
        
        for i, params in enumerate(params_list):
            try:
                power = self._safe_get_numeric(params, 'power', 10.0)
                
                # 下界约束
                if power < self.config.power_min:
                    violation = (self.config.power_min - power) / self.config.power_min
                    constraints.append(min(violation * weight, 5.0))
                else:
                    constraints.append(-0.1)
                
                # 上界约束
                if power > self.config.power_max:
                    violation = (power - self.config.power_max) / self.config.power_max
                    constraints.append(min(violation * weight, 5.0))
                else:
                    constraints.append(-0.1)
                    
            except Exception as e:
                logger.warning(f"功率约束评估出错（链路{i}）: {str(e)}")
                constraints.extend([1.0, 1.0])
        
        return constraints

    def _evaluate_bandwidth_bounds(self, params_list: List[Dict]) -> List[float]:
        """评估带宽边界约束 - 改进版"""
        constraints = []
        weight = self.constraint_weights['bandwidth_bounds']
        
        for i, params in enumerate(params_list):
            try:
                bandwidth = self._safe_get_numeric(params, 'bandwidth', 20e6)
                
                # 下界约束
                if bandwidth < self.config.bandwidth_min:
                    violation = (self.config.bandwidth_min - bandwidth) / self.config.bandwidth_min
                    constraints.append(min(violation * weight, 3.0))
                else:
                    constraints.append(-0.1)
                
                # 上界约束
                if bandwidth > self.config.bandwidth_max:
                    violation = (bandwidth - self.config.bandwidth_max) / self.config.bandwidth_max
                    constraints.append(min(violation * weight, 3.0))
                else:
                    constraints.append(-0.1)
                    
            except Exception as e:
                logger.warning(f"带宽约束评估出错（链路{i}）: {str(e)}")
                constraints.extend([1.0, 1.0])
        
        return constraints

    def _evaluate_frequency_spacing(self, params_list: List[Dict]) -> List[float]:
        """评估频率间隔约束 - 改进版软约束"""
        constraints = []
        weight = self.constraint_weights['frequency_spacing']
        n_links = len(params_list)
        
        if n_links < 2:
            return []  # 单链路无需间隔约束
        
        # 提取所有频率和带宽
        frequencies = []
        bandwidths = []
        
        for params in params_list:
            freq = self._safe_get_numeric(params, 'frequency', 4e9)
            bw = self._safe_get_numeric(params, 'bandwidth', 20e6)
            frequencies.append(freq)
            bandwidths.append(bw)
        
        # 检查每对链路的频率间隔
        for i in range(n_links):
            for j in range(i + 1, n_links):
                try:
                    # 计算频率中心点间隔
                    spacing = abs(frequencies[i] - frequencies[j])
                    
                    # 计算所需的最小间隔
                    required_spacing = (bandwidths[i] + bandwidths[j]) / 2 * 1.1  # 10%保护带
                    
                    if spacing < required_spacing:
                        # 软约束：使用平滑的惩罚函数
                        violation_ratio = (required_spacing - spacing) / required_spacing
                        penalty = self._soft_constraint_penalty(violation_ratio)
                        constraints.append(penalty * weight)
                    else:
                        constraints.append(-0.1)  # 满足约束
                        
                except Exception as e:
                    logger.warning(f"频率间隔约束评估出错（{i}-{j}）: {str(e)}")
                    constraints.append(0.5)  # 轻微违反
        
        return constraints

    def _evaluate_snr_requirements(self, params_list: List[Dict]) -> List[float]:
        """评估SNR要求 - 软约束"""
        constraints = []
        weight = self.constraint_weights['snr_requirement']
        
        for i, params in enumerate(params_list):
            try:
                # 简化的SNR估算
                estimated_snr = self._estimate_snr_simple(params)
                
                if estimated_snr < self.config.snr_min:
                    violation_ratio = (self.config.snr_min - estimated_snr) / self.config.snr_min
                    penalty = self._soft_constraint_penalty(violation_ratio)
                    constraints.append(penalty * weight)
                else:
                    constraints.append(-0.05)  # 满足要求
                    
            except Exception as e:
                logger.warning(f"SNR约束评估出错（链路{i}）: {str(e)}")
                constraints.append(0.3)  # 轻微违反
        
        return constraints

    def _evaluate_delay_requirements(self, params_list: List[Dict]) -> List[float]:
        """评估时延要求 - 软约束"""
        constraints = []
        weight = self.constraint_weights['delay_requirement']
        
        for i, params in enumerate(params_list):
            try:
                # 简化的时延估算
                estimated_delay = self._estimate_delay_simple(params)
                
                if estimated_delay > self.config.delay_max:
                    violation_ratio = (estimated_delay - self.config.delay_max) / self.config.delay_max
                    penalty = self._soft_constraint_penalty(violation_ratio)
                    constraints.append(penalty * weight)
                else:
                    constraints.append(-0.05)  # 满足要求
                    
            except Exception as e:
                logger.warning(f"时延约束评估出错（链路{i}）: {str(e)}")
                constraints.append(0.3)  # 轻微违反
        
        return constraints

    def _soft_constraint_penalty(self, violation_ratio: float) -> float:
        """
        软约束惩罚函数 - 使用平滑的非线性函数
        
        参数:
        violation_ratio: 违反比例 [0, 1]
        
        返回:
        惩罚值
        """
        if violation_ratio <= 0:
            return 0.0
        
        # 使用指数函数实现软约束
        penalty = (np.exp(self.soft_constraint_alpha * violation_ratio) - 1) / (np.exp(self.soft_constraint_alpha) - 1)
        return min(penalty, 2.0)  # 限制最大惩罚

    def _safe_get_numeric(self, params: Dict, key: str, default: float) -> float:
        """
        安全获取数值参数，包含类型转换和边界检查
        
        参数:
        params: 参数字典
        key: 参数键
        default: 默认值
        
        返回:
        数值
        """
        try:
            value = params.get(key, default)
            
            if isinstance(value, str):
                # 尝试从字符串中提取数值
                import re
                numeric_match = re.search(r'-?\d+\.?\d*', value)
                if numeric_match:
                    value = float(numeric_match.group())
                else:
                    value = default
            
            # 转换为浮点数
            value = float(value)
            
            # 数值边界检查
            if np.isnan(value) or np.isinf(value):
                logger.warning(f"检测到异常数值 {value}，使用默认值 {default}")
                return default
                
            # 合理性检查
            if key == 'frequency' and (value < 1e6 or value > 100e9):
                logger.warning(f"频率值 {value} 超出合理范围，使用默认值")
                return default
            elif key == 'power' and (value < 0.1 or value > 1000):
                logger.warning(f"功率值 {value} 超出合理范围，使用默认值")
                return default
            elif key == 'bandwidth' and (value < 1e3 or value > 1e9):
                logger.warning(f"带宽值 {value} 超出合理范围，使用默认值")
                return default
            
            return value
            
        except Exception as e:
            logger.warning(f"参数 {key} 解析失败: {str(e)}，使用默认值 {default}")
            return default

    def _estimate_snr_simple(self, params: Dict) -> float:
        """简化的SNR估算"""
        try:
            power = self._safe_get_numeric(params, 'power', 10.0)
            frequency = self._safe_get_numeric(params, 'frequency', 4e9)
            
            # 简化的SNR模型：功率越大SNR越高，频率越高损耗越大
            base_snr = 10 * np.log10(power + self.EPSILON)
            freq_loss = 10 * np.log10(frequency / 1e9)  # 频率损耗
            
            return base_snr - freq_loss + 15  # 基础SNR
            
        except Exception:
            return 15.0  # 默认SNR

    def _estimate_delay_simple(self, params: Dict) -> float:
        """简化的时延估算"""
        try:
            bandwidth = self._safe_get_numeric(params, 'bandwidth', 20e6)
            
            # 简化模型：带宽越大时延越小
            base_delay = 50.0  # 基础时延 (ms)
            bandwidth_factor = 20e6 / max(bandwidth, 1e6)
            
            return base_delay * bandwidth_factor
            
        except Exception:
            return 50.0  # 默认时延

    def _apply_numerical_stability(self, constraints: List[float]) -> List[float]:
        """应用数值稳定性保护"""
        stable_constraints = []
        
        for constraint in constraints:
            # 处理异常值
            if np.isnan(constraint) or np.isinf(constraint):
                stable_constraints.append(1.0)  # 默认违反约束
                continue
                
            # 限制约束值范围
            constraint = max(self.MIN_CONSTRAINT_VALUE, 
                           min(self.MAX_CONSTRAINT_VALUE, constraint))
            
            stable_constraints.append(constraint)
        
        return stable_constraints

    def _calculate_expected_constraints(self, n_links: int) -> int:
        """修正的约束数量计算 - 与优化器保持一致"""
        try:
            # 基本约束：每个链路6个边界约束
            base_constraints = n_links * 6  # 频率(2) + 功率(2) + 带宽(2)
            
            # 频率间隔约束（链路间）
            if n_links > 1:
                spacing_constraints = n_links * (n_links - 1) // 2
                base_constraints += spacing_constraints
            
            # 软约束（如果启用）
            if self.enable_soft_constraints:
                soft_constraints = n_links * 2  # SNR + 时延
                base_constraints += soft_constraints
            
            # 确保最小约束数量
            return max(base_constraints, n_links * 4)
            
        except Exception as e:
            logger.warning(f"约束数量计算失败：{str(e)}")
            return n_links * 4

    def _adjust_constraint_count(self, constraints: List[float], expected: int) -> List[float]:
        """动态调整约束数量以匹配期望"""
        current_count = len(constraints)
        
        if current_count == expected:
            return constraints
        elif current_count < expected:
            # 添加满足的约束
            padding = [-0.1] * (expected - current_count)
            constraints.extend(padding)
            logger.debug(f"添加 {len(padding)} 个约束以匹配期望数量")
        else:
            # 截断多余的约束，保留最重要的
            constraints = constraints[:expected]
            logger.debug(f"截断约束至期望数量 {expected}")
        
        return constraints

    def get_constraint_summary(self, constraints: np.ndarray) -> Dict[str, Any]:
        """获取约束评估摘要"""
        if len(constraints) == 0:
            return {"total": 0, "violated": 0, "max_violation": 0.0}
        
        violated_count = np.sum(constraints > 0)
        max_violation = np.max(constraints)
        mean_violation = np.mean(constraints[constraints > 0]) if violated_count > 0 else 0.0
        
        return {
            "total": len(constraints),
            "violated": int(violated_count),
            "satisfaction_rate": (len(constraints) - violated_count) / len(constraints),
            "max_violation": float(max_violation),
            "mean_violation": float(mean_violation),
            "constraint_vector_valid": not (np.any(np.isnan(constraints)) or np.any(np.isinf(constraints)))
        }


# 兼容性包装器，保持原有接口
class Constraints(ImprovedConstraints):
    """向后兼容的约束类"""
    
    def __init__(self, config):
        super().__init__(config, enable_soft_constraints=True)
        logger.info("使用改进的约束处理器（兼容性模式）")