import random
import numpy as np
from typing import List, Dict, Any
import logging

# 导入数值稳定性工具
from optimization.numerical_stability import get_stability_manager, SafeCalculator, validate_array
from .noise_model import NoiseModel

logger = logging.getLogger(__name__)

class ImprovedObjectiveFunction:
    """
    改进的目标函数类
    主要改进：
    1. 数值稳定性保护
    2. 目标函数值标准化
    3. 更鲁棒的计算逻辑
    4. 异常处理和日志记录
    """
    
    def __init__(self, task_data: Dict, env_data: Dict, constraint_data: Dict, config=None):
        """初始化改进的目标函数"""
        self.task_data = task_data
        self.env_data = env_data.copy() if env_data else {}
        self.constraint_data = constraint_data.copy() if constraint_data else {}
        self.config = config
        
        # 初始化数值稳定性管理器和计算器
        self.stability_manager = get_stability_manager()
        self.calculator = SafeCalculator()
        
        # 处理环境数据键名映射
        self._process_data_keys()
        
        # 初始化噪声模型
        try:
            self.noise_model = NoiseModel(self.env_data)
        except Exception as e:
            logger.error(f"噪声模型初始化失败：{str(e)}")
            self.noise_model = None
        
        # 目标函数值归一化参数
        self.objective_scales = {
            'reliability': {'min': 0.0, 'max': 5.0, 'target': 'maximize'},
            'spectral_efficiency': {'min': 0.0, 'max': 10.0, 'target': 'maximize'},
            'energy_efficiency': {'min': 0.0, 'max': 100.0, 'target': 'minimize'},
            'interference': {'min': -2.0, 'max': 2.0, 'target': 'maximize'},
            'adaptability': {'min': 0.0, 'max': 3.0, 'target': 'maximize'}
        }
        
        # 初始化参数
        self._setup_params()
        
        logger.info("改进的目标函数初始化完成")

    def _process_data_keys(self):
        """处理数据键名映射"""
        # 环境数据键名映射
        env_key_mapping = {
            '海况等级': 'sea_state',
            '电磁干扰强度': 'emi_intensity',
            '背景噪声': 'background_noise',
            '多径效应': 'multipath_effect',
            '温度': 'temperature',
            '盐度': 'salinity',
            '深度': 'depth'
        }
        
        for cn_key, en_key in env_key_mapping.items():
            if cn_key in self.env_data and en_key not in self.env_data:
                self.env_data[en_key] = self.env_data[cn_key]
        
        # 确保有默认深度值
        if 'depth' not in self.env_data and '深度' not in self.env_data:
            self.env_data['depth'] = 50
        
        # 约束数据键名映射
        if self.constraint_data:
            constraint_mapping = {
                '最小可靠性要求': 'min_reliability',
                '最大时延要求': 'max_delay',
                '最小信噪比': 'min_snr'
            }
            
            for cn_key, en_key in constraint_mapping.items():
                if cn_key in self.constraint_data and en_key not in self.constraint_data:
                    self.constraint_data[en_key] = self.constraint_data[cn_key]

    def _setup_params(self):
        """设置计算所需的参数"""
        try:
            # 从环境数据中提取信息
            self.sea_state = self._parse_numeric_value(self.env_data.get('海况等级', 3))
            self.emi_intensity = self._parse_numeric_value(self.env_data.get('电磁干扰强度', 0.5))
            self.background_noise = self._parse_numeric_value(self.env_data.get('背景噪声', -100))
            
            # 从约束数据中提取信息
            self.min_reliability = self._parse_numeric_value(self.constraint_data.get('最小可靠性要求', 0.95))
            self.max_delay = self._parse_numeric_value(self.constraint_data.get('最大时延要求', 100))
            self.min_snr = self._parse_numeric_value(self.constraint_data.get('最小信噪比', 15))
            
            logger.debug(f"参数设置完成：海况={self.sea_state}, EMI={self.emi_intensity}")
            
        except Exception as e:
            logger.error(f"参数设置失败：{str(e)}")
            # 使用默认值
            self.sea_state = 3
            self.emi_intensity = 0.5
            self.background_noise = -100
            self.min_reliability = 0.95
            self.max_delay = 100
            self.min_snr = 15

    def reliability_objective(self, params_list: List[Dict]) -> float:
        """
        通信可靠性目标函数 - 改进版
        """
        try:
            if not params_list:
                logger.warning("参数列表为空")
                return self._get_worst_objective('reliability')
            
            total_reliability = 0
            links = self.task_data.get('communication_links', [])
            
            for i, params in enumerate(params_list):
                try:
                    if i < len(links):
                        link = links[i]
                        # 使用更准确的可靠性计算
                        reliability = self._calculate_link_reliability(params, link)
                    else:
                        # 如果没有对应链路，使用简化计算
                        reliability = self._calculate_simple_reliability(params)
                    
                    # 应用链路重要性权重
                    importance = self._get_link_importance(links[i] if i < len(links) else {})
                    total_reliability += reliability * importance
                    
                except Exception as e:
                    logger.warning(f"链路 {i} 可靠性计算失败：{str(e)}")
                    total_reliability += 0.5  # 默认中等可靠性
            
            # 标准化到目标范围
            normalized_reliability = self._normalize_objective(
                total_reliability, 'reliability'
            )
            
            # 返回负值用于最小化
            return -normalized_reliability
            
        except Exception as e:
            logger.error(f"可靠性目标函数计算异常：{str(e)}")
            return self._get_worst_objective('reliability')

    def _calculate_link_reliability(self, params: Dict, link: Dict) -> float:
        """计算单个链路的可靠性"""
        try:
            # 计算SNR
            snr = self._calculate_snr_robust(params, link)
            
            # 计算误码率
            modulation = params.get('modulation', 'BPSK')
            ber = self.calculator.safe_ber_calculation(snr, modulation)
            
            # 计算包成功率（基于包大小）
            packet_size = link.get('packet_size', 1024)
            packet_success_rate = self.stability_manager.safe_power(1 - ber, packet_size, 0.5)
            
            # 考虑环境因素
            env_factor = self._calculate_environment_reliability_factor()
            
            # 综合可靠性
            reliability = packet_success_rate * env_factor
            
            return self.stability_manager.clamp_float(reliability, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"链路可靠性计算失败：{str(e)}")
            return 0.5

    def _calculate_simple_reliability(self, params: Dict) -> float:
        """简化的可靠性计算"""
        try:
            freq = self._parse_numeric_value(params.get('frequency', 0))
            power = self._parse_numeric_value(params.get('power', 0))
            bandwidth = self._parse_numeric_value(params.get('bandwidth', 0))
            
            if not self.config:
                return 0.5
            
            # 频率因子（中频段最佳）
            freq_factor = 1.0 - abs(freq - (self.config.freq_min + self.config.freq_max) / 2) / (self.config.freq_max - self.config.freq_min)
            freq_factor = max(0.2, freq_factor)
            
            # 功率因子
            power_factor = self.stability_manager.safe_divide(
                power, self.config.power_max, 0.5
            )
            
            # 带宽因子
            bw_optimal = (self.config.bandwidth_min + self.config.bandwidth_max) / 2
            bw_factor = 1.0 - abs(bandwidth - bw_optimal) / bw_optimal
            bw_factor = max(0.2, bw_factor)
            
            # 综合可靠性
            reliability = 0.4 * freq_factor + 0.4 * power_factor + 0.2 * bw_factor
            
            return self.stability_manager.clamp_float(reliability, 0.1, 0.99)
            
        except Exception as e:
            logger.warning(f"简化可靠性计算失败：{str(e)}")
            return 0.5

    def spectral_efficiency_objective(self, params_list: List[Dict]) -> float:
        """
        频谱效率目标函数 - 改进版
        """
        try:
            if not params_list:
                return self._get_worst_objective('spectral_efficiency')
            
            total_efficiency = 0
            links = self.task_data.get('communication_links', [])
            
            for i, params in enumerate(params_list):
                try:
                    if i < len(links):
                        link = links[i]
                        efficiency = self._calculate_link_spectral_efficiency(params, link)
                    else:
                        efficiency = self._calculate_simple_spectral_efficiency(params)
                    
                    # 应用链路重要性权重
                    importance = self._get_link_importance(links[i] if i < len(links) else {})
                    total_efficiency += efficiency * importance
                    
                except Exception as e:
                    logger.warning(f"链路 {i} 频谱效率计算失败：{str(e)}")
                    total_efficiency += 1.0  # 默认效率
            
            # 标准化到目标范围
            normalized_efficiency = self._normalize_objective(
                total_efficiency, 'spectral_efficiency'
            )
            
            # 添加随机扰动避免重复值
            noise_factor = 0.98 + 0.04 * random.random()
            normalized_efficiency *= noise_factor
            
            return -normalized_efficiency  # 负值用于最小化
            
        except Exception as e:
            logger.error(f"频谱效率目标函数计算异常：{str(e)}")
            return self._get_worst_objective('spectral_efficiency')

    def _calculate_link_spectral_efficiency(self, params: Dict, link: Dict) -> float:
        """计算单个链路的频谱效率"""
        try:
            bandwidth = self._parse_numeric_value(params.get('bandwidth', 0))
            if bandwidth <= 0:
                return 0.1
            
            snr = self._calculate_snr_robust(params, link)
            
            # Shannon容量计算
            snr_linear = self.stability_manager.safe_exp(snr * np.log(10) / 10, 1.0)
            theoretical_capacity = self.calculator.safe_capacity_calculation(bandwidth, snr_linear)
            
            # 实际调制方式限制
            modulation = params.get('modulation', 'BPSK')
            mod_efficiency_map = {
                'BPSK': 1.0,
                'QPSK': 2.0,
                'QAM16': 4.0,
                'QAM64': 6.0
            }
            max_mod_efficiency = mod_efficiency_map.get(modulation, 1.0)
            
            # 实际频谱效率
            practical_efficiency = min(
                self.stability_manager.safe_divide(theoretical_capacity, bandwidth, 0.1),
                max_mod_efficiency
            )
            
            # 频谱利用率加权
            if bandwidth < 0.3 * self.config.bandwidth_max:
                efficiency_weight = 1.2  # 鼓励高效利用小带宽
            elif bandwidth > 0.8 * self.config.bandwidth_max:
                efficiency_weight = 0.9  # 大带宽稍微降权
            else:
                efficiency_weight = 1.0
            
            return practical_efficiency * efficiency_weight
            
        except Exception as e:
            logger.warning(f"链路频谱效率计算失败：{str(e)}")
            return 1.0

    def _calculate_simple_spectral_efficiency(self, params: Dict) -> float:
        """简化的频谱效率计算"""
        try:
            bandwidth = self._parse_numeric_value(params.get('bandwidth', 0))
            modulation = params.get('modulation', 'BPSK')
            
            if bandwidth <= 0:
                return 0.1
            
            # 基于调制方式的基础效率
            base_efficiency = {
                'BPSK': 1.0,
                'QPSK': 1.8,
                'QAM16': 3.5,
                'QAM64': 5.5
            }.get(modulation, 1.0)
            
            # 带宽利用率因子
            if self.config:
                bw_utilization = min(1.0, bandwidth / self.config.bandwidth_max)
                efficiency = base_efficiency * (0.7 + 0.3 * bw_utilization)
            else:
                efficiency = base_efficiency
            
            return self.stability_manager.clamp_float(efficiency, 0.1, 10.0)
            
        except Exception as e:
            logger.warning(f"简化频谱效率计算失败：{str(e)}")
            return 1.0

    def energy_efficiency_objective(self, params_list: List[Dict]) -> float:
        """
        能量效率目标函数 - 改进版
        """
        try:
            if not params_list:
                return self._get_worst_objective('energy_efficiency')
            
            total_energy_cost = 0
            links = self.task_data.get('communication_links', [])
            
            for i, params in enumerate(params_list):
                try:
                    if i < len(links):
                        link = links[i]
                        energy_cost = self._calculate_link_energy_cost(params, link)
                    else:
                        energy_cost = self._calculate_simple_energy_cost(params)
                    
                    total_energy_cost += energy_cost
                    
                except Exception as e:
                    logger.warning(f"链路 {i} 能量效率计算失败：{str(e)}")
                    total_energy_cost += 10.0  # 默认能量成本
            
            # 标准化到目标范围
            normalized_cost = self._normalize_objective(
                total_energy_cost, 'energy_efficiency'
            )
            
            # 能量效率是最小化目标，直接返回
            return normalized_cost
            
        except Exception as e:
            logger.error(f"能量效率目标函数计算异常：{str(e)}")
            return self._get_worst_objective('energy_efficiency')

    def _calculate_link_energy_cost(self, params: Dict, link: Dict) -> float:
        """计算单个链路的能量成本"""
        try:
            power = self._parse_numeric_value(params.get('power', 0))
            
            # 计算有效数据率
            bandwidth = self._parse_numeric_value(params.get('bandwidth', 0))
            snr = self._calculate_snr_robust(params, link)
            
            # 实际数据率（考虑误码率）
            modulation = params.get('modulation', 'BPSK')
            ber = self.calculator.safe_ber_calculation(snr, modulation)
            
            # 有效数据率
            snr_linear = self.stability_manager.safe_exp(snr * np.log(10) / 10, 1.0)
            theoretical_rate = self.calculator.safe_capacity_calculation(bandwidth, snr_linear)
            effective_rate = theoretical_rate * (1 - ber) * 0.8  # 协议开销
            
            # 电路功耗
            circuit_power = power * 0.3  # 假设电路功耗为发射功率的30%
            total_power = power + circuit_power
            
            # 每比特能耗
            if effective_rate > 0:
                energy_per_bit = self.stability_manager.safe_divide(
                    total_power, effective_rate, 100.0
                )
            else:
                energy_per_bit = 100.0
            
            # 对数缩放到合理范围
            scaled_energy = self.stability_manager.safe_log(energy_per_bit + 1, 10, 1.0)
            
            return self.stability_manager.clamp_float(scaled_energy, 0.1, 50.0)
            
        except Exception as e:
            logger.warning(f"链路能量成本计算失败：{str(e)}")
            return 10.0

    def _calculate_simple_energy_cost(self, params: Dict) -> float:
        """简化的能量成本计算"""
        try:
            power = self._parse_numeric_value(params.get('power', 0))
            bandwidth = self._parse_numeric_value(params.get('bandwidth', 0))
            
            # 基础能量成本与功率成正比
            base_cost = power * 1.3  # 包含电路功耗
            
            # 带宽效率调整
            if bandwidth > 0 and self.config:
                bw_efficiency = bandwidth / self.config.bandwidth_max
                cost_factor = 1.0 + 0.5 * (1 - bw_efficiency)  # 低带宽利用率增加成本
                energy_cost = base_cost * cost_factor
            else:
                energy_cost = base_cost
            
            # 对数缩放
            scaled_cost = self.stability_manager.safe_log(energy_cost + 1, 10, 1.0)
            
            return self.stability_manager.clamp_float(scaled_cost, 0.5, 20.0)
            
        except Exception as e:
            logger.warning(f"简化能量成本计算失败：{str(e)}")
            return 5.0

    def interference_objective(self, params_list: List[Dict]) -> float:
        """
        抗干扰性能目标函数 - 改进版
        """
        try:
            if not params_list:
                return self._get_worst_objective('interference')
            
            interference_score = 0
            links = self.task_data.get('communication_links', [])
            
            # 1. 计算链路间干扰
            mutual_interference = self._calculate_mutual_interference(params_list)
            
            # 2. 计算环境干扰影响
            env_interference = self._calculate_environmental_interference_score(params_list, links)
            
            # 3. 计算抗干扰能力加分
            interference_resistance = self._calculate_interference_resistance(params_list)
            
            # 综合干扰评分
            total_interference = mutual_interference + env_interference - interference_resistance
            
            # 标准化到目标范围
            normalized_interference = self._normalize_objective(
                total_interference, 'interference'
            )
            
            # 返回负值（抗干扰能力越强越好）
            return -normalized_interference
            
        except Exception as e:
            logger.error(f"抗干扰目标函数计算异常：{str(e)}")
            return self._get_worst_objective('interference')

    def _calculate_mutual_interference(self, params_list: List[Dict]) -> float:
        """计算链路间相互干扰"""
        try:
            mutual_interference = 0
            n_links = len(params_list)
            
            for i in range(n_links):
                for j in range(i + 1, n_links):
                    # 频谱重叠计算
                    freq_i = self._parse_numeric_value(params_list[i].get('frequency', 0))
                    bw_i = self._parse_numeric_value(params_list[i].get('bandwidth', 0))
                    
                    freq_j = self._parse_numeric_value(params_list[j].get('frequency', 0))
                    bw_j = self._parse_numeric_value(params_list[j].get('bandwidth', 0))
                    
                    overlap = self._calculate_frequency_overlap(freq_i, bw_i, freq_j, bw_j)
                    
                    if overlap > 0:
                        # 功率比影响干扰强度
                        power_i = self._parse_numeric_value(params_list[i].get('power', 0))
                        power_j = self._parse_numeric_value(params_list[j].get('power', 0))
                        
                        power_ratio = self.stability_manager.safe_divide(
                            min(power_i, power_j), max(power_i, power_j), 0.1
                        )
                        
                        interference_level = overlap * (0.5 + 0.5 * power_ratio)
                        mutual_interference += interference_level
            
            return self.stability_manager.clamp_float(mutual_interference, 0, 10.0)
            
        except Exception as e:
            logger.warning(f"相互干扰计算失败：{str(e)}")
            return 2.0

    def _calculate_environmental_interference_score(self, params_list: List[Dict], 
                                                  links: List[Dict]) -> float:
        """计算环境干扰影响评分"""
        try:
            total_env_interference = 0
            
            for i, params in enumerate(params_list):
                freq = self._parse_numeric_value(params.get('frequency', 0))
                bandwidth = self._parse_numeric_value(params.get('bandwidth', 0))
                
                link = links[i] if i < len(links) else {}
                
                # 环境干扰基础评分
                env_score = self._calculate_single_env_interference(freq, bandwidth, link)
                total_env_interference += env_score
            
            return self.stability_manager.clamp_float(total_env_interference, 0, 5.0)
            
        except Exception as e:
            logger.warning(f"环境干扰评分计算失败：{str(e)}")
            return 1.0

    def _calculate_single_env_interference(self, freq: float, bandwidth: float, 
                                         link: Dict) -> float:
        """计算单个链路的环境干扰"""
        try:
            # 海况影响
            sea_effect = self.sea_state / 9.0
            
            # EMI影响
            emi_effect = self.emi_intensity
            
            # 频率敏感性
            if freq < 500e6:  # HF/VHF频段
                freq_sensitivity = 0.8
            elif freq < 2e9:  # UHF频段
                freq_sensitivity = 0.6
            elif freq < 6e9:  # SHF低频段
                freq_sensitivity = 0.4
            else:  # SHF高频段
                freq_sensitivity = 0.3
            
            # 带宽影响（带宽越大受干扰面越广）
            bw_factor = 1.0 + 0.3 * (bandwidth / 50e6) if bandwidth > 0 else 1.0
            
            # 综合环境干扰
            env_interference = (0.4 * sea_effect + 0.4 * emi_effect + 0.2 * freq_sensitivity) * bw_factor
            
            return self.stability_manager.clamp_float(env_interference, 0.1, 2.0)
            
        except Exception as e:
            logger.warning(f"单链路环境干扰计算失败：{str(e)}")
            return 0.5

    def _calculate_interference_resistance(self, params_list: List[Dict]) -> float:
        """计算抗干扰能力加分"""
        try:
            total_resistance = 0
            
            for params in params_list:
                modulation = params.get('modulation', 'BPSK')
                polarization = params.get('polarization', 'LINEAR')
                
                # 调制方式抗干扰能力
                mod_resistance = {
                    'BPSK': 0.8,   # BPSK抗干扰最强
                    'QPSK': 0.6,
                    'QAM16': 0.4,
                    'QAM64': 0.2
                }.get(modulation, 0.5)
                
                # 极化方式抗干扰能力
                pol_resistance = {
                    'LINEAR': 0.3,
                    'CIRCULAR': 0.5,
                    'DUAL': 0.7,
                    'ADAPTIVE': 0.9
                }.get(polarization, 0.4)
                
                total_resistance += mod_resistance + pol_resistance
            
            return self.stability_manager.clamp_float(total_resistance, 0, 3.0)
            
        except Exception as e:
            logger.warning(f"抗干扰能力计算失败：{str(e)}")
            return 1.0

    def adaptability_objective(self, params_list: List[Dict]) -> float:
        """
        环境适应性目标函数 - 改进版
        """
        try:
            if not params_list:
                return self._get_worst_objective('adaptability')
            
            total_adaptability = 0
            links = self.task_data.get('communication_links', [])
            
            for i, params in enumerate(params_list):
                link = links[i] if i < len(links) else {}
                
                # 频率适应性
                freq_adapt = self._calculate_frequency_adaptability_robust(
                    params.get('frequency', 0), self.sea_state
                )
                
                # 功率适应性
                power_adapt = self._calculate_power_adaptability_robust(
                    params.get('power', 0), self.emi_intensity
                )
                
                # 调制适应性
                mod_adapt = self._calculate_modulation_adaptability_robust(
                    params.get('modulation', 'BPSK'), self.sea_state
                )
                
                # 极化适应性
                pol_adapt = self._calculate_polarization_adaptability(
                    params.get('polarization', 'LINEAR')
                )
                
                # 综合适应性
                link_adaptability = (0.3 * freq_adapt + 0.3 * power_adapt + 
                                   0.25 * mod_adapt + 0.15 * pol_adapt)
                
                # 考虑链路重要性
                importance = self._get_link_importance(link)
                total_adaptability += link_adaptability * importance
            
            # 标准化到目标范围
            normalized_adaptability = self._normalize_objective(
                total_adaptability, 'adaptability'
            )
            
            return -normalized_adaptability  # 负值用于最小化
            
        except Exception as e:
            logger.error(f"环境适应性目标函数计算异常：{str(e)}")
            return self._get_worst_objective('adaptability')

    def _calculate_frequency_adaptability_robust(self, freq: float, sea_state: float) -> float:
        """改进的频率适应性计算"""
        try:
            freq = self._parse_numeric_value(freq)
            
            if sea_state >= 6:  # 恶劣海况
                if freq < 500e6:  # HF/VHF
                    return 0.9
                elif freq < 2e9:  # UHF
                    return 0.7
                elif freq < 6e9:  # SHF低
                    return 0.5
                else:  # SHF高
                    return 0.3
            else:  # 良好海况
                if freq < 500e6:
                    return 0.7
                elif freq < 2e9:
                    return 0.8
                elif freq < 6e9:
                    return 0.9
                else:
                    return 0.7
                    
        except Exception:
            return 0.5

    def _calculate_power_adaptability_robust(self, power: float, emi: float) -> float:
        """改进的功率适应性计算"""
        try:
            power = self._parse_numeric_value(power)
            
            # 功率与EMI的适应性关系
            if emi > 0.7:  # 高干扰环境
                power_threshold = 30.0
                if power >= power_threshold:
                    return 0.9
                else:
                    return 0.4 + 0.5 * (power / power_threshold)
            else:  # 低干扰环境
                # 适中功率最佳
                optimal_power = 20.0
                deviation = abs(power - optimal_power) / optimal_power
                return max(0.3, 0.9 - 0.6 * deviation)
                
        except Exception:
            return 0.5

    def _calculate_modulation_adaptability_robust(self, modulation: str, sea_state: float) -> float:
        """改进的调制适应性计算"""
        try:
            if sea_state >= 6:  # 恶劣海况
                adaptability_map = {
                    'BPSK': 0.9,
                    'QPSK': 0.7,
                    'QAM16': 0.4,
                    'QAM64': 0.2
                }
            else:  # 良好海况
                adaptability_map = {
                    'BPSK': 0.6,
                    'QPSK': 0.8,
                    'QAM16': 0.9,
                    'QAM64': 0.7
                }
            
            return adaptability_map.get(modulation, 0.5)
            
        except Exception:
            return 0.5

    def _calculate_polarization_adaptability(self, polarization: str) -> float:
        """极化方式适应性计算"""
        try:
            adaptability_map = {
                'LINEAR': 0.6,
                'CIRCULAR': 0.7,
                'DUAL': 0.8,
                'ADAPTIVE': 0.9
            }
            return adaptability_map.get(polarization, 0.6)
            
        except Exception:
            return 0.6

    # 辅助方法

    def _calculate_snr_robust(self, params: Dict, link: Dict) -> float:
        """鲁棒的SNR计算"""
        try:
            frequency = self._parse_numeric_value(params.get('frequency', 0))
            power = self._parse_numeric_value(params.get('power', 0))
            
            # 默认距离和天线高度
            distance = 100.0
            antenna_height = 10.0
            
            # 尝试从链路信息获取更准确的参数
            if link:
                if 'distance' in link:
                    distance = self._parse_numeric_value(link['distance'])
                if 'antenna_height' in link:
                    antenna_height = self._parse_numeric_value(link['antenna_height'])
            
            # 构建链路参数
            link_params = {
                'distance': distance,
                'antenna_height': antenna_height
            }
            
            # 计算噪声和路径损耗
            if self.noise_model:
                try:
                    noise = self.noise_model.calculate_total_noise(frequency, link_params)
                    path_loss = self.noise_model.calculate_propagation_loss(
                        frequency, distance, 
                        self._parse_numeric_value(self.env_data.get('depth', 0))
                    )
                except Exception as e:
                    logger.warning(f"噪声模型计算失败：{str(e)}，使用简化模型")
                    noise = -90.0  # 默认噪声
                    path_loss = 100.0  # 默认路径损耗
            else:
                noise = -90.0
                path_loss = 100.0
            
            # SNR计算
            power_dbw = 10 * self.stability_manager.safe_log(power, 10, -10)
            snr = power_dbw - path_loss - noise
            
            return self.stability_manager.clamp_float(snr, -20, 40)
            
        except Exception as e:
            logger.warning(f"SNR计算失败：{str(e)}")
            return 10.0  # 默认SNR

    def _calculate_environment_reliability_factor(self) -> float:
        """计算环境对可靠性的影响因子"""
        try:
            # 海况影响（海况越高可靠性越低）
            sea_factor = max(0.3, 1.0 - (self.sea_state - 1) / 8)
            
            # EMI影响
            emi_factor = max(0.4, 1.0 - self.emi_intensity)
            
            # 综合环境因子
            env_factor = 0.6 * sea_factor + 0.4 * emi_factor
            
            return self.stability_manager.clamp_float(env_factor, 0.2, 1.0)
            
        except Exception:
            return 0.7

    def _calculate_frequency_overlap(self, freq1: float, bw1: float, 
                                   freq2: float, bw2: float) -> float:
        """计算频率重叠度"""
        try:
            # 频段边界
            low1 = freq1 - bw1/2
            high1 = freq1 + bw1/2
            low2 = freq2 - bw2/2
            high2 = freq2 + bw2/2
            
            # 重叠计算
            overlap = max(0, min(high1, high2) - max(low1, low2))
            
            if overlap > 0:
                return self.stability_manager.safe_divide(
                    overlap, min(bw1, bw2), 0
                )
            return 0.0
            
        except Exception:
            return 0.0

    def _get_link_importance(self, link: Dict) -> float:
        """获取链路重要性权重"""
        try:
            importance = 1.0
            
            # 根据链路类型调整
            link_type = link.get('comm_type', '').lower()
            if 'data_link' in link_type or '数据链' in link_type:
                importance = 1.5
            elif 'satellite' in link_type or '卫星' in link_type:
                importance = 1.8
            elif 'hf' in link_type or '短波' in link_type:
                importance = 1.2
            
            # 指挥舰链路额外加权
            if self._is_command_ship_link(link):
                importance *= 1.3
            
            return self.stability_manager.clamp_float(importance, 0.5, 2.0)
            
        except Exception:
            return 1.0

    def _is_command_ship_link(self, link: Dict) -> bool:
        """判断是否是指挥舰链路"""
        try:
            if not self.task_data.get('nodes', {}).get('command_ship'):
                return False
            
            command_ship_id = self.task_data['nodes']['command_ship'].get('identity')
            return (link.get('source_id') == command_ship_id or 
                   link.get('target_id') == command_ship_id)
        except Exception:
            return False

    def _normalize_objective(self, value: float, objective_name: str) -> float:
        """标准化目标函数值"""
        try:
            scale_info = self.objective_scales.get(objective_name, {
                'min': 0.0, 'max': 1.0, 'target': 'maximize'
            })
            
            min_val = scale_info['min']
            max_val = scale_info['max']
            
            # 线性标准化到[0, 1]
            if max_val > min_val:
                normalized = (value - min_val) / (max_val - min_val)
            else:
                normalized = 0.5
            
            return self.stability_manager.clamp_float(normalized, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"目标函数标准化失败：{str(e)}")
            return 0.5

    def _get_worst_objective(self, objective_name: str) -> float:
        """获取最差的目标函数值"""
        scale_info = self.objective_scales.get(objective_name, {})
        
        if scale_info.get('target') == 'maximize':
            return -scale_info.get('max', 1.0)  # 最小化时用负的最大值
        else:
            return scale_info.get('max', 100.0)  # 最小化时用最大值

    def _parse_numeric_value(self, value) -> float:
        """解析数值，包含异常处理"""
        try:
            if value is None:
                return 0.0
                
            if isinstance(value, (int, float)):
                result = float(value)
            elif isinstance(value, str):
                # 提取数字
                import re
                numeric_match = re.search(r'-?\d+\.?\d*', value)
                if numeric_match:
                    result = float(numeric_match.group())
                else:
                    result = 0.0
            else:
                result = 0.0
            
            # 数值有效性检查
            if not np.isfinite(result):
                return 0.0
                
            return result
            
        except Exception:
            return 0.0


# 向后兼容的包装器
class ObjectiveFunction(ImprovedObjectiveFunction):
    """向后兼容的目标函数类"""
    
    def __init__(self, task_data: Dict, env_data: Dict, constraint_data: Dict, config=None):
        super().__init__(task_data, env_data, constraint_data, config)
        logger.info("使用改进的目标函数（兼容性模式）")