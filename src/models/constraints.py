import numpy as np
from typing import List, Dict, Any

class Constraints:
    def __init__(self, config):
        """
        初始化约束条件处理器
        
        参数:
        config: 优化配置
        """
        self.config = config
    
    def evaluate_constraints(self, params_list: List[Dict]) -> np.ndarray:
        """
        评估所有约束条件
        
        参数:
        params_list: 参数字典列表
        
        返回:
        约束值数组，负值表示满足约束，正值表示违反约束
        """
        constraints = []

        # 计算通信链路数量
        n_links = len(params_list)
        
        # 为不同链路分配重要性权重 (所有链路权重先设为1)
        link_importance = [1.0] * n_links  

        # 频率约束 - 更严格的边界约束
        for i, params in enumerate(params_list):
            freq = params.get('frequency', 0)
            # 频率下界约束 - 更严格的惩罚
            if freq < self.config.freq_min:
                c1 = (self.config.freq_min - freq) / self.config.freq_min * 2.0
            else:
                c1 = -1.0
            
            # 频率上界约束
            if freq > self.config.freq_max:
                c2 = (freq - self.config.freq_max) / self.config.freq_max * 2.0
            else:
                c2 = -1.0
            
            # 应用链路重要性权重
            importance = link_importance[i]
            constraints.extend([c1 * importance, c2 * importance])

        # 频率间隔约束 - 使用简化的最小间隔
        for i in range(n_links):
            for j in range(i+1, n_links):
                freq_i = params_list[i].get('frequency', 0)
                bw_i = params_list[i].get('bandwidth', 0)
                
                freq_j = params_list[j].get('frequency', 0)
                bw_j = params_list[j].get('bandwidth', 0)
                
                # 简化的最小间隔计算 - 使用最大带宽的倍数
                min_spacing = max(bw_i, bw_j) * 1.2
                
                # 计算边界间隔
                spacing = abs(freq_i - freq_j)
                edge_spacing = spacing - (bw_i/2 + bw_j/2)
                
                # 严格的约束实现
                if edge_spacing < min_spacing:
                    violation = (min_spacing - edge_spacing) / min_spacing
                    constraints.append(max(-0.5, violation * 3.0))  # 更强的惩罚
                else:
                    constraints.append(-1.0)
        
        # 功率约束 - 只保留上下界约束
        for params in params_list:
            power = params.get('power', 0)
        
            # 功率下界约束
            c1 = max(-1.0, (self.config.power_min - power) / self.config.power_min)
            
            # 功率上界约束
            c2 = max(-1.0, (power - self.config.power_max) / self.config.power_max)
            
            constraints.extend([c1, c2])
        
        # 带宽约束
        bandwidth_constraints = self.bandwidth_constraints(params_list)
        constraints.extend(bandwidth_constraints)
        
        # 频率间隔约束
        spacing_constraints = self.frequency_spacing_constraints(params_list)
        constraints.extend(spacing_constraints)
        
        # 信噪比约束
        snr_constraints = self.snr_constraints(params_list)
        constraints.extend(snr_constraints)
        
        # 时延约束
        delay_constraints = self.delay_constraints(params_list)
        constraints.extend(delay_constraints)
        
        # 计算应该返回的约束数量
        expected_constraints = n_links * 4
        
        # 如果约束数量不匹配，调整到预期的数量
        if len(constraints) < expected_constraints:
            # 填充满足的约束
            padding = [-1.0] * (expected_constraints - len(constraints))
            constraints.extend(padding)
        elif len(constraints) > expected_constraints:
            # 截断多余的约束
            constraints = constraints[:expected_constraints]
        
        return np.array(constraints)
    
    def frequency_constraints(self, params_list: List[Dict]) -> List[float]:
        """
        频率约束
        
        参数:
        params_list: 参数字典列表
        
        返回:
        约束值列表
        """
        constraints = []
        for params in params_list:
            freq = params.get('frequency', 0)
            
            # 频率下界约束
            c1 = self.config.freq_min - freq
            
            # 频率上界约束
            c2 = freq - self.config.freq_max
            
            constraints.extend([c1, c2])
        
        return constraints
    
    def power_constraints(self, params_list: List[Dict]) -> List[float]:
        """
        功率约束
        
        参数:
        params_list: 参数字典列表
        
        返回:
        约束值列表
        """
        constraints = []
        for params in params_list:
            power = params.get('power', 0)
            
            # 功率下界约束
            c1 = self.config.power_min - power
            
            # 功率上界约束
            c2 = power - self.config.power_max
            
            constraints.extend([c1, c2])
        
        return constraints
    
    def bandwidth_constraints(self, params_list: List[Dict]) -> List[float]:
        """
        带宽约束
        
        参数:
        params_list: 参数字典列表
        
        返回:
        约束值列表
        """
        constraints = []
        for params in params_list:
            bandwidth = params.get('bandwidth', 0)
            
            # 带宽下界约束
            c1 = self.config.bandwidth_min - bandwidth
            
            # 带宽上界约束
            c2 = bandwidth - self.config.bandwidth_max
            
            constraints.extend([c1, c2])
        
        return constraints
    
    def frequency_spacing_constraints(self, params_list: List[Dict]) -> List[float]:
        """
        频率间隔约束
        
        参数:
        params_list: 链路参数字典列表
        
        返回:
        约束值列表
        """
        constraints = []
        n = len(params_list)
        
        # 最小频率间隔（防止链路间干扰）
        # 减少最小间隔的要求，从原来的1.2倍降低到1.0倍
        min_spacing = self.config.bandwidth_max * 1.0  
        
        # 检查每对链路的频率间隔
        for i in range(n):
            for j in range(i+1, n):
                freq_i = params_list[i].get('frequency', 0)
                bw_i = params_list[i].get('bandwidth', 0)
                
                freq_j = params_list[j].get('frequency', 0)
                bw_j = params_list[j].get('bandwidth', 0)
                
                # 计算频率中心点间隔
                spacing = abs(freq_i - freq_j)
                
                # 计算两个频段的边界间隔
                edge_spacing = spacing - (bw_i/2 + bw_j/2)
                
                # 修改为软约束 - 当边界间隔不满足要求时才施加惩罚
                if edge_spacing < min_spacing:
                    # 归一化并限制最大约束值
                    c = (min_spacing - edge_spacing) / min_spacing
                    # 限制约束值的大小，避免过大的惩罚
                    c = min(c, 1.0) * 100  # 将约束值限制在0-100范围内
                else:
                    c = 0.0  # 满足约束时，不施加惩罚
                
                constraints.append(c)
        
        return constraints
    
    def snr_constraints(self, params_list: List[Dict]) -> List[float]:
        """
        信噪比约束
        
        参数:
        params_list: 参数字典列表
        
        返回:
        约束值列表
        """
        # 在实际应用中，这里应该包含link信息并计算SNR
        # 这里简化为直接使用参数中的信噪比估计值
        constraints = []
        
        # 最小SNR要求
        min_snr = self.config.snr_min
        
        for params in params_list:
            # 简化处理：通过参数估算SNR
            # 在实际系统中，这里应该调用噪声模型计算实际SNR
            estimated_snr = self._estimate_snr(params)
            
            # 约束: SNR应大于最小要求
            c = min_snr - estimated_snr
            
            constraints.append(c)
        
        return constraints
    
    def delay_constraints(self, params_list: List[Dict]) -> List[float]:
        """
        时延约束
        
        参数:
        params_list: 参数字典列表
        
        返回:
        约束值列表
        """
        constraints = []
        
        # 最大允许时延
        max_delay = self.config.delay_max
        
        for params in params_list:
            # 根据参数估算传输时延
            # 简化处理：假设通信时延与带宽和调制方式有关
            delay = self._estimate_delay(params)
            
            # 约束: 时延应小于最大允许值
            c = delay - max_delay
            
            constraints.append(c)
        
        return constraints
    
    def _estimate_snr(self, params: Dict) -> float:
        """
        估算信噪比
        
        参数:
        params: 通信参数
        
        返回:
        估算的SNR (dB)
        """
        # 简化的SNR估算模型
        power = params.get('power', 0)
        frequency = params.get('frequency', 0)
        
        # 确保功率和频率是数值类型
        if isinstance(power, str):
            power = self._parse_numeric_value(power)
        if isinstance(frequency, str):
            frequency = self._parse_numeric_value(frequency)
        
        # 频率越高，路径损耗越大，SNR越低
        freq_factor = max(0.2, 1 - (frequency / (10 * self.config.freq_max)))
        
        # 功率越大，SNR越高
        power_factor = power / self.config.power_max
        
        # 简化估算
        estimated_snr = 20 * power_factor * freq_factor
        
        return estimated_snr

    def _parse_numeric_value(self, value_str):
        """从可能包含单位的字符串中提取数值"""
        if value_str is None:
            return 0.0
            
        if isinstance(value_str, (int, float)):
            return float(value_str)
        
        if isinstance(value_str, str):
            # 提取数字部分 (包括负号和小数点)
            import re
            numeric_match = re.search(r'-?\d+\.?\d*', value_str)
            if numeric_match:
                try:
                    return float(numeric_match.group())
                except ValueError:
                    pass
        
        # 默认返回0
        return 0.0
    
    def _estimate_delay(self, params: Dict) -> float:
        """
        估算通信时延
        
        参数:
        params: 通信参数
        
        返回:
        估算的时延 (ms)
        """
        # 简化的时延估算模型
        bandwidth = params.get('bandwidth', 0)
        modulation = params.get('modulation', 'BPSK')
        
        # 确保带宽是数值类型
        if isinstance(bandwidth, str):
            bandwidth = self._parse_numeric_value(bandwidth)
        
        # 基础传播时延（假设为10ms）
        base_delay = 10
        
        # 带宽越大，传输时延越小
        bandwidth_factor = self.config.bandwidth_max / max(bandwidth, 1e6)
        
        # 调制方式影响
        modulation_factor = {
            'BPSK': 1.0,
            'QPSK': 0.8,
            'QAM16': 0.6,
            'QAM64': 0.5
        }.get(modulation, 1.0)
        
        # 简化估算
        delay = base_delay * bandwidth_factor * modulation_factor
        
        return delay