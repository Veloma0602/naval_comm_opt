import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import random

class PopulationManager:
    def __init__(self, config, neo4j_handler=None):
        """
        初始化种群管理器
        
        参数:
        config: 优化配置
        neo4j_handler: Neo4j数据库处理器，用于获取历史案例
        """
        self.config = config
        self.neo4j_handler = neo4j_handler        

    def solution_to_parameters(self, solution: np.ndarray, n_links: int) -> List[Dict[str, Any]]:
        """
        将解向量转换为通信参数字典列表
        
        参数:
        solution: 解向量
        n_links: 通信链路数量
        
        返回:
        参数字典列表
        """
        params_list = []
        try:
            # 每个链路需要5个参数，重塑解向量
            solution_reshaped = solution.reshape(-1)  # 先展平为1维
            
            # 确保解向量长度至少为n_links*5
            if len(solution_reshaped) < n_links * 5:
                # 如果解向量不够长，使用默认值填充
                padding = np.array([4e9, 20e6, 10, 0, 0] * (n_links - len(solution_reshaped) // 5))
                solution_reshaped = np.concatenate([solution_reshaped, padding])
                
            # 重新塑造为正确的维度
            solution_reshaped = solution_reshaped[:n_links*5].reshape(n_links, 5)
            
            for i in range(n_links):
                if i < len(solution_reshaped):
                    # 应用参数约束确保合理值
                    freq = self._constrain_param(solution_reshaped[i, 0], self.config.freq_min, self.config.freq_max)
                    bandwidth = self._constrain_param(solution_reshaped[i, 1], self.config.bandwidth_min, self.config.bandwidth_max)
                    power = self._constrain_param(solution_reshaped[i, 2], self.config.power_min, self.config.power_max)
                    
                    params = {
                        'frequency': freq,
                        'bandwidth': bandwidth,
                        'power': power,
                        'modulation': self.index_to_modulation(solution_reshaped[i, 3]),
                        'polarization': self.index_to_polarization(solution_reshaped[i, 4])
                    }
                    params_list.append(params)
        except Exception as e:
            print(f"解向量转换失败: {str(e)}")
            # 如果转换失败，创建默认参数
            for i in range(n_links):
                params = {
                    'frequency': 4e9,  # 4 GHz
                    'bandwidth': 20e6,  # 20 MHz
                    'power': 10,       # 10 W
                    'modulation': 'BPSK',
                    'polarization': 'LINEAR'
                }
                params_list.append(params)
        
        return params_list
    
    def _constrain_param(self, value: float, min_val: float, max_val: float) -> float:
        """
        将参数值约束在指定范围内
        
        参数:
        value: 参数值
        min_val: 最小允许值
        max_val: 最大允许值
        
        返回:
        约束后的参数值
        """
        if value < min_val or value > max_val:
            # 如果大幅超出范围，使用合理的默认值
            if value < min_val * 0.5 or value > max_val * 2:
                return (min_val + max_val) / 2
            # 否则剪裁到边界
            return max(min_val, min(value, max_val))
        return value

    def initialize_population(self, task_id: str, n_var: int, xl: np.ndarray, 
                         xu: np.ndarray, n_samples: int) -> np.ndarray:
        """
        初始化种群，融合历史案例和随机生成的解，增强多样性
        
        参数:
        task_id: 当前任务ID
        n_var: 问题维度
        xl: 下界
        xu: 上界
        n_samples: 种群大小
        
        返回:
        初始化的种群
        """
        population = []
        n_links = n_var // 5
        
        # 1. 获取相似历史案例
        if self.neo4j_handler:
            try:
                historical_cases = self.neo4j_handler.get_similar_cases(
                    task_id, 
                    limit=min(n_samples // 2, self.config.max_historical_cases)
                )
                print(f"找到 {len(historical_cases)} 个相似历史案例用于初始化种群")
            except Exception as e:
                print(f"获取相似历史案例时出错: {str(e)}")
                historical_cases = []
        else:
            historical_cases = []
        
        # 2. 从历史案例提取参数并添加扰动
        historical_solutions = []
        for case_id in historical_cases:
            try:
                # 2.1 获取该任务的通信参数
                communication_params = []
                if self.neo4j_handler:
                    communication_params = self.neo4j_handler.get_historical_communication_parameters(case_id)
                
                if communication_params:
                    # 2.2 转换为解向量
                    solution = self._parameters_to_solution(communication_params)
                    
                    # 2.3 添加扰动，增加多样性
                    # 每个参数有20%概率被扰动，扰动范围为±20%
                    mask = np.random.random(len(solution)) < 0.4
                    perturbation = np.random.uniform(0.8, 1.2, len(solution))
                    solution = np.where(mask, solution * perturbation, solution)
                    
                    # 2.4 调整解向量维度
                    if len(solution) > n_var:
                        # 截断
                        solution = solution[:n_var]
                    elif len(solution) < n_var:
                        # 扩展（用随机值填充）
                        padding = np.random.uniform(
                            low=xl[len(solution):],
                            high=xu[len(solution):],
                            size=n_var - len(solution)
                        )
                        solution = np.concatenate([solution, padding])
                    
                    # 2.5 确保边界约束
                    solution = np.clip(solution, xl, xu)
                    
                    # 2.6 验证解的有效性 
                    if self.validate_solution(solution, xl, xu):
                        historical_solutions.append(solution)
                        # 如果这是一个高质量解，生成其变体以增加多样性
                        if communication_params[0].get('link_type', '') in ['卫星通信', '数据链通信']:
                            # 为重要链路类型生成多个变体
                            for _ in range(3):
                                variant = solution.copy()
                                # 更大扰动
                                mask = np.random.random(len(variant)) < 0.5
                                perturbation = np.random.uniform(0.6, 1.4, len(variant))
                                variant = np.where(mask, variant * perturbation, variant)
                                variant = np.clip(variant, xl, xu)
                                if self.validate_solution(variant, xl, xu):
                                    historical_solutions.append(variant)
            except Exception as e:
                print(f"处理历史案例 {case_id} 时出错: {str(e)}")
        # 3. 添加频率分布策略解
        freq_strategies = [
            # 低频-高频均匀分布
            lambda: [self.config.freq_min + i * (self.config.freq_max - self.config.freq_min) / (n_links - 1) 
                    for i in range(n_links)],
            # 聚类频率分配 (在几个频段内聚类)
            lambda: [random.choice([500e6, 1e9, 5e9, 9e9]) + random.uniform(-100e6, 100e6) 
                    for _ in range(n_links)],
            # 基于链路指数的频率分配
            lambda: [self.config.freq_min * (self.config.freq_max/self.config.freq_min)**(i/(n_links-1))
                    for i in range(n_links)],
            # 双峰分布 (一部分在低频，一部分在高频)
            lambda: [random.choice([random.uniform(self.config.freq_min, 1e9), 
                                    random.uniform(5e9, self.config.freq_max)])
                    for _ in range(n_links)]
        ]
        
        # 4. 添加带宽分配策略
        bw_strategies = [
            # 均匀带宽
            lambda: [self.config.bandwidth_min + 0.5 * (self.config.bandwidth_max - self.config.bandwidth_min)
                    for _ in range(n_links)],
            # 与频率相关带宽 (高频更大带宽)
            lambda freq: [(self.config.bandwidth_min + 
                        (f - self.config.freq_min) / (self.config.freq_max - self.config.freq_min) * 
                        (self.config.bandwidth_max - self.config.bandwidth_min))
                        for f in freq],
            # 随机但有序带宽
            lambda: sorted([random.uniform(self.config.bandwidth_min, self.config.bandwidth_max) 
                        for _ in range(n_links)])
        ]
        
        # 创建10-20个策略解
        n_strategy_solutions = min(20, n_samples // 5)
        for _ in range(n_strategy_solutions):
            # 选择频率策略
            freq_strategy = random.choice(freq_strategies)
            freq_values = freq_strategy()
            
            # 选择带宽策略
            bw_strategy = random.choice(bw_strategies)
            if random.random() < 0.7:  # 70%概率使用与频率相关的带宽
                bw_values = bw_strategy(freq_values)
            else:
                bw_values = bw_strategy()
            
            # 创建解向量
            solution = np.zeros(n_var)
            
            # 填充频率
            solution[:n_links] = freq_values
            
            # 填充带宽
            solution[n_links:2*n_links] = bw_values
            
            # 填充功率 (与频率相关)
            for i in range(n_links):
                freq = freq_values[i]
                if freq < 1e9:
                    # 低频使用较高功率
                    solution[2*n_links + i] = random.uniform(self.config.power_max*0.6, self.config.power_max)
                else:
                    # 高频使用中等功率
                    solution[2*n_links + i] = random.uniform(self.config.power_min, self.config.power_max*0.7)
            
            # 填充调制方式 (小带宽低阶调制，大带宽高阶调制)
            for i in range(n_links):
                bw = bw_values[i]
                if bw < 0.3 * self.config.bandwidth_max:
                    solution[3*n_links + i] = random.randint(0, 1)  # BPSK或QPSK
                else:
                    solution[3*n_links + i] = random.randint(1, 3)  # QPSK, QAM16或QAM64
            
            # 填充极化方式 (随机)
            solution[4*n_links:] = [random.randint(0, 3) for _ in range(n_links)]
            
            # 确保解在边界内
            solution = np.clip(solution, xl, xu)
            
            # 添加到种群
            population.append(solution)
        # 在添加策略解后
        extreme_solutions = population.copy()  # 将策略解也视为极端解

        # 5. 使用拉丁超立方抽样(LHS)生成剩余个体，确保参数空间覆盖
        n_missing = n_samples - len(historical_solutions) - len(extreme_solutions)
        if n_missing > 0:
            try:
                from scipy.stats.qmc import LatinHypercube
                print(f"使用拉丁超立方抽样生成 {n_missing} 个多样化个体")
                
                # 4.1 初始化LHS采样器
                sampler = LatinHypercube(d=n_var)
                samples = sampler.random(n=n_missing)
                
                # 4.2 将[0,1]范围映射到参数实际范围
                random_solutions = xl + samples * (xu - xl)
                
                # 4.3 进行频段分配优化
                n_links = n_var // 5  # 每个链路5个参数
                
                # 频率参数优化 - 在不同频段分配频率，避免拥挤
                freq_bands = [
                    (100e6, 500e6),    # 低频段
                    (500e6, 1.5e9),    # 中低频段
                    (1.5e9, 3e9),      # 中频段
                    (3e9, 5e9)         # 高频段
                ]
                
                for i in range(n_missing):
                    solution = random_solutions[i]
                    
                    # 随机选择频段分配策略
                    if random.random() < 0.7:  # 70%概率使用分散频段策略
                        # 打乱频段顺序
                        bands = freq_bands.copy()
                        random.shuffle(bands)
                        
                        # 为每个链路分配不同频段
                        for j in range(min(n_links, len(bands))):
                            band = bands[j % len(bands)]
                            # 在频段内随机选择频率
                            solution[j] = random.uniform(band[0], band[1])
                    
                    # 优化带宽分配 - 考虑频率和应用场景
                    for j in range(n_links):
                        freq = solution[j]
                        # 根据频率调整带宽
                        if freq < 500e6:
                            # 低频通常使用较窄带宽
                            solution[n_links + j] = random.uniform(5e6, 15e6)
                        elif freq < 2e9:
                            # 中频使用中等带宽
                            solution[n_links + j] = random.uniform(10e6, 30e6)
                        else:
                            # 高频可使用较宽带宽
                            solution[n_links + j] = random.uniform(20e6, 50e6)
                    
                    # 优化功率分配 - 与频率相关
                    for j in range(n_links):
                        freq = solution[j]
                        if freq < 500e6:
                            # 低频通常需要较大功率
                            solution[2*n_links + j] = random.uniform(25, 50)
                        elif freq < 2e9:
                            # 中频使用中等功率
                            solution[2*n_links + j] = random.uniform(15, 40)
                        else:
                            # 高频可使用较低功率
                            solution[2*n_links + j] = random.uniform(5, 30)
                    
                    # 确保参数在边界内
                    random_solutions[i] = np.clip(solution, xl, xu)
                
                # 4.4 添加到种群
                for solution in random_solutions:
                    if self.validate_solution(solution, xl, xu):
                        population.append(solution)
                    else:
                        # 如果验证失败，生成一个简单的有效解
                        simple_solution = self._generate_simple_valid_solution(n_var, xl, xu)
                        population.append(simple_solution)
                
            except ImportError:
                print("scipy.stats.qmc不可用，使用普通随机采样")
                # 使用普通随机采样
                for _ in range(n_missing):
                    solution = np.random.uniform(low=xl, high=xu)
                    population.append(solution)
            except Exception as e:
                print(f"生成随机解时出错: {str(e)}")
                # 简单随机生成
                for _ in range(n_missing):
                    solution = xl + np.random.random(n_var) * (xu - xl)
                    population.append(solution)
        
        # 4. 添加历史解并确保种群大小正确
        population.extend(historical_solutions)
        
        # 如果种群超出所需大小，随机选择n_samples个
        if len(population) > n_samples:
            population = random.sample(population, n_samples)
        
        # 如果种群不足所需大小，添加随机解
        while len(population) < n_samples:
            solution = xl + np.random.random(n_var) * (xu - xl)
            population.append(solution)
        
        print(f"种群初始化完成，大小: {len(population)}, 包含 {len(historical_solutions)} 个历史案例解")
        
        return np.array(population)

    def _generate_simple_valid_solution(self, n_var, xl, xu):
        """生成一个简单的有效解，用于验证失败时的回退方案"""
        n_links = n_var // 5
        
        # 创建基本解向量
        solution = np.zeros(n_var)
        
        # 分配不同频段的频率
        freq_step = (xu[0] - xl[0]) / (n_links + 1)
        for i in range(n_links):
            solution[i] = xl[0] + freq_step * (i + 1)  # 均匀分布频率
        
        # 分配中等带宽
        for i in range(n_links):
            solution[n_links + i] = (xl[n_links] + xu[n_links]) / 2
        
        # 分配中等功率
        for i in range(n_links):
            solution[2*n_links + i] = (xl[2*n_links] + xu[2*n_links]) / 2
        
        # 使用简单的调制和极化方式
        for i in range(n_links):
            solution[3*n_links + i] = 0  # BPSK
            solution[4*n_links + i] = 0  # LINEAR
        
        return solution
    
    def generate_random_solutions(self, n_solutions: int, problem_size: int,
                             lower_bounds: np.ndarray, upper_bounds: np.ndarray) -> List[np.ndarray]:
        """
        生成随机解 - 改进版
        
        参数:
        n_solutions: 需要生成的解的数量
        problem_size: 问题维度
        lower_bounds: 下界
        upper_bounds: 上界
        
        返回:
        随机生成的解列表
        """
        solutions = []
        n_links = problem_size // 5  # 每个链路5个参数
        
        # 常用频段和带宽配置
        freq_bands = [
            (400e6, 500e6),    # UHF频段
            (1.5e9, 1.6e9),    # L频段
            (2.3e9, 2.5e9),    # S频段
            (3.4e9, 3.8e9),    # C频段
            (4.8e9, 5.2e9)     # C频段
        ]
        
        bandwidth_options = [5e6, 10e6, 20e6, 40e6, 50e6]  # 常用带宽选项
        
        for _ in range(n_solutions):
            # 为每个链路生成合理参数
            solution = []
            
            # 确保不同链路使用不同频段
            available_bands = freq_bands.copy()
            random.shuffle(available_bands)
            
            for i in range(n_links):
                # 1. 频率 - 从可用频段中选择
                if i < len(available_bands):
                    band = available_bands[i]
                    freq = random.uniform(band[0], band[1])
                else:
                    # 如果链路数量大于预设频段数量，则随机选择频段
                    band = random.choice(freq_bands)
                    freq = random.uniform(band[0], band[1])
                solution.append(freq)
                
                # 2. 带宽 - 根据频率选择合适的带宽
                if freq < 1e9:
                    bw = random.choice(bandwidth_options[:2])  # 低频段使用较小带宽
                elif freq < 3e9:
                    bw = random.choice(bandwidth_options[1:3])  # 中频段使用中等带宽
                else:
                    bw = random.choice(bandwidth_options[2:])  # 高频段使用较大带宽
                solution.append(bw)
                
                # 3. 功率 - 根据频率选择合适的功率
                if freq < 1e9:
                    power = random.uniform(30, 60)  # 低频段使用较大功率
                elif freq < 3e9:
                    power = random.uniform(20, 40)  # 中频段使用中等功率
                else:
                    power = random.uniform(5, 25)   # 高频段使用较小功率
                solution.append(power)
                
                # 4. 调制方式 - 根据频率和带宽选择合适的调制方式
                if freq < 1e9 or bw < 10e6:
                    mod = random.randint(0, 1)  # 低频/小带宽使用简单调制(BPSK/QPSK)
                else:
                    mod = random.randint(0, 3)  # 高频/大带宽可使用复杂调制
                solution.append(mod)
                
                # 5. 极化方式 - 根据应用场景选择
                if freq < 1e9:
                    pol = 0  # 低频段常用线性极化
                elif freq > 3e9:
                    pol = 1  # 高频段常用圆极化
                else:
                    pol = random.randint(0, 3)  # 中频段可使用各种极化
                solution.append(pol)
            
            # 将解转换为numpy数组
            sol_array = np.array(solution)
            
            # 检查解是否在边界内，如果不在则截断
            sol_array = np.clip(sol_array, lower_bounds, upper_bounds)
            
            solutions.append(sol_array)
        
        return solutions
    
    def validate_solution(self, solution: np.ndarray,
                     lower_bounds: np.ndarray, upper_bounds: np.ndarray) -> bool:
        """
        验证解的有效性 - 改进版
        
        参数:
        solution: 待验证的解
        lower_bounds: 下界
        upper_bounds: 上界
        
        返回:
        解是否有效
        """
        try:
            # 检查基本边界约束
            if np.any(solution < lower_bounds) or np.any(solution > upper_bounds):
                return False
            
            # 检查频率间隔与带宽关系
            # 每个链路需要5个参数，前n_links个元素表示频率
            n_links = len(solution) // 5
            
            # 获取所有链路的频率和带宽
            frequencies = solution[:n_links]
            bandwidths = solution[n_links:2*n_links]
            
            # 检查频率间隔是否满足要求
            for i in range(n_links):
                for j in range(i+1, n_links):
                    # 跳过5GHz左右的链路间隔检查 - 它们通常是不同类型的系统
                    if abs(frequencies[i] - 5e9) < 1e8 or abs(frequencies[j] - 5e9) < 1e8:
                        continue
                        
                    # 计算频率中心点间隔
                    spacing = abs(frequencies[i] - frequencies[j])
                    
                    # 计算频段边界间隔
                    edge_spacing = spacing - (bandwidths[i]/2 + bandwidths[j]/2)
                    
                    # 最小频率间隔（防止链路间干扰）- 降低要求以增加多样性
                    # 只对最小间隔进行严格要求
                    min_spacing = max(bandwidths[i], bandwidths[j]) * 0.8
                    
                    if edge_spacing < min_spacing:
                        return False
            
            # 全部检查通过
            return True
            
        except Exception as e:
            print(f"解验证出错: {str(e)}")
            return False

    def convert_case_to_solution(self, case_data: Dict[str, Any], problem_size: int) -> Optional[np.ndarray]:
        """
        将历史案例转换为解向量
        
        参数:
        case_data: 历史案例数据
        problem_size: 问题维度
        
        返回:
        解向量或None（如果转换失败）
        """
        try:
            # 案例中应该包含通信链路的参数
            solution = []
            comm_links = case_data.get('communication_links', [])
            
            for link in comm_links:
                # 提取每个链路的参数
                freq = float(link.get('frequency', 0))
                bandwidth = float(link.get('bandwidth', 0))
                power = float(link.get('power', 0))
                modulation = self.modulation_to_index(link.get('modulation', ''))
                polarization = self.polarization_to_index(link.get('polarization', ''))
                
                solution.extend([freq, bandwidth, power, modulation, polarization])
            
            # 检查维度是否匹配
            if len(solution) != problem_size:
                print(f"警告: 解向量维度不匹配 (实际 {len(solution)}, 预期 {problem_size})")
                return None
            
            return np.array(solution)
            
        except Exception as e:
            print(f"转换案例到解向量时出错: {str(e)}")
            return None
    
    def modulation_to_index(self, modulation: str) -> float:
        """将调制方式转换为数值索引"""
        modulation_map = {
            'BPSK': 0.0,
            'QPSK': 1.0,
            'QAM16': 2.0,
            'QAM64': 3.0
        }
        return modulation_map.get(modulation.upper(), 0.0)
    
    def polarization_to_index(self, polarization: str) -> float:
        """将极化方式转换为数值索引"""
        polarization_map = {
            'LINEAR': 0.0,
            'CIRCULAR': 1.0,
            'DUAL': 2.0,
            'ADAPTIVE': 3.0
        }
        return polarization_map.get(polarization.upper(), 0.0)
    
    def index_to_modulation(self, index: float) -> str:
        """将数值索引转换为调制方式"""
        index_map = {
            0: 'BPSK',
            1: 'QPSK',
            2: 'QAM16',
            3: 'QAM64'
        }
        return index_map.get(int(round(index)), 'BPSK')
    
    def index_to_polarization(self, index: float) -> str:
        """将数值索引转换为极化方式"""
        index_map = {
            0: 'LINEAR',
            1: 'CIRCULAR',
            2: 'DUAL',
            3: 'ADAPTIVE'
        }
        return index_map.get(int(round(index)), 'LINEAR')
    
    def _parameters_to_solution(self, parameters: List[Dict]) -> np.ndarray:
        """
        将通信参数列表转换为解向量
        """
        solution = []
        
        for params in parameters:
            # 添加频率
            solution.append(params.get('frequency', 0))
            
            # 添加带宽
            solution.append(params.get('bandwidth', 0))
            
            # 添加功率
            solution.append(params.get('power', 0))
            
            # 添加调制方式(转换为数值索引)
            modulation = params.get('modulation', 'BPSK')
            solution.append(self.modulation_to_index(modulation))
            
            # 添加极化方式(转换为数值索引)
            polarization = params.get('polarization', 'LINEAR')
            solution.append(self.polarization_to_index(polarization))
        
        return np.array(solution)