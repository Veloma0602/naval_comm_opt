from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import random
import logging

logger = logging.getLogger(__name__)

class ImprovedPopulationManager:
    """
    改进的种群管理器
    主要改进：
    1. 更智能的历史案例利用
    2. 多样化的初始化策略
    3. 领域知识引导
    4. 数值稳定性保护
    5. 自适应种群质量控制
    """
    
    def __init__(self, config, neo4j_handler=None):
        """
        初始化改进的种群管理器
        
        参数:
        config: 优化配置
        neo4j_handler: Neo4j数据库处理器
        """
        self.config = config
        self.neo4j_handler = neo4j_handler
        
        # 数值稳定性保护
        self.EPSILON = 1e-10
        self.MAX_PARAM_VALUE = 1e12
        self.MIN_PARAM_VALUE = -1e12
        
        # 初始化策略权重
        self.init_strategies = {
            'historical_guided': 0.3,    # 历史案例引导
            'domain_knowledge': 0.25,    # 领域知识引导
            'frequency_optimized': 0.2,  # 频率优化策略
            'latin_hypercube': 0.15,     # 拉丁超立方
            'random_diverse': 0.1        # 随机多样化
        }
        
        # 频段分配策略
        self.frequency_bands = [
            {'name': 'HF', 'range': (3e6, 30e6), 'characteristics': 'long_range'},
            {'name': 'VHF', 'range': (30e6, 300e6), 'characteristics': 'line_of_sight'},
            {'name': 'UHF', 'range': (300e6, 3e9), 'characteristics': 'mobile'},
            {'name': 'SHF_L', 'range': (3e9, 30e9), 'characteristics': 'satellite'},
            {'name': 'SHF_H', 'range': (30e9, 300e9), 'characteristics': 'high_capacity'}
        ]
        
        logger.info("改进的种群管理器初始化完成")

    def initialize_population(self, task_id: str, n_var: int, xl: np.ndarray, 
                            xu: np.ndarray, n_samples: int) -> np.ndarray:
        """
        智能化种群初始化
        
        参数:
        task_id: 当前任务ID
        n_var: 问题维度
        xl: 下界
        xu: 上界
        n_samples: 种群大小
        
        返回:
        初始化的种群
        """
        try:
            logger.info(f"开始智能初始化种群：任务={task_id}, 维度={n_var}, 大小={n_samples}")
            
            # 验证输入参数
            if not self._validate_inputs(n_var, xl, xu, n_samples):
                return self._create_fallback_population(n_var, xl, xu, n_samples)
            
            n_links = n_var // 5
            population = []
            
            # 1. 历史案例引导初始化
            historical_count = int(n_samples * self.init_strategies['historical_guided'])
            historical_solutions = self._create_historical_guided_population(
                task_id, n_links, xl, xu, historical_count
            )
            population.extend(historical_solutions)
            
            # 2. 领域知识引导初始化
            domain_count = int(n_samples * self.init_strategies['domain_knowledge'])
            domain_solutions = self._create_domain_knowledge_population(
                n_links, xl, xu, domain_count
            )
            population.extend(domain_solutions)
            
            # 3. 频率优化策略初始化
            freq_count = int(n_samples * self.init_strategies['frequency_optimized'])
            freq_solutions = self._create_frequency_optimized_population(
                n_links, xl, xu, freq_count
            )
            population.extend(freq_solutions)
            
            # 4. 拉丁超立方采样
            lhs_count = int(n_samples * self.init_strategies['latin_hypercube'])
            lhs_solutions = self._create_latin_hypercube_population(
                n_var, xl, xu, lhs_count
            )
            population.extend(lhs_solutions)
            
            # 5. 填充剩余位置
            remaining = n_samples - len(population)
            if remaining > 0:
                random_solutions = self._create_diverse_random_population(
                    n_links, xl, xu, remaining
                )
                population.extend(random_solutions)
            
            # 确保种群大小正确
            population = self._adjust_population_size(population, n_samples)
            
            # 质量验证和修复
            population = self._validate_and_repair_population(population, xl, xu)
            
            # 多样性检查和调整
            population = self._ensure_population_diversity(population, xl, xu)
            
            logger.info(f"种群初始化完成：生成{len(population)}个个体")
            
            return np.array(population)
            
        except Exception as e:
            logger.error(f"种群初始化失败：{str(e)}")
            return self._create_fallback_population(n_var, xl, xu, n_samples)

    def _validate_inputs(self, n_var: int, xl: np.ndarray, xu: np.ndarray, 
                        n_samples: int) -> bool:
        """验证输入参数的有效性"""
        try:
            if n_var <= 0 or n_samples <= 0:
                logger.error("无效的问题维度或种群大小")
                return False
            
            if len(xl) != n_var or len(xu) != n_var:
                logger.error("边界向量长度与问题维度不匹配")
                return False
            
            if np.any(xl >= xu):
                logger.error("下界大于等于上界")
                return False
            
            if n_var % 5 != 0:
                logger.warning("问题维度不是5的倍数，可能存在问题")
            
            return True
            
        except Exception as e:
            logger.error(f"输入验证异常：{str(e)}")
            return False

    def _create_historical_guided_population(self, task_id: str, n_links: int,
                                           xl: np.ndarray, xu: np.ndarray, 
                                           count: int) -> List[np.ndarray]:
        """基于历史案例的智能初始化"""
        solutions = []
        
        try:
            if not self.neo4j_handler or count <= 0:
                return solutions
            
            # 获取相似历史案例
            similar_cases = self.neo4j_handler.get_similar_cases(
                task_id, limit=min(count * 2, 20)
            )
            
            if not similar_cases:
                logger.info("未找到相似历史案例，跳过历史引导初始化")
                return solutions
            
            logger.info(f"找到{len(similar_cases)}个相似案例用于引导初始化")
            
            # 从历史案例中提取解
            for case_id in similar_cases:
                if len(solutions) >= count:
                    break
                
                try:
                    # 获取历史通信参数
                    comm_params = self.neo4j_handler.get_historical_communication_parameters(case_id)
                    
                    if comm_params:
                        # 转换为解向量
                        base_solution = self._parameters_to_solution(comm_params)
                        
                        # 生成多个变体
                        variants = self._generate_historical_variants(
                            base_solution, xl, xu, n_links, min(3, count - len(solutions))
                        )
                        
                        for variant in variants:
                            if len(solutions) < count:
                                solutions.append(variant)
                            
                except Exception as e:
                    logger.warning(f"处理历史案例{case_id}失败：{str(e)}")
                    continue
            
            logger.info(f"历史引导初始化生成{len(solutions)}个解")
            
        except Exception as e:
            logger.error(f"历史引导初始化异常：{str(e)}")
        
        return solutions

    def _generate_historical_variants(self, base_solution: np.ndarray, 
                                    xl: np.ndarray, xu: np.ndarray,
                                    n_links: int, count: int) -> List[np.ndarray]:
        """基于历史解生成变体"""
        variants = []
        
        try:
            # 调整解向量维度
            target_dim = n_links * 5
            adjusted_solution = self._adjust_solution_dimension(base_solution, target_dim, xl, xu)
            
            # 原始解（轻微扰动）
            if len(variants) < count:
                perturbed = self._apply_light_perturbation(adjusted_solution, xl, xu, 0.05)
                variants.append(perturbed)
            
            # 适应性变体（根据当前环境调整）
            if len(variants) < count:
                adaptive = self._create_adaptive_variant(adjusted_solution, xl, xu)
                variants.append(adaptive)
            
            # 保守变体（减少风险参数）
            if len(variants) < count:
                conservative = self._create_conservative_variant(adjusted_solution, xl, xu)
                variants.append(conservative)
            
        except Exception as e:
            logger.warning(f"变体生成失败：{str(e)}")
        
        return variants

    def _create_domain_knowledge_population(self, n_links: int, xl: np.ndarray, 
                                          xu: np.ndarray, count: int) -> List[np.ndarray]:
        """基于领域知识的初始化"""
        solutions = []
        
        try:
            if count <= 0:
                return solutions
            
            # 通信系统设计原则
            design_principles = [
                'frequency_diversity',    # 频率分集
                'power_optimization',     # 功率优化
                'modulation_matching',    # 调制匹配
                'interference_avoidance', # 干扰避免
                'robust_design'          # 鲁棒设计
            ]
            
            per_principle = max(1, count // len(design_principles))
            
            for principle in design_principles:
                if len(solutions) >= count:
                    break
                
                principle_solutions = self._apply_design_principle(
                    principle, n_links, xl, xu, per_principle
                )
                solutions.extend(principle_solutions)
            
            # 填充剩余
            while len(solutions) < count:
                random_principle = random.choice(design_principles)
                additional = self._apply_design_principle(
                    random_principle, n_links, xl, xu, 1
                )
                solutions.extend(additional)
            
            logger.info(f"领域知识初始化生成{len(solutions)}个解")
            
        except Exception as e:
            logger.error(f"领域知识初始化异常：{str(e)}")
        
        return solutions[:count]

    def _apply_design_principle(self, principle: str, n_links: int,
                              xl: np.ndarray, xu: np.ndarray, count: int) -> List[np.ndarray]:
        """应用特定的设计原则"""
        solutions = []
        
        try:
            for _ in range(count):
                if principle == 'frequency_diversity':
                    solution = self._create_frequency_diverse_solution(n_links, xl, xu)
                elif principle == 'power_optimization':
                    solution = self._create_power_optimized_solution(n_links, xl, xu)
                elif principle == 'modulation_matching':
                    solution = self._create_modulation_matched_solution(n_links, xl, xu)
                elif principle == 'interference_avoidance':
                    solution = self._create_interference_avoiding_solution(n_links, xl, xu)
                elif principle == 'robust_design':
                    solution = self._create_robust_design_solution(n_links, xl, xu)
                else:
                    solution = self._create_balanced_solution(n_links, xl, xu)
                
                if solution is not None:
                    solutions.append(solution)
                    
        except Exception as e:
            logger.warning(f"设计原则{principle}应用失败：{str(e)}")
        
        return solutions

    def _create_frequency_diverse_solution(self, n_links: int, xl: np.ndarray, 
                                         xu: np.ndarray) -> np.ndarray:
        """创建频率分集解"""
        try:
            solution = np.zeros(n_links * 5)
            
            # 频率分配：均匀分布在不同频段
            freq_min = xl[0]
            freq_max = xu[0]
            
            if n_links > 1:
                freq_step = (freq_max - freq_min) / n_links
                for i in range(n_links):
                    # 在各自频段内随机选择
                    band_min = freq_min + i * freq_step
                    band_max = freq_min + (i + 1) * freq_step
                    solution[i] = random.uniform(band_min, band_max)
            else:
                solution[0] = random.uniform(freq_min, freq_max)
            
            # 带宽：适中值，避免过度重叠
            bw_min = xl[n_links]
            bw_max = xu[n_links]
            target_bw = (bw_min + bw_max) / 2
            
            for i in range(n_links):
                solution[n_links + i] = target_bw * random.uniform(0.8, 1.2)
            
            # 功率：根据频率调整
            power_min = xl[2 * n_links]
            power_max = xu[2 * n_links]
            
            for i in range(n_links):
                freq = solution[i]
                # 高频用稍低功率，低频用稍高功率
                if freq < (freq_min + freq_max) / 2:
                    power = random.uniform(power_max * 0.6, power_max)
                else:
                    power = random.uniform(power_min, power_max * 0.8)
                solution[2 * n_links + i] = power
            
            # 调制和极化：随机但合理
            for i in range(n_links):
                solution[3 * n_links + i] = random.randint(0, 3)  # 调制
                solution[4 * n_links + i] = random.randint(0, 3)  # 极化
            
            return np.clip(solution, xl, xu)
            
        except Exception as e:
            logger.warning(f"频率分集解创建失败：{str(e)}")
            return None

    def _create_power_optimized_solution(self, n_links: int, xl: np.ndarray, 
                                       xu: np.ndarray) -> np.ndarray:
        """创建功率优化解"""
        try:
            solution = np.zeros(n_links * 5)
            
            # 功率分配：梯度分配策略
            power_min = xl[2 * n_links]
            power_max = xu[2 * n_links]
            
            # 为重要链路分配更高功率
            power_levels = np.linspace(power_max, power_min, n_links)
            np.random.shuffle(power_levels)  # 随机化分配
            
            for i in range(n_links):
                solution[2 * n_links + i] = power_levels[i]
            
            # 频率：避免集中，适度分散
            freq_min = xl[0]
            freq_max = xu[0]
            
            # 使用黄金分割点分配频率
            golden_ratio = (1 + np.sqrt(5)) / 2
            for i in range(n_links):
                ratio = (i * golden_ratio) % 1
                solution[i] = freq_min + ratio * (freq_max - freq_min)
            
            # 带宽：根据功率调整
            bw_min = xl[n_links]
            bw_max = xu[n_links]
            
            for i in range(n_links):
                power_ratio = (solution[2 * n_links + i] - power_min) / (power_max - power_min)
                # 高功率允许更大带宽
                bw = bw_min + power_ratio * (bw_max - bw_min)
                solution[n_links + i] = bw
            
            # 调制：根据功率选择
            for i in range(n_links):
                power_ratio = (solution[2 * n_links + i] - power_min) / (power_max - power_min)
                if power_ratio > 0.7:
                    solution[3 * n_links + i] = random.choice([2, 3])  # 高阶调制
                elif power_ratio > 0.4:
                    solution[3 * n_links + i] = random.choice([1, 2])  # 中阶调制
                else:
                    solution[3 * n_links + i] = random.choice([0, 1])  # 低阶调制
            
            # 极化：随机
            for i in range(n_links):
                solution[4 * n_links + i] = random.randint(0, 3)
            
            return np.clip(solution, xl, xu)
            
        except Exception as e:
            logger.warning(f"功率优化解创建失败：{str(e)}")
            return None

    def _create_frequency_optimized_population(self, n_links: int, xl: np.ndarray,
                                             xu: np.ndarray, count: int) -> List[np.ndarray]:
        """创建频率优化策略种群"""
        solutions = []
        
        try:
            if count <= 0:
                return solutions
            
            # 频率分配策略
            strategies = [
                'band_separation',    # 频段分离
                'harmonic_spacing',   # 谐波间隔
                'optimal_coverage',   # 最优覆盖
                'interference_min'    # 干扰最小化
            ]
            
            per_strategy = max(1, count // len(strategies))
            
            for strategy in strategies:
                if len(solutions) >= count:
                    break
                
                strategy_solutions = self._apply_frequency_strategy(
                    strategy, n_links, xl, xu, per_strategy
                )
                solutions.extend(strategy_solutions)
            
            # 填充剩余
            while len(solutions) < count:
                random_strategy = random.choice(strategies)
                additional = self._apply_frequency_strategy(
                    random_strategy, n_links, xl, xu, 1
                )
                solutions.extend(additional)
            
            logger.info(f"频率优化初始化生成{len(solutions)}个解")
            
        except Exception as e:
            logger.error(f"频率优化初始化异常：{str(e)}")
        
        return solutions[:count]

    def _apply_frequency_strategy(self, strategy: str, n_links: int,
                                xl: np.ndarray, xu: np.ndarray, count: int) -> List[np.ndarray]:
        """应用频率分配策略"""
        solutions = []
        
        try:
            for _ in range(count):
                solution = np.zeros(n_links * 5)
                
                freq_min = xl[0]
                freq_max = xu[0]
                
                if strategy == 'band_separation':
                    # 频段分离：每个链路使用不同的预定义频段
                    available_bands = self.frequency_bands.copy()
                    random.shuffle(available_bands)
                    
                    for i in range(n_links):
                        if i < len(available_bands):
                            band = available_bands[i]
                            band_min = max(band['range'][0], freq_min)
                            band_max = min(band['range'][1], freq_max)
                            if band_max > band_min:
                                solution[i] = random.uniform(band_min, band_max)
                            else:
                                solution[i] = random.uniform(freq_min, freq_max)
                        else:
                            solution[i] = random.uniform(freq_min, freq_max)
                
                elif strategy == 'harmonic_spacing':
                    # 谐波间隔：使用谐波关系避免干扰
                    base_freq = random.uniform(freq_min, freq_max / 4)
                    for i in range(n_links):
                        harmonic = 2 ** i  # 倍频关系
                        candidate_freq = base_freq * harmonic
                        if candidate_freq <= freq_max:
                            solution[i] = candidate_freq
                        else:
                            solution[i] = random.uniform(freq_min, freq_max)
                
                elif strategy == 'optimal_coverage':
                    # 最优覆盖：在整个频谱范围内均匀分布
                    if n_links > 1:
                        spacing = (freq_max - freq_min) / (n_links - 1)
                        for i in range(n_links):
                            base_freq = freq_min + i * spacing
                            # 添加小扰动
                            perturbation = spacing * 0.1 * (random.random() - 0.5)
                            solution[i] = np.clip(base_freq + perturbation, freq_min, freq_max)
                    else:
                        solution[0] = (freq_min + freq_max) / 2
                
                elif strategy == 'interference_min':
                    # 干扰最小化：基于干扰矩阵选择频率
                    selected_freqs = []
                    candidates = np.linspace(freq_min, freq_max, n_links * 10)
                    
                    for i in range(n_links):
                        best_freq = None
                        min_interference = float('inf')
                        
                        for candidate in candidates:
                            interference = sum(
                                abs(candidate - selected) for selected in selected_freqs
                            )
                            if interference < min_interference:
                                min_interference = interference
                                best_freq = candidate
                        
                        if best_freq is not None:
                            selected_freqs.append(best_freq)
                            solution[i] = best_freq
                        else:
                            solution[i] = random.uniform(freq_min, freq_max)
                
                # 填充其他参数
                self._fill_remaining_parameters(solution, n_links, xl, xu)
                
                solutions.append(np.clip(solution, xl, xu))
                
        except Exception as e:
            logger.warning(f"频率策略{strategy}应用失败：{str(e)}")
        
        return solutions

    def _create_latin_hypercube_population(self, n_var: int, xl: np.ndarray,
                                         xu: np.ndarray, count: int) -> List[np.ndarray]:
        """创建拉丁超立方采样种群"""
        solutions = []
        
        try:
            if count <= 0:
                return solutions
            
            # 尝试使用scipy的LHS
            try:
                from scipy.stats.qmc import LatinHypercube
                
                sampler = LatinHypercube(d=n_var, seed=random.randint(0, 10000))
                lhs_samples = sampler.random(n=count)
                
                # 映射到实际参数范围
                for sample in lhs_samples:
                    solution = xl + sample * (xu - xl)
                    solutions.append(solution)
                
                logger.info(f"LHS采样生成{len(solutions)}个解")
                
            except ImportError:
                logger.warning("scipy不可用，使用简化LHS实现")
                # 简化的LHS实现
                for _ in range(count):
                    solution = np.zeros(n_var)
                    for j in range(n_var):
                        # 简单的分层采样
                        stratum = random.random()
                        solution[j] = xl[j] + stratum * (xu[j] - xl[j])
                    solutions.append(solution)
            
        except Exception as e:
            logger.error(f"LHS采样异常：{str(e)}")
        
        return solutions

    def _create_diverse_random_population(self, n_links: int, xl: np.ndarray,
                                        xu: np.ndarray, count: int) -> List[np.ndarray]:
        """创建多样化随机种群"""
        solutions = []
        
        try:
            for _ in range(count):
                solution = np.zeros(n_links * 5)
                
                # 随机但有约束的参数生成
                for i in range(n_links):
                    # 频率：在合理范围内随机
                    solution[i] = random.uniform(xl[i], xu[i])
                    
                    # 带宽：偏向中等大小
                    bw_min = xl[n_links + i]
                    bw_max = xu[n_links + i]
                    bw_center = (bw_min + bw_max) / 2
                    bw_range = bw_max - bw_min
                    # 使用正态分布，偏向中心值
                    bandwidth = np.random.normal(bw_center, bw_range / 4)
                    solution[n_links + i] = np.clip(bandwidth, bw_min, bw_max)
                    
                    # 功率：随机但考虑实际约束
                    power_min = xl[2 * n_links + i]
                    power_max = xu[2 * n_links + i]
                    solution[2 * n_links + i] = random.uniform(power_min, power_max)
                    
                    # 调制方式：随机
                    solution[3 * n_links + i] = random.randint(
                        int(xl[3 * n_links + i]), int(xu[3 * n_links + i])
                    )
                    
                    # 极化方式：随机
                    solution[4 * n_links + i] = random.randint(
                        int(xl[4 * n_links + i]), int(xu[4 * n_links + i])
                    )
                
                solutions.append(np.clip(solution, xl, xu))
                
        except Exception as e:
            logger.error(f"随机种群生成异常：{str(e)}")
        
        return solutions

    def _fill_remaining_parameters(self, solution: np.ndarray, n_links: int,
                                 xl: np.ndarray, xu: np.ndarray):
        """填充解向量的剩余参数"""
        try:
            # 带宽：基于频率自适应选择
            for i in range(n_links):
                freq = solution[i]
                bw_min = xl[n_links + i]
                bw_max = xu[n_links + i]
                
                # 高频允许更大带宽
                freq_ratio = (freq - xl[i]) / (xu[i] - xl[i])
                target_bw = bw_min + freq_ratio * (bw_max - bw_min)
                solution[n_links + i] = target_bw
            
            # 功率：中等偏上
            for i in range(n_links):
                power_min = xl[2 * n_links + i]
                power_max = xu[2 * n_links + i]
                solution[2 * n_links + i] = random.uniform(
                    power_min + 0.3 * (power_max - power_min),
                    power_max
                )
            
            # 调制和极化：随机
            for i in range(n_links):
                solution[3 * n_links + i] = random.randint(
                    int(xl[3 * n_links + i]), int(xu[3 * n_links + i])
                )
                solution[4 * n_links + i] = random.randint(
                    int(xl[4 * n_links + i]), int(xu[4 * n_links + i])
                )
                
        except Exception as e:
            logger.warning(f"参数填充失败：{str(e)}")

    def _adjust_population_size(self, population: List[np.ndarray], 
                              target_size: int) -> List[np.ndarray]:
        """调整种群大小"""
        try:
            current_size = len(population)
            
            if current_size == target_size:
                return population
            elif current_size > target_size:
                # 随机选择目标数量
                return random.sample(population, target_size)
            else:
                # 复制现有个体填充
                while len(population) < target_size:
                    source_individual = random.choice(population)
                    # 添加轻微扰动
                    perturbed = source_individual.copy()
                    noise = np.random.normal(0, 0.01, len(perturbed))
                    perturbed += noise
                    population.append(perturbed)
                
                return population
                
        except Exception as e:
            logger.error(f"种群大小调整失败：{str(e)}")
            return population

    def _validate_and_repair_population(self, population: List[np.ndarray],
                                      xl: np.ndarray, xu: np.ndarray) -> List[np.ndarray]:
        """验证和修复种群"""
        repaired_population = []
        
        try:
            for i, individual in enumerate(population):
                try:
                    # 检查维度
                    if len(individual) != len(xl):
                        logger.warning(f"个体{i}维度不匹配，进行修复")
                        individual = self._repair_dimension(individual, xl, xu)
                    
                    # 检查数值有效性
                    if not np.all(np.isfinite(individual)):
                        logger.warning(f"个体{i}包含无效数值，进行修复")
                        individual = self._repair_invalid_values(individual, xl, xu)
                    
                    # 边界约束
                    individual = np.clip(individual, xl, xu)
                    
                    # 领域约束检查（例如频率间隔）
                    individual = self._repair_domain_constraints(individual, xl, xu)
                    
                    repaired_population.append(individual)
                    
                except Exception as e:
                    logger.warning(f"个体{i}修复失败：{str(e)}，使用随机个体替换")
                    # 生成替换个体
                    replacement = xl + np.random.random(len(xl)) * (xu - xl)
                    repaired_population.append(replacement)
            
        except Exception as e:
            logger.error(f"种群验证修复异常：{str(e)}")
        
        return repaired_population

    def _ensure_population_diversity(self, population: List[np.ndarray],
                                   xl: np.ndarray, xu: np.ndarray) -> List[np.ndarray]:
        """确保种群多样性"""
        try:
            if len(population) < 2:
                return population
            
            # 计算个体间距离矩阵
            distances = np.zeros((len(population), len(population)))
            for i in range(len(population)):
                for j in range(i + 1, len(population)):
                    dist = np.linalg.norm(population[i] - population[j])
                    distances[i, j] = distances[j, i] = dist
            
            # 识别过于相似的个体
            min_distance = 0.01 * np.linalg.norm(xu - xl)  # 最小距离阈值
            to_replace = []
            
            for i in range(len(population)):
                for j in range(i + 1, len(population)):
                    if distances[i, j] < min_distance:
                        to_replace.append(j)
            
            # 替换相似个体
            unique_replace = list(set(to_replace))
            for idx in unique_replace:
                # 生成新的多样化个体
                new_individual = xl + np.random.random(len(xl)) * (xu - xl)
                population[idx] = new_individual
                logger.debug(f"替换相似个体{idx}以增加多样性")
            
        except Exception as e:
            logger.warning(f"多样性检查失败：{str(e)}")
        
        return population

    def _repair_dimension(self, individual: np.ndarray, xl: np.ndarray, 
                         xu: np.ndarray) -> np.ndarray:
        """修复个体维度"""
        try:
            target_dim = len(xl)
            current_dim = len(individual)
            
            if current_dim < target_dim:
                # 扩展维度
                padding = np.random.uniform(
                    xl[current_dim:], xu[current_dim:], target_dim - current_dim
                )
                return np.concatenate([individual, padding])
            else:
                # 截断维度
                return individual[:target_dim]
                
        except Exception as e:
            logger.warning(f"维度修复失败：{str(e)}")
            return xl + np.random.random(len(xl)) * (xu - xl)

    def _repair_invalid_values(self, individual: np.ndarray, xl: np.ndarray,
                             xu: np.ndarray) -> np.ndarray:
        """修复无效数值"""
        try:
            repaired = individual.copy()
            invalid_mask = ~np.isfinite(repaired)
            
            if np.any(invalid_mask):
                # 用随机值替换无效值
                repaired[invalid_mask] = np.random.uniform(
                    xl[invalid_mask], xu[invalid_mask]
                )
            
            return repaired
            
        except Exception as e:
            logger.warning(f"数值修复失败：{str(e)}")
            return xl + np.random.random(len(xl)) * (xu - xl)

    def _repair_domain_constraints(self, individual: np.ndarray, xl: np.ndarray,
                                 xu: np.ndarray) -> np.ndarray:
        """修复领域约束"""
        try:
            repaired = individual.copy()
            n_links = len(individual) // 5
            
            # 检查频率间隔约束
            frequencies = repaired[:n_links]
            bandwidths = repaired[n_links:2*n_links]
            
            # 简单的频率冲突修复
            for i in range(n_links):
                for j in range(i + 1, n_links):
                    spacing = abs(frequencies[i] - frequencies[j])
                    required_spacing = (bandwidths[i] + bandwidths[j]) / 2
                    
                    if spacing < required_spacing:
                        # 调整第二个频率
                        if frequencies[j] > frequencies[i]:
                            new_freq = frequencies[i] + required_spacing * 1.1
                        else:
                            new_freq = frequencies[i] - required_spacing * 1.1
                        
                        # 确保在边界内
                        new_freq = np.clip(new_freq, xl[j], xu[j])
                        repaired[j] = new_freq
            
            return repaired
            
        except Exception as e:
            logger.warning(f"领域约束修复失败：{str(e)}")
            return individual

    def _create_fallback_population(self, n_var: int, xl: np.ndarray,
                                  xu: np.ndarray, n_samples: int) -> np.ndarray:
        """创建备用种群"""
        try:
            logger.warning("使用备用种群初始化方法")
            population = []
            
            for _ in range(n_samples):
                individual = xl + np.random.random(n_var) * (xu - xl)
                population.append(individual)
            
            return np.array(population)
            
        except Exception as e:
            logger.error(f"备用种群创建失败：{str(e)}")
            # 最后的备用方案
            return np.tile((xl + xu) / 2, (n_samples, 1))

    # 从原有代码继承的方法
    def solution_to_parameters(self, solution: np.ndarray, n_links: int) -> List[Dict[str, Any]]:
        """将解向量转换为通信参数字典列表"""
        params_list = []
        try:
            solution_reshaped = solution.reshape(-1)
            
            if len(solution_reshaped) < n_links * 5:
                padding = np.array([4e9, 20e6, 10, 0, 0] * (n_links - len(solution_reshaped) // 5))
                solution_reshaped = np.concatenate([solution_reshaped, padding])
                
            solution_reshaped = solution_reshaped[:n_links*5].reshape(n_links, 5)
            
            for i in range(n_links):
                if i < len(solution_reshaped):
                    freq = self._constrain_param(solution_reshaped[i, 0], 
                                               self.config.freq_min, self.config.freq_max)
                    bandwidth = self._constrain_param(solution_reshaped[i, 1], 
                                                    self.config.bandwidth_min, self.config.bandwidth_max)
                    power = self._constrain_param(solution_reshaped[i, 2], 
                                                 self.config.power_min, self.config.power_max)
                    
                    params = {
                        'frequency': freq,
                        'bandwidth': bandwidth,
                        'power': power,
                        'modulation': self.index_to_modulation(solution_reshaped[i, 3]),
                        'polarization': self.index_to_polarization(solution_reshaped[i, 4])
                    }
                    params_list.append(params)
        except Exception as e:
            logger.error(f"解向量转换失败: {str(e)}")
            for i in range(n_links):
                params = {
                    'frequency': 4e9,
                    'bandwidth': 20e6,
                    'power': 10,
                    'modulation': 'BPSK',
                    'polarization': 'LINEAR'
                }
                params_list.append(params)
        
        return params_list

    def _constrain_param(self, value: float, min_val: float, max_val: float) -> float:
        """将参数值约束在指定范围内"""
        if value < min_val or value > max_val:
            if value < min_val * 0.5 or value > max_val * 2:
                return (min_val + max_val) / 2
            return max(min_val, min(value, max_val))
        return value

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

    def _parameters_to_solution(self, parameters: List[Dict]) -> np.ndarray:
        """将通信参数列表转换为解向量"""
        solution = []
        
        for params in parameters:
            solution.append(params.get('frequency', 0))
            solution.append(params.get('bandwidth', 0))
            solution.append(params.get('power', 0))
            
            modulation = params.get('modulation', 'BPSK')
            solution.append(self.modulation_to_index(modulation))
            
            polarization = params.get('polarization', 'LINEAR')
            solution.append(self.polarization_to_index(polarization))
        
        return np.array(solution)

    def _adjust_solution_dimension(self, solution: np.ndarray, target_dim: int,
                                 xl: np.ndarray, xu: np.ndarray) -> np.ndarray:
        """调整解向量维度"""
        try:
            current_dim = len(solution)
            
            if current_dim == target_dim:
                return solution
            elif current_dim > target_dim:
                return solution[:target_dim]
            else:
                # 扩展维度
                padding_size = target_dim - current_dim
                padding = xl[-padding_size:] + np.random.random(padding_size) * (xu[-padding_size:] - xl[-padding_size:])
                return np.concatenate([solution, padding])
                
        except Exception as e:
            logger.warning(f"维度调整失败：{str(e)}")
            return xl + np.random.random(len(xl)) * (xu - xl)

    def _apply_light_perturbation(self, solution: np.ndarray, xl: np.ndarray,
                                xu: np.ndarray, strength: float = 0.05) -> np.ndarray:
        """应用轻微扰动"""
        try:
            perturbed = solution.copy()
            
            # 对每个参数应用小扰动
            for i in range(len(solution)):
                range_val = xu[i] - xl[i]
                perturbation = np.random.normal(0, strength * range_val)
                perturbed[i] += perturbation
            
            return np.clip(perturbed, xl, xu)
            
        except Exception as e:
            logger.warning(f"扰动应用失败：{str(e)}")
            return solution

    def _create_adaptive_variant(self, solution: np.ndarray, xl: np.ndarray,
                               xu: np.ndarray) -> np.ndarray:
        """创建适应性变体"""
        try:
            adaptive = solution.copy()
            n_links = len(solution) // 5
            
            # 根据环境调整功率
            for i in range(n_links):
                power_idx = 2 * n_links + i
                current_power = adaptive[power_idx]
                
                # 在恶劣环境中增加功率
                if hasattr(self, 'config') and self.config:
                    power_boost = random.uniform(1.1, 1.3)
                    new_power = min(current_power * power_boost, xu[power_idx])
                    adaptive[power_idx] = new_power
            
            return np.clip(adaptive, xl, xu)
            
        except Exception as e:
            logger.warning(f"适应性变体创建失败：{str(e)}")
            return solution

    def _create_conservative_variant(self, solution: np.ndarray, xl: np.ndarray,
                                   xu: np.ndarray) -> np.ndarray:
        """创建保守变体"""
        try:
            conservative = solution.copy()
            n_links = len(solution) // 5
            
            # 使用保守的调制方式
            for i in range(n_links):
                mod_idx = 3 * n_links + i
                conservative[mod_idx] = min(1, conservative[mod_idx])  # 最多使用QPSK
            
            # 使用适中的功率
            for i in range(n_links):
                power_idx = 2 * n_links + i
                power_range = xu[power_idx] - xl[power_idx]
                conservative[power_idx] = xl[power_idx] + 0.6 * power_range
            
            return np.clip(conservative, xl, xu)
            
        except Exception as e:
            logger.warning(f"保守变体创建失败：{str(e)}")
            return solution

    def _create_modulation_matched_solution(self, n_links: int, xl: np.ndarray,
                                          xu: np.ndarray) -> np.ndarray:
        """创建调制匹配解"""
        try:
            solution = np.zeros(n_links * 5)
            
            # 频率和带宽：随机合理值
            for i in range(n_links):
                solution[i] = random.uniform(xl[i], xu[i])
                solution[n_links + i] = random.uniform(xl[n_links + i], xu[n_links + i])
            
            # 功率和调制匹配
            for i in range(n_links):
                power_min = xl[2 * n_links + i]
                power_max = xu[2 * n_links + i]
                
                # 根据功率选择合适的调制
                power_level = random.uniform(power_min, power_max)
                solution[2 * n_links + i] = power_level
                
                # 功率-调制匹配
                power_ratio = (power_level - power_min) / (power_max - power_min)
                if power_ratio > 0.8:
                    solution[3 * n_links + i] = 3  # QAM64
                elif power_ratio > 0.6:
                    solution[3 * n_links + i] = 2  # QAM16
                elif power_ratio > 0.3:
                    solution[3 * n_links + i] = 1  # QPSK
                else:
                    solution[3 * n_links + i] = 0  # BPSK
                
                # 极化方式
                solution[4 * n_links + i] = random.randint(0, 3)
            
            return np.clip(solution, xl, xu)
            
        except Exception as e:
            logger.warning(f"调制匹配解创建失败：{str(e)}")
            return None

    def _create_interference_avoiding_solution(self, n_links: int, xl: np.ndarray,
                                             xu: np.ndarray) -> np.ndarray:
        """创建干扰避免解"""
        try:
            solution = np.zeros(n_links * 5)
            
            # 频率分配：最大化间隔
            freq_min = xl[0]
            freq_max = xu[0]
            
            if n_links > 1:
                # 均匀分布频率
                freq_step = (freq_max - freq_min) / n_links
                for i in range(n_links):
                    solution[i] = freq_min + (i + 0.5) * freq_step
            else:
                solution[0] = (freq_min + freq_max) / 2
            
            # 带宽：适中，避免过度重叠
            bw_min = xl[n_links]
            bw_max = xu[n_links]
            conservative_bw = bw_min + 0.4 * (bw_max - bw_min)
            
            for i in range(n_links):
                solution[n_links + i] = conservative_bw
            
            # 功率：适中
            for i in range(n_links):
                power_min = xl[2 * n_links + i]
                power_max = xu[2 * n_links + i]
                solution[2 * n_links + i] = (power_min + power_max) / 2
            
            # 调制：保守选择
            for i in range(n_links):
                solution[3 * n_links + i] = random.choice([0, 1])  # BPSK或QPSK
            
            # 极化：多样化
            for i in range(n_links):
                solution[4 * n_links + i] = i % 4  # 循环使用不同极化
            
            return np.clip(solution, xl, xu)
            
        except Exception as e:
            logger.warning(f"干扰避免解创建失败：{str(e)}")
            return None

    def _create_robust_design_solution(self, n_links: int, xl: np.ndarray,
                                     xu: np.ndarray) -> np.ndarray:
        """创建鲁棒设计解"""
        try:
            solution = np.zeros(n_links * 5)
            
            # 频率：选择鲁棒频段
            robust_freqs = []
            freq_min = xl[0]
            freq_max = xu[0]
            
            # 偏好低频段（更鲁棒）
            for i in range(n_links):
                if i < n_links // 2:
                    # 前半部分使用低频
                    freq = random.uniform(freq_min, freq_min + (freq_max - freq_min) * 0.4)
                else:
                    # 后半部分使用中频
                    freq = random.uniform(freq_min + (freq_max - freq_min) * 0.3, 
                                        freq_min + (freq_max - freq_min) * 0.7)
                robust_freqs.append(freq)
                solution[i] = freq
            
            # 带宽：保守选择
            for i in range(n_links):
                bw_min = xl[n_links + i]
                bw_max = xu[n_links + i]
                solution[n_links + i] = bw_min + 0.3 * (bw_max - bw_min)
            
            # 功率：较高但不过度
            for i in range(n_links):
                power_min = xl[2 * n_links + i]
                power_max = xu[2 * n_links + i]
                solution[2 * n_links + i] = power_min + 0.7 * (power_max - power_min)
            
            # 调制：鲁棒选择
            for i in range(n_links):
                solution[3 * n_links + i] = random.choice([0, 1])  # BPSK或QPSK
            
            # 极化：使用鲁棒极化
            for i in range(n_links):
                solution[4 * n_links + i] = random.choice([0, 1])  # LINEAR或CIRCULAR
            
            return np.clip(solution, xl, xu)
            
        except Exception as e:
            logger.warning(f"鲁棒设计解创建失败：{str(e)}")
            return None

    def _create_balanced_solution(self, n_links: int, xl: np.ndarray,
                                xu: np.ndarray) -> np.ndarray:
        """创建平衡解"""
        try:
            solution = np.zeros(n_links * 5)
            
            # 所有参数使用中等值
            for i in range(n_links * 5):
                solution[i] = (xl[i] + xu[i]) / 2
            
            # 添加适度随机化
            for i in range(len(solution)):
                range_val = xu[i] - xl[i]
                perturbation = random.uniform(-0.1 * range_val, 0.1 * range_val)
                solution[i] += perturbation
            
            return np.clip(solution, xl, xu)
            
        except Exception as e:
            logger.warning(f"平衡解创建失败：{str(e)}")
            return None


# 向后兼容的包装器
class PopulationManager(ImprovedPopulationManager):
    """向后兼容的种群管理类"""
    
    def __init__(self, config, neo4j_handler=None):
        super().__init__(config, neo4j_handler)
        logger.info("使用改进的种群管理器（兼容性模式）")
