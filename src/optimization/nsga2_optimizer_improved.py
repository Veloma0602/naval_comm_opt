import time
import os
import json
import logging
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.core.sampling import Sampling
from pymoo.core.callback import Callback
from typing import Dict, List, Tuple, Any, Optional

# 导入改进的模块
from optimization.numerical_stability import get_stability_manager, validate_array
from models.constraints_improved import ImprovedConstraints
from models.objectives_improved import ImprovedObjectiveFunction
from optimization.population_improved import ImprovedPopulationManager

logger = logging.getLogger(__name__)

class ImprovedCommunicationProblem(Problem):
    """改进的通信优化问题定义"""
    
    def __init__(self, optimizer):
        """
        初始化改进的通信优化问题
        
        参数:
        optimizer: 优化器实例
        """
        super().__init__(
            n_var=optimizer.n_vars,
            n_obj=5,
            n_constr=optimizer.n_constraints,
            xl=optimizer.lower_bounds,
            xu=optimizer.upper_bounds
        )
        self.optimizer = optimizer
        self.stability_manager = get_stability_manager()
        self.evaluation_count = 0
        
    def _evaluate(self, X, out, *args, **kwargs):
        """
        评估解的目标函数值和约束 - 改进版
        包含完整的数值稳定性保护和错误处理
        """
        n_solutions = len(X)
        f = np.zeros((n_solutions, 5))
        g = np.zeros((n_solutions, self.n_constr))
        
        # 批量验证输入
        X = validate_array(X, "解向量矩阵")
        
        for i, xi in enumerate(X):
            try:
                self.evaluation_count += 1
                
                # 验证单个解向量
                xi = self.stability_manager.validate_parameter_vector(
                    xi, (self.xl, self.xu)
                )
                
                # 转换为参数字典
                params_list = self.optimizer.population_manager.solution_to_parameters(
                    xi, self.optimizer.n_links
                )
                
                # 验证参数有效性
                if not self._validate_parameters(params_list):
                    logger.warning(f"解 {i} 参数无效，使用默认目标值")
                    f[i, :] = self._get_default_objectives()
                    g[i, :] = self._get_default_constraints()
                    continue
                
                # 计算目标函数 - 逐个计算以便错误隔离
                objectives = self._calculate_objectives_safe(params_list, i)
                f[i, :] = objectives
                
                # 计算约束
                constraints = self._calculate_constraints_safe(params_list, i)
                g[i, :] = constraints
                
                # 记录评估统计
                if self.evaluation_count % 100 == 0:
                    logger.info(f"已评估 {self.evaluation_count} 个解")
                
            except Exception as e:
                logger.error(f"解 {i} 评估失败：{str(e)}")
                f[i, :] = self._get_default_objectives()
                g[i, :] = self._get_default_constraints()
        
        # 最终验证输出
        f = validate_array(f, "目标函数矩阵")
        g = validate_array(g, "约束矩阵")
        
        # 设置输出
        out["F"] = f
        out["G"] = g
        
        # 记录评估摘要
        self._log_evaluation_summary(f, g)

    def _validate_parameters(self, params_list: List[Dict]) -> bool:
        """验证参数列表的有效性"""
        try:
            if not params_list:
                return False
            
            required_keys = ['frequency', 'bandwidth', 'power', 'modulation', 'polarization']
            
            for params in params_list:
                if not all(key in params for key in required_keys):
                    return False
                
                # 基本数值验证
                if not (0 < params['frequency'] < 1e12):
                    return False
                if not (0 < params['bandwidth'] < 1e9):
                    return False
                if not (0 < params['power'] < 1000):
                    return False
            
            return True
            
        except Exception:
            return False

    def _calculate_objectives_safe(self, params_list: List[Dict], solution_id: int) -> np.ndarray:
        """安全计算目标函数"""
        objectives = np.zeros(5)
        
        try:
            # 可靠性目标
            try:
                obj_val = self.optimizer.objectives.reliability_objective(params_list)
                objectives[0] = self.stability_manager.validate_objective_value(obj_val)
            except Exception as e:
                logger.warning(f"解{solution_id}可靠性目标计算失败：{str(e)}")
                objectives[0] = -1.0  # 默认值
            
            # 频谱效率目标
            try:
                obj_val = self.optimizer.objectives.spectral_efficiency_objective(params_list)
                objectives[1] = self.stability_manager.validate_objective_value(obj_val)
            except Exception as e:
                logger.warning(f"解{solution_id}频谱效率目标计算失败：{str(e)}")
                objectives[1] = -1.0
            
            # 能量效率目标
            try:
                obj_val = self.optimizer.objectives.energy_efficiency_objective(params_list)
                objectives[2] = self.stability_manager.validate_objective_value(obj_val)
            except Exception as e:
                logger.warning(f"解{solution_id}能量效率目标计算失败：{str(e)}")
                objectives[2] = 50.0  # 默认中等能耗
            
            # 抗干扰目标
            try:
                obj_val = self.optimizer.objectives.interference_objective(params_list)
                objectives[3] = self.stability_manager.validate_objective_value(obj_val)
            except Exception as e:
                logger.warning(f"解{solution_id}抗干扰目标计算失败：{str(e)}")
                objectives[3] = -1.0
            
            # 环境适应性目标
            try:
                obj_val = self.optimizer.objectives.adaptability_objective(params_list)
                objectives[4] = self.stability_manager.validate_objective_value(obj_val)
            except Exception as e:
                logger.warning(f"解{solution_id}适应性目标计算失败：{str(e)}")
                objectives[4] = -1.0
                
        except Exception as e:
            logger.error(f"目标函数计算异常：{str(e)}")
            objectives = self._get_default_objectives()
        
        return objectives

    def _calculate_constraints_safe(self, params_list: List[Dict], solution_id: int) -> np.ndarray:
        """安全计算约束"""
        try:
            constraints = self.optimizer.constraints.evaluate_constraints(params_list)
            
            # 验证约束数组
            if len(constraints) != self.n_constr:
                logger.warning(f"解{solution_id}约束数量不匹配，进行调整")
                constraints = self._adjust_constraint_array(constraints)
            
            # 验证约束值
            for j, c_val in enumerate(constraints):
                constraints[j] = self.stability_manager.validate_constraint_value(c_val)
            
            return constraints
            
        except Exception as e:
            logger.warning(f"解{solution_id}约束计算失败：{str(e)}")
            return self._get_default_constraints()

    def _adjust_constraint_array(self, constraints: np.ndarray) -> np.ndarray:
        """调整约束数组大小"""
        current_size = len(constraints)
        
        if current_size < self.n_constr:
            # 填充满足的约束
            padding = np.full(self.n_constr - current_size, -0.1)
            return np.concatenate([constraints, padding])
        elif current_size > self.n_constr:
            # 截断
            return constraints[:self.n_constr]
        else:
            return constraints

    def _get_default_objectives(self) -> np.ndarray:
        """获取默认目标函数值（最差情况）"""
        return np.array([-100.0, -100.0, 100.0, -100.0, -100.0])

    def _get_default_constraints(self) -> np.ndarray:
        """获取默认约束值（轻微违反）"""
        return np.full(self.n_constr, 0.5)

    def _log_evaluation_summary(self, f: np.ndarray, g: np.ndarray):
        """记录评估摘要"""
        try:
            if self.evaluation_count % 500 == 0:  # 每500次评估记录一次
                # 目标函数统计
                f_stats = {
                    'reliability': {'min': f[:, 0].min(), 'max': f[:, 0].max(), 'mean': f[:, 0].mean()},
                    'spectral': {'min': f[:, 1].min(), 'max': f[:, 1].max(), 'mean': f[:, 1].mean()},
                    'energy': {'min': f[:, 2].min(), 'max': f[:, 2].max(), 'mean': f[:, 2].mean()},
                    'interference': {'min': f[:, 3].min(), 'max': f[:, 3].max(), 'mean': f[:, 3].mean()},
                    'adaptability': {'min': f[:, 4].min(), 'max': f[:, 4].max(), 'mean': f[:, 4].mean()}
                }
                
                # 约束统计
                violations = np.sum(g > 0, axis=1)
                constraint_stats = {
                    'total_violations': int(np.sum(violations)),
                    'feasible_solutions': int(np.sum(violations == 0)),
                    'max_violation': float(g.max())
                }
                
                logger.info(f"评估摘要（{self.evaluation_count}次）：")
                logger.info(f"  目标函数范围：{f_stats}")
                logger.info(f"  约束统计：{constraint_stats}")
                
        except Exception as e:
            logger.warning(f"评估摘要记录失败：{str(e)}")


class ImprovedCommunicationOptimizer:
    """
    改进的通信优化器
    集成所有改进模块，提供稳定、高效的优化能力
    """
    
    def __init__(self, task_data: Dict, env_data: Dict, constraint_data: Dict, 
                config: Any, neo4j_handler=None):
        """
        初始化改进的通信优化器
        
        参数:
        task_data: 任务数据
        env_data: 环境数据
        constraint_data: 约束数据
        config: 优化配置
        neo4j_handler: Neo4j数据库处理器
        """
        
        logger.info("初始化改进的通信优化器")
        
        # 数据预处理
        self.task_data = self._preprocess_task_data(task_data)
        self.env_data = self._preprocess_env_data(env_data)
        self.constraint_data = self._preprocess_constraint_data(constraint_data)
        self.config = config
        
        # 数值稳定性管理器
        self.stability_manager = get_stability_manager()
        
        # 保存数据库连接信息
        self._setup_database_connection(neo4j_handler)
        
        # 初始化改进的组件
        self.objectives = ImprovedObjectiveFunction(
            self.task_data, self.env_data, self.constraint_data, config
        )
        
        self.constraints = ImprovedConstraints(config, enable_soft_constraints=True)
        
        self.population_manager = ImprovedPopulationManager(config)
        
        # 设置问题维度
        self.setup_problem_dimensions()
        
        # 优化统计
        self.optimization_stats = {
            'start_time': None,
            'end_time': None,
            'total_evaluations': 0,
            'best_objectives': None,
            'convergence_data': []
        }
        
        logger.info("改进的通信优化器初始化完成")

    def _preprocess_task_data(self, task_data: Dict) -> Dict:
        """预处理任务数据"""
        try:
            processed_data = task_data.copy() if task_data else {}
            
            # 确保有通信链路数据
            if 'communication_links' not in processed_data:
                processed_data['communication_links'] = []
            
            # 预处理通信链路
            for link in processed_data['communication_links']:
                # 数值类型转换
                for key in ['distance', 'bandwidth', 'power']:
                    if key in link and isinstance(link[key], str):
                        link[key] = self._parse_numeric_value(link[key])
            
            return processed_data
            
        except Exception as e:
            logger.error(f"任务数据预处理失败：{str(e)}")
            return task_data or {}

    def _preprocess_env_data(self, env_data: Dict) -> Dict:
        """预处理环境数据"""
        try:
            processed_data = env_data.copy() if env_data else {}
            
            # 数值转换           
            numeric_keys = ['海况等级', '电磁干扰强度', '温度', '盐度', '深度']
            for key in numeric_keys:
                if key in processed_data and isinstance(processed_data[key], str):
                    processed_data[key] = self._parse_numeric_value(processed_data[key])
            
            # 确保有默认深度值
            if 'depth' not in processed_data and '深度' not in processed_data:
                processed_data['depth'] = 50
            
            return processed_data
            
        except Exception as e:
            logger.error(f"环境数据预处理失败：{str(e)}")
            return env_data or {}

    def _preprocess_constraint_data(self, constraint_data: Dict) -> Dict:
        """预处理约束数据"""
        try:
            processed_data = constraint_data.copy() if constraint_data else {}
            
            # 数值转换
            numeric_keys = ['最小可靠性要求', '最大时延要求', '最小信噪比', 
                          '频谱最小频率', '频谱最大频率', '带宽限制', '发射功率限制']
            
            for key in numeric_keys:
                if key in processed_data and isinstance(processed_data[key], str):
                    processed_data[key] = self._parse_numeric_value(processed_data[key])
            
            return processed_data
            
        except Exception as e:
            logger.error(f"约束数据预处理失败：{str(e)}")
            return constraint_data or {}

    def _setup_database_connection(self, neo4j_handler):
        """设置数据库连接"""
        try:
            if neo4j_handler:
                self.db_info = {
                    'uri': getattr(neo4j_handler, '_uri', 'bolt://localhost:7699'),
                    'user': getattr(neo4j_handler, '_user', 'neo4j'),
                    'password': getattr(neo4j_handler, '_password', '12345678')
                }
                
                logger.info(f"保存数据库连接信息：{self.db_info['uri']}")
                
                # 关闭原始连接，优化时重新创建
                try:
                    neo4j_handler.close()
                except:
                    pass
                self.neo4j_handler = None
            else:
                self.db_info = None
                self.neo4j_handler = None
                
        except Exception as e:
            logger.warning(f"数据库连接设置失败：{str(e)}")
            self.db_info = None
            self.neo4j_handler = None

    def setup_problem_dimensions(self):
        """设置问题维度和边界"""
        try:
            # 获取通信链路数量
            self.n_links = len(self.task_data.get('communication_links', []))
            logger.info(f"检测到 {self.n_links} 个通信链路")
            
            # 如果没有链路，创建默认链路
            if self.n_links == 0:
                logger.warning("没有通信链路，创建默认链路")
                self.n_links = 1
                self._create_default_link()
            
            # 设置问题维度
            self.n_vars = self.n_links * 5
            self.n_constraints = self._calculate_constraint_count()
            
            # 设置边界
            self._setup_bounds()
            
            logger.info(f"问题维度设置完成：变量={self.n_vars}, 约束={self.n_constraints}")
            
        except Exception as e:
            logger.error(f"问题维度设置失败：{str(e)}")
            # 使用默认配置
            self.n_links = 1
            self.n_vars = 5
            self.n_constraints = 4
            self._setup_bounds()

    def _create_default_link(self):
        """创建默认通信链路"""
        default_link = {
            'source_id': 1,
            'target_id': 2,
            'frequency_min': 1000e6,
            'frequency_max': 2000e6,
            'bandwidth': 10e6,
            'power': 10,
            'link_type': '默认通信',
            'comm_type': '短波通信'
        }
        
        if 'communication_links' not in self.task_data:
            self.task_data['communication_links'] = []
        
        self.task_data['communication_links'].append(default_link)

    def _calculate_constraint_count(self) -> int:
        """计算约束数量"""
        try:
            # 基本约束：每个链路的频率、功率、带宽边界约束
            base_constraints = self.n_links * 6  # 每个链路6个基本约束
            
            # 频率间隔约束
            if self.n_links > 1:
                spacing_constraints = self.n_links * (self.n_links - 1) // 2
                base_constraints += spacing_constraints
            
            # 软约束（SNR和时延）
            soft_constraints = self.n_links * 2
            base_constraints += soft_constraints
            
            return max(base_constraints, self.n_links * 4)  # 确保最小约束数量
            
        except Exception as e:
            logger.warning(f"约束数量计算失败：{str(e)}")
            return self.n_links * 4

    def _setup_bounds(self):
        """设置参数边界"""
        try:
            # 下界
            self.lower_bounds = np.array([
                *[self.config.freq_min] * self.n_links,
                *[self.config.bandwidth_min] * self.n_links,
                *[self.config.power_min] * self.n_links,
                *[0] * self.n_links,  # 调制方式
                *[0] * self.n_links   # 极化方式
            ])
            
            # 上界
            self.upper_bounds = np.array([
                *[self.config.freq_max] * self.n_links,
                *[self.config.bandwidth_max] * self.n_links,
                *[self.config.power_max] * self.n_links,
                *[3] * self.n_links,  # 调制方式
                *[3] * self.n_links   # 极化方式
            ])
            
            # 验证边界
            if len(self.lower_bounds) != self.n_vars or len(self.upper_bounds) != self.n_vars:
                raise ValueError(f"边界维度不匹配：期望{self.n_vars}, 实际{len(self.lower_bounds)}")
            
            if np.any(self.lower_bounds >= self.upper_bounds):
                raise ValueError("存在下界大于等于上界的情况")
                
        except Exception as e:
            logger.error(f"边界设置失败：{str(e)}")
            # 使用默认边界
            self.lower_bounds = np.zeros(self.n_vars)
            self.upper_bounds = np.ones(self.n_vars) * 1000

    def optimize(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        运行改进的优化过程
        
        返回:
        Tuple[np.ndarray, np.ndarray, Dict]: Pareto前沿、对应的解、历史数据
        """
        logger.info("开始改进的NSGA-II优化")
        self.optimization_stats['start_time'] = time.time()
        
        try:
            # 创建临时数据库连接
            temp_neo4j_handler = self._create_temp_neo4j_connection()
            
            # 创建问题实例
            problem = ImprovedCommunicationProblem(self)
            
            # 初始化种群
            initial_population = self._initialize_population(temp_neo4j_handler)
            
            # 创建算法
            algorithm = self._create_algorithm(initial_population)
            
            # 创建回调
            callbacks = self._create_callbacks()
            
            # 运行优化
            logger.info(f"开始优化：{self.config.n_generations}代，种群大小{self.config.population_size}")
            
            result = minimize(
                problem,
                algorithm,
                ('n_gen', self.config.n_generations),
                callback=callbacks,
                verbose=True
            )
            
            # 处理结果
            pareto_front, optimal_variables, history = self._process_optimization_result(
                result, callbacks
            )
            
            # 记录统计信息
            self._record_optimization_stats(pareto_front, optimal_variables, problem)
            
            # 清理临时连接
            if temp_neo4j_handler:
                temp_neo4j_handler.close()
            
            return pareto_front, optimal_variables, history
            
        except Exception as e:
            logger.error(f"优化过程异常：{str(e)}")
            import traceback
            traceback.print_exc()
            
            # 返回默认结果
            return self._create_fallback_result()

    def _create_temp_neo4j_connection(self):
        """创建临时Neo4j连接"""
        temp_handler = None
        try:
            if self.db_info:
                from data.neo4j_handler import Neo4jHandler
                temp_handler = Neo4jHandler(
                    uri=self.db_info['uri'],
                    user=self.db_info['user'],
                    password=self.db_info['password']
                )
                self.population_manager.neo4j_handler = temp_handler
                logger.info("临时Neo4j连接创建成功")
        except Exception as e:
            logger.warning(f"临时Neo4j连接创建失败：{str(e)}")
        
        return temp_handler

    def _initialize_population(self, temp_neo4j_handler):
        """初始化种群"""
        try:
            task_id = self.task_data.get('task_info', {}).get('task_id', 'unknown')
            
            initial_population = self.population_manager.initialize_population(
                task_id,
                self.n_vars,
                self.lower_bounds,
                self.upper_bounds,
                self.config.population_size
            )
            
            logger.info(f"种群初始化完成：{len(initial_population)}个个体")
            return initial_population
            
        except Exception as e:
            logger.error(f"种群初始化失败：{str(e)}")
            return None

    def _create_algorithm(self, initial_population):
        """创建NSGA-II算法"""
        try:
            # 选择采样策略
            if initial_population is not None:
                class CustomSampling(Sampling):
                    def __init__(self, initial_pop):
                        super().__init__()
                        self.initial_pop = initial_pop
                        
                    def _do(self, problem, n_samples, **kwargs):
                        return self.initial_pop
                
                sampling = CustomSampling(initial_population)
                logger.info("使用自定义初始种群")
            else:
                from pymoo.operators.sampling.rnd import FloatRandomSampling
                sampling = FloatRandomSampling()
                logger.info("使用随机初始种群")
            
            # 创建算法
            algorithm = NSGA2(
                pop_size=self.config.population_size,
                n_offsprings=self.config.population_size,
                sampling=sampling,
                crossover=SBX(
                    prob=self.config.crossover_prob, 
                    eta=self.config.crossover_eta
                ),
                mutation=PM(
                    prob=self.config.mutation_prob, 
                    eta=self.config.mutation_eta
                ),
                eliminate_duplicates=True
            )
            
            return algorithm
            
        except Exception as e:
            logger.error(f"算法创建失败：{str(e)}")
            raise

    def _create_callbacks(self):
        """创建回调函数"""
        try:
            # 历史记录回调
            history_callback = ImprovedHistoryCallback()
            
            # 自适应机制回调
            adaptive_callback = ImprovedAdaptiveMechanism(self.config)
            
            # 组合回调
            class CombinedCallback(Callback):
                def __init__(self, history_cb, adaptive_cb):
                    super().__init__()
                    self.history_cb = history_cb
                    self.adaptive_cb = adaptive_cb
                    
                def notify(self, algorithm):
                    try:
                        self.history_cb.notify(algorithm)
                        self.adaptive_cb.notify(algorithm)
                    except Exception as e:
                        logger.warning(f"回调执行失败：{str(e)}")
            
            combined_callback = CombinedCallback(history_callback, adaptive_callback)
            combined_callback.history_callback = history_callback  # 保存引用以便后续访问
            
            return combined_callback
            
        except Exception as e:
            logger.error(f"回调创建失败：{str(e)}")
            return ImprovedHistoryCallback()

    def _process_optimization_result(self, result, callbacks):
        """处理优化结果"""
        try:
            # 检查结果有效性
            if result is None or not hasattr(result, 'F') or result.F is None:
                logger.warning("优化结果无效，使用默认结果")
                return self._create_fallback_result()
            
            pareto_front = result.F
            optimal_variables = result.X
            
            # 获取历史数据
            if hasattr(callbacks, 'history_callback'):
                history = callbacks.history_callback.history
            else:
                history = getattr(callbacks, 'history', {})
            
            # 验证结果
            pareto_front = validate_array(pareto_front, "Pareto前沿")
            optimal_variables = validate_array(optimal_variables, "最优变量")
            
            logger.info(f"优化完成：找到{len(pareto_front)}个非支配解")
            
            return pareto_front, optimal_variables, history
            
        except Exception as e:
            logger.error(f"结果处理失败：{str(e)}")
            return self._create_fallback_result()

    def _record_optimization_stats(self, pareto_front, optimal_variables, problem):
        """记录优化统计信息"""
        try:
            self.optimization_stats['end_time'] = time.time()
            self.optimization_stats['total_evaluations'] = problem.evaluation_count
            
            if len(pareto_front) > 0:
                self.optimization_stats['best_objectives'] = {
                    'reliability': float(-pareto_front[:, 0].max()),  # 取反得到真实值
                    'spectral_efficiency': float(-pareto_front[:, 1].max()),
                    'energy_efficiency': float(pareto_front[:, 2].min()),
                    'interference': float(-pareto_front[:, 3].max()),
                    'adaptability': float(-pareto_front[:, 4].max())
                }
            
            duration = self.optimization_stats['end_time'] - self.optimization_stats['start_time']
            
            logger.info(f"优化统计：")
            logger.info(f"  耗时：{duration:.2f}秒")
            logger.info(f"  评估次数：{self.optimization_stats['total_evaluations']}")
            logger.info(f"  最优目标：{self.optimization_stats['best_objectives']}")
            
        except Exception as e:
            logger.warning(f"统计记录失败：{str(e)}")

    def _create_fallback_result(self):
        """创建备用结果"""
        try:
            # 创建中点解
            midpoint_solution = (self.lower_bounds + self.upper_bounds) / 2
            
            # 默认目标函数值
            default_objectives = np.array([[-1.0, -1.0, 50.0, -1.0, -1.0]])
            default_variables = midpoint_solution.reshape(1, -1)
            default_history = {'n_gen': [], 'f_min': [[] for _ in range(5)]}
            
            logger.info("使用备用优化结果")
            
            return default_objectives, default_variables, default_history
            
        except Exception as e:
            logger.error(f"备用结果创建失败：{str(e)}")
            # 最简结果
            return (np.array([[-100.0, -100.0, 100.0, -100.0, -100.0]]), 
                   np.zeros((1, self.n_vars)), 
                   {})

    def post_process_solutions(self, objectives: np.ndarray, variables: np.ndarray) -> List[Dict]:
        """对优化结果进行后处理"""
        try:
            results = []
            
            for i, (obj, var) in enumerate(zip(objectives, variables)):
                # 转换参数
                params = self.population_manager.solution_to_parameters(var, self.n_links)
                
                # 计算实际适应度值（取反负值目标）
                actual_objectives = {
                    'reliability': float(-obj[0]),
                    'spectral_efficiency': float(-obj[1]),
                    'energy_efficiency': float(obj[2]),
                    'interference': float(-obj[3]),
                    'adaptability': float(-obj[4])
                }
                
                # 计算加权目标
                weighted_obj = (
                    self.config.reliability_weight * actual_objectives['reliability'] +
                    self.config.spectral_weight * actual_objectives['spectral_efficiency'] +
                    self.config.energy_weight * (-actual_objectives['energy_efficiency']) +  # 能量效率是最小化
                    self.config.interference_weight * actual_objectives['interference'] +
                    self.config.adaptability_weight * actual_objectives['adaptability']
                )
                
                result = {
                    'solution_id': i,
                    'objectives': actual_objectives,
                    'parameters': params,
                    'weighted_objective': float(weighted_obj)
                }
                
                results.append(result)
            
            # 按加权目标排序
            results.sort(key=lambda x: x['weighted_objective'], reverse=True)
            
            logger.info(f"后处理完成：{len(results)}个解")
            
            return results
            
        except Exception as e:
            logger.error(f"结果后处理失败：{str(e)}")
            return []

    def _parse_numeric_value(self, value_str):
        """解析数值字符串"""
        try:
            if value_str is None:
                return 0.0
                
            if isinstance(value_str, (int, float)):
                return float(value_str)
            
            if isinstance(value_str, str):
                import re
                numeric_match = re.search(r'-?\d+\.?\d*', value_str)
                if numeric_match:
                    return float(numeric_match.group())
            
            return 0.0
            
        except Exception:
            return 0.0

    def optimize_traditional(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        传统NSGA-II优化（为了兼容性保留）
        
        返回:
        Tuple[np.ndarray, np.ndarray, Dict]: Pareto前沿、对应的解、历史数据
        """
        logger.info("执行传统优化模式")
        
        try:
            # 临时禁用历史引导
            original_strategies = self.population_manager.init_strategies.copy()
            
            # 设置为纯随机初始化
            self.population_manager.init_strategies = {
                'historical_guided': 0.0,
                'domain_knowledge': 0.0,
                'frequency_optimized': 0.0,
                'latin_hypercube': 0.3,
                'random_diverse': 0.7
            }
            
            # 创建临时数据库连接
            temp_neo4j_handler = self._create_temp_neo4j_connection()
            
            # 创建问题实例
            problem = ImprovedCommunicationProblem(self)
            
            # 使用纯随机初始化
            from pymoo.operators.sampling.rnd import FloatRandomSampling
            sampling = FloatRandomSampling()
            
            # 创建算法
            algorithm = NSGA2(
                pop_size=self.config.population_size,
                n_offsprings=self.config.population_size,
                sampling=sampling,  # 使用纯随机采样
                crossover=SBX(
                    prob=self.config.crossover_prob, 
                    eta=self.config.crossover_eta
                ),
                mutation=PM(
                    prob=self.config.mutation_prob, 
                    eta=self.config.mutation_eta
                ),
                eliminate_duplicates=True
            )

            # 创建历史记录回调（不使用自适应机制）
            history_callback = ImprovedHistoryCallback()
                
            # 运行优化
            logger.info(f"开始传统NSGA-II算法，共 {self.config.n_generations} 代")
            
            result = minimize(
                problem,
                algorithm,
                ('n_gen', self.config.n_generations),
                callback=history_callback,
                verbose=True
            )
            
            # 处理结果
            pareto_front, optimal_variables, history = self._process_optimization_result(
                result, history_callback
            )
            
            # 记录统计信息
            self._record_optimization_stats(pareto_front, optimal_variables, problem)
            
            # 清理临时连接
            if temp_neo4j_handler:
                temp_neo4j_handler.close()
            
            return pareto_front, optimal_variables, history
            
        except Exception as e:
            logger.error(f"传统优化过程异常：{str(e)}")
            import traceback
            traceback.print_exc()
            
            # 返回默认结果
            return self._create_fallback_result()
            
        finally:
            # 恢复原始策略
            try:
                self.population_manager.init_strategies = original_strategies
            except Exception as e:
                logger.warning(f"恢复初始化策略失败：{str(e)}")

    # 同时建议修改种群管理器的初始化策略以改善多样性
    def improve_initialization_strategies(self):
        """改进初始化策略以增强多样性"""
        self.population_manager.init_strategies = {
            'historical_guided': 0.2,     # 降低历史引导比例
            'domain_knowledge': 0.2,      # 保持领域知识比例
            'frequency_optimized': 0.2,   # 保持频率优化比例
            'latin_hypercube': 0.25,      # 增加LHS比例
            'random_diverse': 0.15        # 增加随机多样性
        }

    # 修改约束权重以获得更好的平衡
    def improve_constraint_weights(self):
        """改进约束权重配置"""
        if hasattr(self, 'constraints'):
            self.constraints.constraint_weights = {
                'frequency_bounds': 1.2,      # 提高频率约束重要性
                'power_bounds': 1.1,          # 适度提高功率约束
                'bandwidth_bounds': 1.0,      # 保持带宽约束
                'frequency_spacing': 1.3,     # 强化频率间隔约束
                'snr_requirement': 1.0,       # 提高SNR软约束权重
                'delay_requirement': 0.9      # 提高时延软约束权重
            }


class ImprovedHistoryCallback(Callback):
    """改进的历史记录回调"""
    
    def __init__(self):
        super().__init__()
        self.history = {
            'n_gen': [],
            'n_eval': [],
            'n_nds': [],
            'cv_min': [],
            'cv_avg': [],
            'f_min': [[] for _ in range(5)],
            'f_avg': [[] for _ in range(5)],
            'convergence_indicators': []
        }
        self.stability_manager = get_stability_manager()
    
    def notify(self, algorithm):
        """记录每代数据"""
        try:
            # 基本信息
            self.history['n_gen'].append(algorithm.n_gen)
            self.history['n_eval'].append(algorithm.evaluator.n_eval)
            self.history['n_nds'].append(len(algorithm.opt) if algorithm.opt is not None else 0)
            
            # 约束违反度
            if algorithm.pop is not None and len(algorithm.pop) > 0:
                feasible = algorithm.pop.get("feasible")
                cv = algorithm.pop.get("CV")
                
                if cv is not None and len(cv) > 0:
                    cv_min = float(cv.min())
                    cv_avg = float(cv.mean())
                else:
                    cv_min = cv_avg = 0.0
                
                self.history['cv_min'].append(cv_min)
                self.history['cv_avg'].append(cv_avg)
            else:
                self.history['cv_min'].append(0.0)
                self.history['cv_avg'].append(0.0)
            
            # 目标函数值
            if algorithm.pop is not None and len(algorithm.pop) > 0:
                F = algorithm.pop.get("F")
                if F is not None and len(F) > 0:
                    # 验证目标函数数组
                    F = validate_array(F, "目标函数数组")
                    
                    for i in range(min(5, F.shape[1])):
                        col = F[:, i]
                        # 数值稳定性检查
                        valid_mask = np.isfinite(col)
                        if np.any(valid_mask):
                            min_val = float(col[valid_mask].min())
                            avg_val = float(col[valid_mask].mean())
                        else:
                            min_val = avg_val = 0.0
                        
                        self.history['f_min'][i].append(min_val)
                        self.history['f_avg'][i].append(avg_val)
            
            # 收敛指标
            if len(self.history['f_min'][0]) > 10:  # 至少10代后开始计算
                convergence_indicator = self._calculate_convergence_indicator()
                self.history['convergence_indicators'].append(convergence_indicator)
                
        except Exception as e:
            logger.warning(f"历史记录失败：{str(e)}")

    def _calculate_convergence_indicator(self) -> float:
        """计算收敛指标"""
        try:
            # 使用最近10代的目标函数变化率
            if len(self.history['f_min'][0]) < 10:
                return 1.0
            
            recent_values = self.history['f_min'][0][-10:]
            if len(recent_values) < 2:
                return 1.0
            
            # 计算变化率
            changes = [abs(recent_values[i] - recent_values[i-1]) 
                      for i in range(1, len(recent_values))]
            
            avg_change = np.mean(changes) if changes else 0.0
            
            # 归一化收敛指标（值越小表示越收敛）
            return min(1.0, avg_change / 0.01)
            
        except Exception:
            return 1.0


class ImprovedAdaptiveMechanism(Callback):
    """改进的自适应机制"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.generation = 0
        self.stagnation_counter = 0
        self.last_best_values = []
        self.diversity_history = []
        
    def notify(self, algorithm):
        """自适应调整算法参数"""
        try:
            self.generation += 1
            
            if algorithm.pop is None or len(algorithm.pop) == 0:
                return
            
            # 计算种群多样性
            diversity = self._calculate_diversity(algorithm)
            self.diversity_history.append(diversity)
            
            # 检测停滞
            current_best = self._get_current_best(algorithm)
            self._update_stagnation_counter(current_best)
            
            # 自适应调整
            self._adaptive_parameter_adjustment(algorithm)
            
        except Exception as e:
            logger.warning(f"自适应机制执行失败：{str(e)}")

    def _calculate_diversity(self, algorithm) -> float:
        """计算种群多样性"""
        try:
            X = algorithm.pop.get("X")
            if X is None or len(X) < 2:
                return 1.0
            
            # 计算参数空间中的平均距离
            distances = []
            for i in range(len(X)):
                for j in range(i + 1, len(X)):
                    dist = np.linalg.norm(X[i] - X[j])
                    distances.append(dist)
            
            return np.mean(distances) if distances else 1.0
            
        except Exception:
            return 1.0

    def _get_current_best(self, algorithm) -> float:
        """获取当前最佳值"""
        try:
            F = algorithm.pop.get("F")
            if F is None or len(F) == 0:
                return 0.0
            
            # 使用第一个目标的最小值作为指标
            return float(F[:, 0].min())
            
        except Exception:
            return 0.0

    def _update_stagnation_counter(self, current_best: float):
        """更新停滞计数器"""
        try:
            if len(self.last_best_values) > 0:
                improvement = abs(self.last_best_values[-1] - current_best)
                if improvement < 0.001:  # 改进阈值
                    self.stagnation_counter += 1
                else:
                    self.stagnation_counter = 0
            
            self.last_best_values.append(current_best)
            
            # 保持历史长度
            if len(self.last_best_values) > 20:
                self.last_best_values = self.last_best_values[-20:]
                
        except Exception as e:
            logger.warning(f"停滞检测失败：{str(e)}")

    def _adaptive_parameter_adjustment(self, algorithm):
        """自适应参数调整"""
        try:
            # 根据停滞情况调整
            if self.stagnation_counter > 15:
                # 增加探索
                self._increase_exploration(algorithm)
                logger.info(f"检测到停滞({self.stagnation_counter}代)，增加探索")
                
            elif self.stagnation_counter == 0 and self.generation > 50:
                # 增加开发
                self._increase_exploitation(algorithm)
                
            # 根据多样性调整
            if len(self.diversity_history) > 5:
                recent_diversity = np.mean(self.diversity_history[-5:])
                if recent_diversity < 0.01:  # 多样性过低
                    self._restore_diversity(algorithm)
                    logger.info("多样性过低，执行多样性恢复")
                    
        except Exception as e:
            logger.warning(f"参数调整失败：{str(e)}")

    def _increase_exploration(self, algorithm):
        """增加探索"""
        try:
            if hasattr(algorithm, "mutation") and hasattr(algorithm.mutation, "prob"):
                algorithm.mutation.prob = min(0.5, algorithm.mutation.prob * 1.3)
            
            if hasattr(algorithm, "crossover") and hasattr(algorithm.crossover, "eta"):
                algorithm.crossover.eta = max(5, algorithm.crossover.eta * 0.7)
                
        except Exception as e:
            logger.warning(f"增加探索失败：{str(e)}")

    def _increase_exploitation(self, algorithm):
        """增加开发"""
        try:
            if hasattr(algorithm, "mutation") and hasattr(algorithm.mutation, "prob"):
                algorithm.mutation.prob = max(0.05, algorithm.mutation.prob * 0.8)
            
            if hasattr(algorithm, "crossover") and hasattr(algorithm.crossover, "eta"):
                algorithm.crossover.eta = min(50, algorithm.crossover.eta * 1.2)
                
        except Exception as e:
            logger.warning(f"增加开发失败：{str(e)}")

    def _restore_diversity(self, algorithm):
        """恢复多样性"""
        try:
            if hasattr(algorithm, "pop") and algorithm.pop is not None:
                pop_size = len(algorithm.pop)
                restart_count = pop_size // 4  # 重启25%的种群
                
                # 生成新的多样化个体（这里需要访问问题的边界）
                # 简化实现：增加变异强度
                if hasattr(algorithm, "mutation"):
                    original_prob = algorithm.mutation.prob
                    algorithm.mutation.prob = min(0.8, original_prob * 2)
                    
                    # 恢复原始概率（延迟恢复）
                    def restore_mutation():
                        algorithm.mutation.prob = original_prob
                    
                    # 这里可以实现延迟恢复逻辑
                    
        except Exception as e:
            logger.warning(f"多样性恢复失败：{str(e)}")


# # 向后兼容的包装器
# class CommunicationOptimizer(ImprovedCommunicationOptimizer):
#     """向后兼容的优化器类"""
    
#     def __init__(self, task_data: Dict, env_data: Dict, constraint_data: Dict, 
#                 config: Any, neo4j_handler=None):
#         super().__init__(task_data, env_data, constraint_data, config, neo4j_handler)
#         logger.info("使用改进的通信优化器（兼容性模式）")
        
#     # 将以下方法添加到 nsga2_optimizer_improved.py 中的 ImprovedCommunicationOptimizer 类

