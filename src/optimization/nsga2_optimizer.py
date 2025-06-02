import numpy as np
import time
import os  
import json
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.core.sampling import Sampling
from pymoo.operators.sampling.rnd import FloatRandomSampling
from typing import Dict, List, Tuple, Any, Optional
from .population import PopulationManager
from pymoo.core.callback import Callback


class HistoricalCaseSampling(Sampling):
    """
    基于历史案例的采样策略
    """
    def __init__(self, population_manager, task_id, n_var, xl, xu, pop_size):
        super().__init__()
        self.population_manager = population_manager
        self.task_id = task_id
        self.n_var = n_var
        self.xl = xl
        self.xu = xu
        self.pop_size = pop_size

    def _do(self, problem, n_samples, **kwargs):
        return self.population_manager.initialize_population(
            self.task_id, 
            self.n_var, 
            self.xl, 
            self.xu, 
            n_samples
        )

class CommunicationProblem(Problem):
    def __init__(self, optimizer):
        """
        初始化通信优化问题
        
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
    
    def _evaluate(self, X, out, *args, **kwargs):
        """
        评估解的目标函数值和约束 - 更详细的错误捕获
        """
        # 初始化目标函数值矩阵
        f = np.zeros((len(X), 5))
        g = np.zeros((len(X), self.n_constr))
        
        for i, xi in enumerate(X):
            try:
                n_links = self.optimizer.n_links
                # 将解向量转换为参数字典列表
                params_list = self.optimizer.population_manager.solution_to_parameters(
                    xi, 
                    n_links
                )
                
                # 打印参数检查
                if i == 0:  # 只打印第一个解的参数，避免日志过多
                    print(f"解向量示例: {xi[:10]}...")
                    print(f"参数列表示例: {params_list[0] if params_list else 'Empty'}")
                
                # 逐个尝试计算目标函数，并捕获各种错误
                try:
                    # 验证参数合法性
                    valid_params = True
                    for params in params_list:
                        if not all(key in params for key in ['frequency', 'bandwidth', 'power', 'modulation', 'polarization']):
                            print(f"警告: 参数字典缺少必要键: {params}")
                            valid_params = False
                            break
                    
                    if valid_params:
                        # 计算目标函数
                        f[i, 0] = self.optimizer.objectives.reliability_objective(params_list)
                        f[i, 1] = self.optimizer.objectives.spectral_efficiency_objective(params_list)
                        f[i, 2] = self.optimizer.objectives.energy_efficiency_objective(params_list)
                        f[i, 3] = self.optimizer.objectives.interference_objective(params_list)
                        f[i, 4] = self.optimizer.objectives.adaptability_objective(params_list)
                    else:
                        # 参数不合法，使用备用值
                        f[i, :] = np.array([-100, -100, 100, -100, -100])
                except Exception as e:
                    print(f"计算目标函数时出错: {str(e)}")
                    print(f"  解向量: {xi[:10]}...")
                    f[i, :] = np.array([-100, -100, 100, -100, -100])
                
                # 计算约束
                try:
                    constraints = self.optimizer.constraints.evaluate_constraints(params_list)
                    if len(constraints) != self.n_constr:
                        if len(constraints) < self.n_constr:
                            constraints = np.pad(constraints, (0, self.n_constr - len(constraints)), 
                                            constant_values=-1.0)
                        else:
                            constraints = constraints[:self.n_constr]
                    g[i] = constraints
                except Exception as e:
                    print(f"计算约束时出错: {str(e)}")
                    g[i] = np.ones(self.n_constr) * 0.5
            except Exception as e:
                print(f"评估解 {i} 时出错: {str(e)}")
                f[i, :] = np.array([-100, -100, 100, -100, -100])
                g[i] = np.ones(self.n_constr) * 0.5
        
        # 将目标函数值设置到输出
        out["F"] = f
        out["G"] = g

class CommunicationOptimizer:
    def __init__(self, task_data: Dict, env_data: Dict, constraint_data: Dict, 
                config: Any, neo4j_handler=None):
        """
        初始化通信优化器
        
        参数:
        task_data: 任务数据
        env_data: 环境数据
        constraint_data: 约束数据
        config: 优化配置
        neo4j_handler: Neo4j数据库处理器
        """

        # 添加环境数据预处理
        if env_data:
            # 确保深度是数值型
            if 'depth' in env_data and isinstance(env_data['depth'], str):
                env_data['depth'] = self._parse_numeric_value(env_data['depth'])
            if '深度' in env_data and isinstance(env_data['深度'], str):
                env_data['深度'] = self._parse_numeric_value(env_data['深度'])
        
        # 添加通信链路数据预处理
        if 'communication_links' in task_data:
            for link in task_data['communication_links']:
                # 预处理距离字段
                if 'distance' in link and isinstance(link['distance'], str):
                    link['distance'] = self._parse_numeric_value(link['distance'])
                # 预处理带宽字段
                if 'bandwidth' in link and isinstance(link['bandwidth'], str):
                    link['bandwidth'] = self._parse_numeric_value(link['bandwidth'])

        self.task_data = task_data
        self.env_data = env_data
        self.constraint_data = constraint_data
        self.config = config
        
            # 保存连接信息，使用创建Neo4j处理器时使用的原始参数
        if neo4j_handler:
            self.db_info = {
                'uri': getattr(neo4j_handler, '_uri', 'bolt://localhost:7699'),
                'user': getattr(neo4j_handler, '_user', 'neo4j'),
                'password': getattr(neo4j_handler, '_password', '12345678')
            }
            
            # 调试输出连接信息
            print(f"保存Neo4j连接信息: {self.db_info['uri']}, 用户: {self.db_info['user']}")
            
            # 关闭原始连接
            try:
                neo4j_handler.close()
            except:
                pass
            self.neo4j_handler = None
        else:
            self.db_info = None
            self.neo4j_handler = None
        
        # 初始化目标函数和约束
        from models.objectives import ObjectiveFunction
        from models.constraints import Constraints
        
        self.objectives = ObjectiveFunction(task_data, env_data, constraint_data,config)
        self.constraints = Constraints(config)
        
        # 初始化种群管理器
        self.population_manager = PopulationManager(config)
        
        # 设置问题维度
        self.setup_problem_dimensions()

    def setup_problem_dimensions(self):
        """设置问题维度和边界"""
        # 获取通信链路数量
        self.n_links = len(self.task_data.get('communication_links', []))
        print(f"当前任务包含 {self.n_links} 个通信链路")
        
        # 如果没有通信链路，创建一个默认链路
        if self.n_links == 0:
            # 为了继续优化流程，我们需要创建至少一个默认的通信链路
            print("警告：当前任务没有通信链路，将创建一个默认链路进行优化")
            self.n_links = 1
            
            # 添加一个默认的通信链路到任务数据中
            default_link = {
                'source_id': 1,
                'target_id': 2,
                'frequency_min': 1000e6,  # 1 GHz
                'frequency_max': 2000e6,  # 2 GHz
                'bandwidth': 10e6,  # 10 MHz
                'power': 10,  # 10 W
                'link_type': '默认通信',
                'comm_type': '短波通信'
            }
            
            if 'communication_links' not in self.task_data:
                self.task_data['communication_links'] = []
            
            self.task_data['communication_links'].append(default_link)
        
        # 设置问题维度：每个链路有5个参数
        self.n_vars = self.n_links * 5
        
        # 确保 n_vars 是正整数且为 5 的倍数
        if self.n_vars <= 0 or self.n_vars % 5 != 0:
            print(f"警告: 问题维度设置异常 ({self.n_vars})，调整为默认值")
            self.n_links = 1
            self.n_vars = 5
        
        # 输出确认
        print(f"确认问题维度: {self.n_vars}，通信链路数: {self.n_links}")
        
        # 设置约束数量：每个链路有多个约束
        self.n_constraints = self.n_links * 4
        
        # 设置边界
        self.lower_bounds = np.array([
            *[self.config.freq_min] * self.n_links,            # 频率下界
            *[self.config.bandwidth_min] * self.n_links,       # 带宽下界
            *[self.config.power_min] * self.n_links,           # 功率下界
            *[0] * self.n_links,                               # 调制方式下界
            *[0] * self.n_links                                # 极化方式下界
        ])
        
        self.upper_bounds = np.array([
            *[self.config.freq_max] * self.n_links,            # 频率上界
            *[self.config.bandwidth_max] * self.n_links,       # 带宽上界
            *[self.config.power_max] * self.n_links,           # 功率上界
            *[3] * self.n_links,                               # 调制方式上界
            *[3] * self.n_links                                # 极化方式上界
        ])
        
        # 检查边界长度是否正确
        if len(self.lower_bounds) != self.n_vars or len(self.upper_bounds) != self.n_vars:
            print(f"警告: 边界维度不匹配! 期望 {self.n_vars}, 实际: {len(self.lower_bounds)}/{len(self.upper_bounds)}")
        
        print(f"优化问题维度: {self.n_vars}, 约束数量: {self.n_constraints}")
    
    def optimize(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        运行优化过程，添加自适应机制
        
        返回:
        Tuple[np.ndarray, np.ndarray, Dict]: Pareto前沿、对应的解、以及历史数据
        """
        # 创建临时Neo4j处理器用于初始化种群
        temp_neo4j_handler = None
        try:
            if self.db_info:
                from data.neo4j_handler import Neo4jHandler
                temp_neo4j_handler = Neo4jHandler(
                    uri=self.db_info['uri'],
                    user=self.db_info['user'],
                    password=self.db_info['password']
                )
                self.population_manager.neo4j_handler = temp_neo4j_handler
        except Exception as e:
            print(f"创建临时Neo4j连接失败: {str(e)}")
        
        # 创建问题实例
        problem = CommunicationProblem(self)
        
        # 使用历史案例初始化种群
        task_id = self.task_data.get('task_info', {}).get('task_id', 'unknown')
        print(f"为任务 {task_id} 启动优化过程")
        
        # 获取初始种群
        initial_population = None
        if temp_neo4j_handler:
            try:
                initial_population = self.population_manager.initialize_population(
                    task_id,
                    self.n_vars,
                    self.lower_bounds,
                    self.upper_bounds,
                    self.config.population_size
                )
            except Exception as e:
                print(f"初始化种群失败: {str(e)}")
                initial_population = None
                
        # 如果初始化失败，创建一个随机初始种群
        if initial_population is None:
            print("使用随机初始化种群")
            from pymoo.operators.sampling.rnd import FloatRandomSampling
            sampling = FloatRandomSampling()
        else:
            # 自定义初始种群采样
            print(f"使用自定义初始种群，大小: {len(initial_population)}")
            class CustomSampling(Sampling):
                def __init__(self, initial_pop):
                    super().__init__()
                    self.initial_pop = initial_pop
                    
                def _do(self, problem, n_samples, **kwargs):
                    return self.initial_pop
                    
            sampling = CustomSampling(initial_population)
            
        # 创建和配置算法
        algorithm = NSGA2(
            pop_size=self.config.population_size,
            n_offsprings=self.config.population_size,
            sampling=sampling,
            crossover=SBX(prob=self.config.crossover_prob, eta=self.config.crossover_eta),
            mutation=PM(prob=self.config.mutation_prob, eta=self.config.mutation_eta),
            eliminate_duplicates=True
            )

        # 创建历史数据收集回调
        history_callback = HistoryCallback()
        
        # 创建自适应机制
        adaptive_mechanism = AdaptiveMechanism(self.config, problem)  
        
        # 创建结合了历史记录和自适应机制的回调函数
        class CombinedCallback(Callback):
            def __init__(self, history_callback, adaptive_mechanism):
                super().__init__()
                self.history_callback = history_callback
                self.adaptive_mechanism = adaptive_mechanism
                
            def notify(self, algorithm):
                # 记录历史
                self.history_callback.notify(algorithm)
                # 自适应调整
                self.adaptive_mechanism.update(algorithm)
        
        combined_callback = CombinedCallback(history_callback, adaptive_mechanism)
            
        # 运行优化
        print(f"开始运行NSGA-II算法，共 {self.config.n_generations} 代")
        try:
            res = minimize(
                problem,
                algorithm,
                ('n_gen', self.config.n_generations),
                callback=combined_callback,
                verbose=True
            )
                
            # 检查结果
            if res is None:
                print("警告: 优化未能返回有效结果")
                # 创建默认结果
                default_f = np.ones((1, 5)) * 1e5  # 默认目标函数值
                default_x = (self.lower_bounds + self.upper_bounds) / 2  # 默认参数值 (取中点)
                default_x = default_x.reshape(1, -1)  # 重塑为二维数组
                return default_f, default_x, getattr(history_callback, 'history', {})
                
            if not hasattr(res, 'F') or res.F is None or len(res.F) == 0:
                print("警告: 优化结果中缺少目标函数值")
                default_f = np.ones((1, 5)) * 1e5
                default_x = res.X if hasattr(res, 'X') and res.X is not None else (self.lower_bounds + self.upper_bounds) / 2
                default_x = default_x.reshape(1, -1)
                return default_f, default_x, getattr(history_callback, 'history', {})
                
            print(f"优化完成，找到 {len(res.F)} 个非支配解")
                
            # 返回Pareto前沿和对应的解
            return res.F, res.X, history_callback.history
                
        except Exception as e:
            # 确保清理临时资源
            if temp_neo4j_handler:
                temp_neo4j_handler.close()
                self.population_manager.neo4j_handler = None
            print(f"优化过程发生异常: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # 发生异常时仍然返回一个有效的结果
            default_f = np.ones((1, 5)) * 1e5  # 默认目标函数值
            default_x = (self.lower_bounds + self.upper_bounds) / 2  # 默认参数值 (取中点)
            default_x = default_x.reshape(1, -1)  # 重塑为二维数组
            return default_f, default_x, getattr(history_callback, 'history', {})
    
    def optimize_traditional(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        运行传统NSGA-II优化过程，不使用历史知识
        
        返回:
        Tuple[np.ndarray, np.ndarray, Dict]: Pareto前沿、对应的解、以及历史数据
        """
        # 创建问题实例
        problem = CommunicationProblem(self)
        
        # 使用完全随机初始化 - 这是与知识引导方法的主要区别
        sampling = FloatRandomSampling()
        
        algorithm = NSGA2(
            pop_size=self.config.population_size,
            n_offsprings=self.config.population_size,
            sampling=sampling,  # 使用纯随机采样
            crossover=SBX(prob=self.config.crossover_prob, eta=self.config.crossover_eta),
            mutation=PM(prob=self.config.mutation_prob, eta=self.config.mutation_eta),
            eliminate_duplicates=True
        )

        # 创建历史数据收集回调
        history_callback = HistoryCallback()
            
        # 运行优化
        print(f"开始运行传统NSGA-II算法，共 {self.config.n_generations} 代")
        try:
            res = minimize(
                problem,
                algorithm,
                ('n_gen', self.config.n_generations),
                callback=history_callback,
                verbose=True
            )
                
            # 检查结果
            if res is None:
                print("警告: 优化未能返回有效结果")
                # 创建默认结果
                default_f = np.ones((1, 5)) * 1e5  # 默认目标函数值
                default_x = (self.lower_bounds + self.upper_bounds) / 2  # 默认参数值
                default_x = default_x.reshape(1, -1)
                return default_f, default_x, getattr(history_callback, 'history', {})
                
            if not hasattr(res, 'F') or res.F is None or len(res.F) == 0:
                print("警告: 优化结果中缺少目标函数值")
                default_f = np.ones((1, 5)) * 1e5
                default_x = res.X if hasattr(res, 'X') and res.X is not None else (self.lower_bounds + self.upper_bounds) / 2
                default_x = default_x.reshape(1, -1)
                return default_f, default_x, getattr(history_callback, 'history', {})
                
            print(f"传统优化完成，找到 {len(res.F)} 个非支配解")
                
            # 返回Pareto前沿和对应的解
            return res.F, res.X, history_callback.history
                
        except Exception as e:
            print(f"传统优化过程发生异常: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # 发生异常时仍然返回一个有效的结果
            default_f = np.ones((1, 5)) * 1e5
            default_x = (self.lower_bounds + self.upper_bounds) / 2
            default_x = default_x.reshape(1, -1)
            return default_f, default_x, getattr(history_callback, 'history', {})

    
    def post_process_solutions(self, objectives: np.ndarray, variables: np.ndarray) -> List[Dict]:
        """
        对优化结果进行后处理
            
        参数:
        objectives: 目标函数值
        variables: 对应的解变量
            
        返回:
        处理后的结果列表
        """
        results = []
            
        for i, (obj, var) in enumerate(zip(objectives, variables)):
            # 将解向量转换为参数
            params = self.population_manager.solution_to_parameters(var, self.n_links)
                
            # 创建结果字典
            result = {
                'solution_id': i,
                'objectives': {
                    'reliability': float(-obj[0]),  # 注意取反，因为优化过程中是最小化目标
                    'spectral_efficiency': float(-obj[1]),
                    'energy_efficiency': float(obj[2]),
                    'interference': float(-obj[3]),
                    'adaptability': float(-obj[4])
                },
                'parameters': params
            }
                
            # 计算加权目标
            weighted_obj = (
                self.config.reliability_weight * (-obj[0]) +
                self.config.spectral_weight * (-obj[1]) +
                self.config.energy_weight * (-obj[2]) +
                self.config.interference_weight * (-obj[3]) +
                self.config.adaptability_weight * (-obj[4])
            )
            result['weighted_objective'] = float(weighted_obj)
                
            results.append(result)
            
        # 按加权目标排序（降序 - 越大越好）
        results.sort(key=lambda x: x['weighted_objective'], reverse=True)
            
        return results

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

    def compare_optimization_methods(self, task_id: str):
        """
        比较知识引导NSGA-II与传统NSGA-II的性能
        
        参数:
        task_id: 任务ID
        """
        # 执行知识引导优化
        start_time_kg = time.time()
        pareto_front_kg, optimal_variables_kg, history_kg = self.optimize()
        kg_time = time.time() - start_time_kg
        
        # 执行传统NSGA-II优化
        start_time_trad = time.time()
        pareto_front_trad, optimal_variables_trad, history_trad = self.optimize_traditional()
        trad_time = time.time() - start_time_trad
        
        # 后处理两种方法的结果
        results_kg = self.post_process_solutions(pareto_front_kg, optimal_variables_kg)
        results_trad = self.post_process_solutions(pareto_front_trad, optimal_variables_trad)
        
        # 保存对比结果
        comparison_results = {
            "task_id": task_id,
            "knowledge_guided": {
                "execution_time": kg_time,
                "pareto_front_size": len(pareto_front_kg),
                "best_weighted_objective": results_kg[0]["weighted_objective"] if results_kg else None,
                "convergence_history": history_kg,
                "objectives": [r["objectives"] for r in results_kg]
            },
            "traditional": {
                "execution_time": trad_time,
                "pareto_front_size": len(pareto_front_trad),
                "best_weighted_objective": results_trad[0]["weighted_objective"] if results_trad else None,
                "convergence_history": history_trad,
                "objectives": [r["objectives"] for r in results_trad]
            }
        }
        
        # 保存对比结果到文件
        comparison_file = os.path.join(self.config.output_dir, f"{task_id}_method_comparison.json")
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_results, f, ensure_ascii=False, indent=2)
        
        # 可视化对比结果
        self._visualize_comparison(history_kg, history_trad, task_id)
        
        return comparison_results

    def _visualize_comparison(self, history_kg, history_trad, task_id):
        """
        可视化两种方法的性能对比
        
        参数:
        history_kg: 知识引导方法的历史记录
        history_trad: 传统方法的历史记录
        task_id: 任务ID
        """
        try:
            import matplotlib.pyplot as plt
            
            # 1. 收敛速度对比
            plt.figure(figsize=(12, 8))
            
            # 收集各代最好的加权目标值
            kg_gen = history_kg.get("n_gen", [])
            trad_gen = history_trad.get("n_gen", [])
            
            # 假设有五个目标
            for obj_idx in range(5):
                plt.subplot(2, 3, obj_idx+1)
                
                # 只有在历史记录中存在目标数据时才绘制
                if "f_min" in history_kg and len(history_kg["f_min"]) > obj_idx:
                    kg_values = history_kg["f_min"][obj_idx]
                    plt.plot(kg_gen, kg_values, 'b-', label="Knowledge-Guided")
                
                if "f_min" in history_trad and len(history_trad["f_min"]) > obj_idx:
                    trad_values = history_trad["f_min"][obj_idx]
                    plt.plot(trad_gen, trad_values, 'r-', label="Traditional")
                
                plt.xlabel("Generation")
                plt.ylabel(f"Objective {obj_idx+1}")
                plt.grid(True)
                plt.legend()
                
            plt.subplot(2, 3, 6)
            # 绘制约束违反度
            if "cv_min" in history_kg and "cv_min" in history_trad:
                plt.plot(kg_gen, history_kg["cv_min"], 'b-', label="KG-Constraints")
                plt.plot(trad_gen, history_trad["cv_min"], 'r-', label="Trad-Constraints")
                plt.yscale('log')
                plt.xlabel("Generation")
                plt.ylabel("Constraint Violation")
                plt.grid(True)
                plt.legend()
                
            plt.tight_layout()
            plt.suptitle(f"Convergence Comparison for Task {task_id}")
            
            # 保存图表
            plt.savefig(os.path.join(self.config.output_dir, f"{task_id}_convergence_comparison.png"))
            plt.close()
            
        except Exception as e:
            print(f"可视化对比结果时出错: {str(e)}")

class HistoryCallback(Callback):
    """收集优化过程历史数据的回调"""
    
    def __init__(self):
        super().__init__()
        self.history = {
            'n_gen': [],
            'n_eval': [],
            'n_nds': [],
            'cv_min': [],
            'cv_avg': [],
            'f_min': None,
            'f_avg': None
        }
    
    def notify(self, algorithm):
        """每一代结束后记录数据"""
        self.history['n_gen'].append(algorithm.n_gen)
        self.history['n_eval'].append(algorithm.evaluator.n_eval)
        self.history['n_nds'].append(len(algorithm.opt))
        
        if algorithm.pop is not None and len(algorithm.pop) > 0:
            # 约束违反度
            feasible = algorithm.pop.get("feasible")
            if feasible is not None and any(~feasible):
                cv = algorithm.pop.get("CV")
                if cv is not None:
                    self.history['cv_min'].append(float(cv.min()))
                    self.history['cv_avg'].append(float(cv.mean()))
                else:
                    self.history['cv_min'].append(0.0)
                    self.history['cv_avg'].append(0.0)
            else:
                self.history['cv_min'].append(0.0)
                self.history['cv_avg'].append(0.0)
                
            # 目标函数值
            F = algorithm.pop.get("F")
            if F is not None and len(F) > 0:
                n_obj = F.shape[1]
                
                # 初始化目标函数历史数组
                if self.history['f_min'] is None:
                    self.history['f_min'] = [[] for _ in range(n_obj)]
                    self.history['f_avg'] = [[] for _ in range(n_obj)]
                
                # 记录每个目标的最小值和平均值
                for i in range(n_obj):
                    min_val = float(F[:, i].min())
                    avg_val = float(F[:, i].mean())
                    self.history['f_min'][i].append(min_val)
                    self.history['f_avg'][i].append(avg_val)

class AdaptiveMechanism:
    """更强大的自适应机制类"""
    
    def __init__(self, config, problem):
        self.config = config
        self.problem = problem
        self.generation = 0
        self.last_improvements = []
        self.stagnation_counter = 0
        self.population_diversity = []
        
    def update(self, algorithm):
        """基于优化进度更新算法参数"""
        self.generation += 1
        
        # 计算种群多样性
        if hasattr(algorithm, "pop") and algorithm.pop is not None:
            X = algorithm.pop.get("X")
            if X is not None:
                diversity = np.mean(np.std(X, axis=0))
                self.population_diversity.append(diversity)
                
                # 计算适应度改进情况
                F = algorithm.pop.get("F")
                if F is not None and len(F) > 0:
                    current_best = np.min(F[:, 0])  # 使用第一个目标作为指标
                    
                    if len(self.last_improvements) > 0:
                        improvement = self.last_improvements[-1] - current_best
                        
                        # 检测停滞
                        if improvement < 0.001:
                            self.stagnation_counter += 1
                        else:
                            self.stagnation_counter = 0
                            
                    self.last_improvements.append(current_best)
                    
                    # 进入探索阶段 - 增加变异，减少交叉的局部性
                    if self.stagnation_counter > 10:
                        if hasattr(algorithm, "mutation"):
                            algorithm.mutation.prob = min(0.4, algorithm.mutation.prob * 1.2)
                            
                        if hasattr(algorithm, "crossover"):
                            algorithm.crossover.eta = max(5, algorithm.crossover.eta * 0.8)
                            
                        print(f"检测到优化停滞，切换到探索模式: 变异={algorithm.mutation.prob:.2f}, 交叉参数={algorithm.crossover.eta:.1f}")
                        
                    # 进入开发阶段 - 较低变异，提高交叉的局部性
                    elif self.generation > 50 and self.stagnation_counter == 0:
                        if hasattr(algorithm, "mutation"):
                            algorithm.mutation.prob = max(0.05, algorithm.mutation.prob * 0.9)
                            
                        if hasattr(algorithm, "crossover"):
                            algorithm.crossover.eta = min(40, algorithm.crossover.eta * 1.1)
                            
                        if self.generation % 20 == 0:
                            print(f"优化进展良好，切换到开发模式: 变异={algorithm.mutation.prob:.2f}, 交叉参数={algorithm.crossover.eta:.1f}")
                    
                    # 如果多样性过低，进行部分重启
                    if len(self.population_diversity) > 10:
                        avg_diversity = np.mean(self.population_diversity[-10:])
                        if avg_diversity < 0.05 and self.generation > 30:
                            self._partial_restart(algorithm)
    
    def _partial_restart(self, algorithm):
        """部分重启种群"""
        if hasattr(algorithm, "pop") and algorithm.pop is not None:
            print(f"\n执行种群部分重启，增加多样性... (代数: {self.generation})")
            
            pop_size = len(algorithm.pop)
            elite_count = int(pop_size * 0.3)  # 保留30%精英
            
            X = algorithm.pop.get("X")
            F = algorithm.pop.get("F")
            
            if X is not None and F is not None:
                # 根据支配等级和拥挤距离排序
                if hasattr(algorithm, "opt"):
                    elite_indices = [i for i in range(min(elite_count, len(algorithm.opt)))]
                else:
                    # 如果没有优化集，根据第一个目标排序
                    elite_indices = np.argsort(F[:, 0])[:elite_count]
                    
                # 为其余70%生成多样化个体
                for i in range(elite_count, pop_size):
                    # 生成随机解
                    if random.random() < 0.5:
                        # 完全随机解
                        new_x = np.random.uniform(self.problem.xl, self.problem.xu)
                    else:
                        # 基于精英变异的解
                        parent_idx = random.choice(elite_indices)
                        parent = X[parent_idx].copy()
                        
                        # 大幅变异 (50%的基因)
                        mask = np.random.random(len(parent)) < 0.5
                        parent[mask] = np.random.uniform(
                            self.problem.xl[mask],
                            self.problem.xu[mask]
                        )
                        new_x = parent
                        
                    # 更新种群
                    algorithm.pop.set("X", i, new_x)
                    
            # 重置停滞计数器
            self.stagnation_counter = 0
            
            # 重置自适应参数
            if hasattr(algorithm, "mutation"):
                algorithm.mutation.prob = self.config.mutation_prob
                
            if hasattr(algorithm, "crossover"):
                algorithm.crossover.eta = self.config.crossover_eta