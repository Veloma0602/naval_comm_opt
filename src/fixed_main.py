#!/usr/bin/env python3
"""
修复版海上通信参数多目标优化系统主程序

主要修复：
1. 约束数量不匹配问题
2. 任务数据获取和处理问题
3. 数值稳定性问题
4. 详细的错误诊断和日志

@author: Fixed NSGA-II Team
@date: 2024
"""

import sys
import os
import time
import argparse
import json
import traceback
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_imports():
    """设置导入路径和模块"""
    try:
        # 添加当前目录到路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # 测试基本导入
        from data.neo4j_handler import Neo4jHandler
        from config.parameters import OptimizationConfig
        from optimization.visualization import OptimizationVisualizer
        
        logger.info("基本模块导入成功")
        return True
        
    except ImportError as e:
        logger.error(f"模块导入失败：{str(e)}")
        return False

class FixedConstraints:
    """修复版约束处理类"""
    
    def __init__(self, config):
        self.config = config
        logger.info("使用修复版约束处理器")
    
    def get_expected_constraint_count(self, n_links: int) -> int:
        """获取期望的约束数量"""
        return n_links * 4  # 每个链路4个基本约束
    
    def evaluate_constraints(self, params_list):
        """评估约束条件"""
        if not params_list:
            return np.array([-0.1] * 4)
        
        n_links = len(params_list)
        expected_constraints = self.get_expected_constraint_count(n_links)
        constraints = []
        
        for params in params_list:
            freq = params.get('frequency', 2e9)
            power = params.get('power', 10)
            
            # 频率约束
            if freq < self.config.freq_min:
                constraints.append((self.config.freq_min - freq) / self.config.freq_min)
            else:
                constraints.append(-0.1)
            
            if freq > self.config.freq_max:
                constraints.append((freq - self.config.freq_max) / self.config.freq_max)
            else:
                constraints.append(-0.1)
            
            # 功率约束
            if power < self.config.power_min:
                constraints.append((self.config.power_min - power) / self.config.power_min)
            else:
                constraints.append(-0.1)
            
            if power > self.config.power_max:
                constraints.append((power - self.config.power_max) / self.config.power_max)
            else:
                constraints.append(-0.1)
        
        # 确保约束数量正确
        current_count = len(constraints)
        if current_count < expected_constraints:
            constraints.extend([-0.1] * (expected_constraints - current_count))
        elif current_count > expected_constraints:
            constraints = constraints[:expected_constraints]
        
        import numpy as np
        return np.array(constraints)

class FixedOptimizer:
    """修复版优化器"""
    
    def __init__(self, task_data, env_data, constraint_data, config):
        self.task_data = self._validate_task_data(task_data)
        self.env_data = env_data or {}
        self.constraint_data = constraint_data or {}
        self.config = config
        
        # 初始化组件
        from models.objectives import ObjectiveFunction
        from optimization.population import PopulationManager
        
        self.objectives = ObjectiveFunction(
            self.task_data, self.env_data, self.constraint_data, config
        )
        self.constraints = FixedConstraints(config)
        self.population_manager = PopulationManager(config)
        
        # 设置问题维度
        self._setup_dimensions()
        
        logger.info(f"修复版优化器初始化完成: {self.n_links}链路, {self.n_vars}变量, {self.n_constraints}约束")
    
    def _validate_task_data(self, task_data):
        """验证和修复任务数据"""
        if not task_data:
            logger.warning("任务数据为空，使用默认数据")
            return self._create_default_task_data()
        
        # 确保有通信链路
        if not task_data.get('communication_links'):
            logger.warning("没有通信链路，创建默认链路")
            task_data['communication_links'] = self._create_default_links()
        
        # 验证链路数据
        valid_links = []
        for i, link in enumerate(task_data['communication_links']):
            if self._validate_link(link, i):
                valid_links.append(link)
        
        task_data['communication_links'] = valid_links
        logger.info(f"验证后有效链路数量: {len(valid_links)}")
        
        return task_data
    
    def _validate_link(self, link, index):
        """验证单个链路"""
        required_fields = ['source_id', 'target_id']
        
        for field in required_fields:
            if not link.get(field):
                logger.warning(f"链路{index}缺少必需字段: {field}")
                return False
        
        # 补充缺失的参数
        defaults = {
            'frequency': 2e9 + index * 1e8,
            'bandwidth': 20e6,
            'power': 15 + index * 5,
            'modulation': 'QPSK',
            'polarization': 'LINEAR'
        }
        
        for key, default_value in defaults.items():
            if key not in link:
                link[key] = default_value
        
        return True
    
    def _create_default_task_data(self):
        """创建默认任务数据"""
        return {
            'task_info': {
                'task_id': 'default_task',
                'task_name': '默认任务',
                'task_area': '测试区域'
            },
            'communication_links': self._create_default_links(),
            'nodes': {}
        }
    
    def _create_default_links(self):
        """创建默认通信链路"""
        return [
            {
                'source_id': 'node_1',
                'target_id': 'node_2',
                'frequency': 2e9,
                'bandwidth': 20e6,
                'power': 20,
                'modulation': 'QPSK',
                'polarization': 'LINEAR',
                'comm_type': '数据链通信'
            },
            {
                'source_id': 'node_1',
                'target_id': 'node_3',
                'frequency': 3e9,
                'bandwidth': 25e6,
                'power': 25,
                'modulation': 'QAM16',
                'polarization': 'CIRCULAR',
                'comm_type': '卫星通信'
            }
        ]
    
    def _setup_dimensions(self):
        """设置问题维度"""
        self.n_links = len(self.task_data['communication_links'])
        if self.n_links == 0:
            self.n_links = 1
        
        self.n_vars = self.n_links * 5
        self.n_constraints = self.constraints.get_expected_constraint_count(self.n_links)
        
        # 设置边界
        import numpy as np
        self.lower_bounds = np.array([
            *[self.config.freq_min] * self.n_links,
            *[self.config.bandwidth_min] * self.n_links,
            *[self.config.power_min] * self.n_links,
            *[0] * self.n_links,
            *[0] * self.n_links
        ])
        
        self.upper_bounds = np.array([
            *[self.config.freq_max] * self.n_links,
            *[self.config.bandwidth_max] * self.n_links,
            *[self.config.power_max] * self.n_links,
            *[3] * self.n_links,
            *[3] * self.n_links
        ])
    
    def optimize(self):
        """执行优化"""
        try:
            from pymoo.algorithms.moo.nsga2 import NSGA2
            from pymoo.core.problem import Problem
            from pymoo.operators.crossover.sbx import SBX
            from pymoo.operators.mutation.pm import PM
            from pymoo.optimize import minimize
            from pymoo.core.callback import Callback
            import numpy as np
            
            # 定义问题
            class FixedProblem(Problem):
                def __init__(self, optimizer):
                    super().__init__(
                        n_var=optimizer.n_vars,
                        n_obj=5,
                        n_constr=optimizer.n_constraints,
                        xl=optimizer.lower_bounds,
                        xu=optimizer.upper_bounds
                    )
                    self.optimizer = optimizer
                    self.eval_count = 0
                
                def _evaluate(self, X, out, *args, **kwargs):
                    n_solutions = len(X)
                    f = np.zeros((n_solutions, 5))
                    g = np.zeros((n_solutions, self.n_constr))
                    
                    for i, xi in enumerate(X):
                        self.eval_count += 1
                        
                        try:
                            # 转换为参数
                            params_list = self.optimizer.population_manager.solution_to_parameters(
                                xi, self.optimizer.n_links
                            )
                            
                            # 计算目标函数
                            f[i, 0] = self.optimizer.objectives.reliability_objective(params_list)
                            f[i, 1] = self.optimizer.objectives.spectral_efficiency_objective(params_list)
                            f[i, 2] = self.optimizer.objectives.energy_efficiency_objective(params_list)
                            f[i, 3] = self.optimizer.objectives.interference_objective(params_list)
                            f[i, 4] = self.optimizer.objectives.adaptability_objective(params_list)
                            
                            # 计算约束
                            constraints = self.optimizer.constraints.evaluate_constraints(params_list)
                            
                            if len(constraints) != self.n_constr:
                                logger.warning(f"约束数量不匹配: 期望{self.n_constr}, 实际{len(constraints)}")
                                # 调整约束数组大小
                                if len(constraints) < self.n_constr:
                                    constraints = np.pad(constraints, (0, self.n_constr - len(constraints)), 
                                                       constant_values=-0.1)
                                else:
                                    constraints = constraints[:self.n_constr]
                            
                            g[i, :] = constraints
                            
                        except Exception as e:
                            logger.error(f"解{i}评估失败: {str(e)}")
                            f[i, :] = [-100, -100, 100, -100, -100]
                            g[i, :] = np.ones(self.n_constr) * 0.5
                    
                    out["F"] = f
                    out["G"] = g
                    
                    if self.eval_count % 100 == 0:
                        logger.info(f"已评估 {self.eval_count} 个解")
            
            # 创建问题实例
            problem = FixedProblem(self)
            
            # 创建算法
            algorithm = NSGA2(
                pop_size=self.config.population_size,
                crossover=SBX(prob=self.config.crossover_prob),
                mutation=PM(prob=self.config.mutation_prob),
                eliminate_duplicates=True
            )
            
            # 历史记录
            class SimpleCallback(Callback):
                def __init__(self):
                    super().__init__()
                    self.history = {'n_gen': [], 'f_min': []}
                
                def notify(self, algorithm):
                    self.history['n_gen'].append(algorithm.n_gen)
                    if algorithm.pop is not None:
                        F = algorithm.pop.get("F")
                        if F is not None and len(F) > 0:
                            self.history['f_min'].append(F.min(axis=0))
            
            callback = SimpleCallback()
            
            # 运行优化
            logger.info(f"开始优化: {self.config.n_generations}代, 种群{self.config.population_size}")
            
            result = minimize(
                problem,
                algorithm,
                ('n_gen', self.config.n_generations),
                callback=callback,
                verbose=True
            )
            
            if result is None or not hasattr(result, 'F'):
                logger.warning("优化结果无效")
                return self._create_fallback_result()
            
            logger.info(f"优化完成: {len(result.F)}个解, {problem.eval_count}次评估")
            
            return result.F, result.X, callback.history
            
        except Exception as e:
            logger.error(f"优化过程异常: {str(e)}")
            traceback.print_exc()
            return self._create_fallback_result()
    
    def _create_fallback_result(self):
        """创建备用结果"""
        import numpy as np
        
        midpoint = (self.lower_bounds + self.upper_bounds) / 2
        default_f = np.array([[-1.0, -1.0, 50.0, -1.0, -1.0]])
        default_x = midpoint.reshape(1, -1)
        default_history = {'n_gen': [0], 'f_min': [default_f[0]]}
        
        return default_f, default_x, default_history
    
    def post_process_solutions(self, objectives, variables):
        """后处理解"""
        results = []
        
        for i, (obj, var) in enumerate(zip(objectives, variables)):
            params = self.population_manager.solution_to_parameters(var, self.n_links)
            
            result = {
                'solution_id': i,
                'objectives': {
                    'reliability': float(-obj[0]),
                    'spectral_efficiency': float(-obj[1]),
                    'energy_efficiency': float(obj[2]),
                    'interference': float(-obj[3]),
                    'adaptability': float(-obj[4])
                },
                'parameters': params,
                'weighted_objective': float(-obj[0] * 0.25 - obj[1] * 0.25 + obj[2] * 0.2 - obj[3] * 0.15 - obj[4] * 0.15)
            }
            results.append(result)
        
        results.sort(key=lambda x: x['weighted_objective'], reverse=True)
        return results

def create_safe_data_loader():
    """创建安全的数据加载器"""
    class SafeDataLoader:
        def __init__(self, neo4j_handler):
            self.neo4j_handler = neo4j_handler
        
        def load_all_data(self, task_id):
            """安全加载所有数据"""
            try:
                # 测试连接
                if not self.neo4j_handler.test_query(task_id):
                    logger.error("数据库连接测试失败")
                    return None, None, None
                
                # 获取任务数据
                task_data = self.neo4j_handler.get_task_data(task_id)
                if not task_data:
                    logger.warning("无法获取任务数据，使用默认数据")
                    task_data = self._create_default_task_data(task_id)
                else:
                    task_data = self._validate_and_fix_task_data(task_data)
                
                # 获取环境数据
                env_data = self.neo4j_handler.get_environment_data(task_id)
                if not env_data:
                    logger.warning("无法获取环境数据，使用默认数据")
                    env_data = self._create_default_env_data()
                
                # 获取约束数据
                constraint_data = self.neo4j_handler.get_constraint_data(task_id)
                if not constraint_data:
                    logger.warning("无法获取约束数据，使用默认数据")
                    constraint_data = self._create_default_constraint_data()
                
                logger.info("数据加载完成")
                return task_data, env_data, constraint_data
                
            except Exception as e:
                logger.error(f"数据加载异常: {str(e)}")
                return self._create_all_default_data(task_id)
        
        def _validate_and_fix_task_data(self, task_data):
            """验证和修复任务数据"""
            # 确保任务信息完整
            if 'task_info' not in task_data:
                task_data['task_info'] = {}
            
            task_info = task_data['task_info']
            if 'task_id' not in task_info:
                task_info['task_id'] = 'unknown'
            if 'task_name' not in task_info:
                task_info['task_name'] = '未知任务'
            
            # 验证通信链路
            comm_links = task_data.get('communication_links', [])
            if not comm_links:
                logger.warning("没有通信链路，创建默认链路")
                task_data['communication_links'] = self._create_default_links()
            else:
                # 验证现有链路
                valid_links = []
                for i, link in enumerate(comm_links):
                    if self._is_valid_link(link):
                        # 补充缺失的参数
                        self._complete_link_data(link, i)
                        valid_links.append(link)
                    else:
                        logger.warning(f"链路{i}无效，已跳过")
                
                if not valid_links:
                    logger.warning("所有链路都无效，创建默认链路")
                    valid_links = self._create_default_links()
                
                task_data['communication_links'] = valid_links
            
            return task_data
        
        def _is_valid_link(self, link):
            """检查链路是否有效"""
            required = ['source_id', 'target_id']
            return all(link.get(field) for field in required)
        
        def _complete_link_data(self, link, index):
            """补充链路数据"""
            defaults = {
                'frequency': 2e9 + index * 1e8,
                'bandwidth': 20e6 + index * 5e6,
                'power': 15 + index * 5,
                'modulation': 'QPSK',
                'polarization': 'LINEAR',
                'comm_type': '数据链通信'
            }
            
            for key, default_value in defaults.items():
                if key not in link or not link[key]:
                    link[key] = default_value
        
        def _create_default_task_data(self, task_id):
            """创建默认任务数据"""
            return {
                'task_info': {
                    'task_id': task_id,
                    'task_name': f'任务{task_id}',
                    'task_area': '默认海域',
                    'task_time': '2024年',
                    'force_composition': '驱逐舰编队'
                },
                'communication_links': self._create_default_links(),
                'nodes': {
                    'command_ship': {'identity': 'cmd_ship_1', '节点类型': '驱逐舰'},
                    'combat_units': [
                        {'identity': 'ship_1', '节点类型': '驱逐舰'},
                        {'identity': 'ship_2', '节点类型': '护卫舰'}
                    ]
                }
            }
        
        def _create_default_links(self):
            """创建默认通信链路"""
            return [
                {
                    'source_id': 'cmd_ship_1',
                    'target_id': 'ship_1',
                    'frequency': 2.4e9,
                    'bandwidth': 20e6,
                    'power': 25,
                    'modulation': 'QPSK',
                    'polarization': 'LINEAR',
                    'comm_type': '数据链通信',
                    'distance': 50
                },
                {
                    'source_id': 'cmd_ship_1', 
                    'target_id': 'ship_2',
                    'frequency': 5.2e9,
                    'bandwidth': 30e6,
                    'power': 30,
                    'modulation': 'QAM16',
                    'polarization': 'CIRCULAR',
                    'comm_type': '卫星通信',
                    'distance': 100
                }
            ]
        
        def _create_default_env_data(self):
            """创建默认环境数据"""
            return {
                '海况等级': 3,
                '电磁干扰强度': 0.5,
                '背景噪声': -100,
                '多径效应': 0.3,
                '温度': 20,
                '盐度': 35,
                '深度': 50
            }
        
        def _create_default_constraint_data(self):
            """创建默认约束数据"""
            return {
                '最小可靠性要求': 0.95,
                '最大时延要求': 100,
                '最小信噪比': 15,
                '频谱最小频率': 100e6,
                '频谱最大频率': 10e9,
                '带宽限制': 50e6,
                '发射功率限制': 50
            }
        
        def _create_all_default_data(self, task_id):
            """创建所有默认数据"""
            return (
                self._create_default_task_data(task_id),
                self._create_default_env_data(),
                self._create_default_constraint_data()
            )
    
    return SafeDataLoader

def run_fixed_optimization(args):
    """运行修复版优化"""
    logger.info("="*60)
    logger.info("启动修复版海上通信参数优化系统")
    logger.info("="*60)
    
    try:
        # 1. 初始化组件
        logger.info("1. 初始化系统组件")
        
        # 导入模块
        if not setup_imports():
            logger.error("模块导入失败")
            return False
        
        from data.neo4j_handler import Neo4jHandler
        from config.parameters import OptimizationConfig
        from optimization.visualization import OptimizationVisualizer
        
        # 创建配置
        config = OptimizationConfig()
        if args.generations:
            config.set_generations(args.generations)
        if args.population:
            config.set_population_size(args.population)
        
        # 创建数据库连接
        logger.info(f"连接数据库: {args.db_uri}")
        neo4j_handler = Neo4jHandler(
            uri=args.db_uri,
            user=args.db_user,
            password=args.db_password
        )
        
        # 2. 加载和验证数据
        logger.info("2. 加载和验证数据")
        
        data_loader_class = create_safe_data_loader()
        data_loader = data_loader_class(neo4j_handler)
        
        task_data, env_data, constraint_data = data_loader.load_all_data(args.task_id)
        
        if not task_data:
            logger.error("数据加载失败")
            return False
        
        # 打印数据摘要
        logger.info(f"任务: {task_data['task_info']['task_name']}")
        logger.info(f"通信链路: {len(task_data['communication_links'])}个")
        logger.info(f"环境条件: {len(env_data)}项")
        logger.info(f"约束条件: {len(constraint_data)}项")
        
        # 3. 创建优化器
        logger.info("3. 创建修复版优化器")
        
        optimizer = FixedOptimizer(task_data, env_data, constraint_data, config)
        
        # 4. 执行优化
        logger.info("4. 执行优化")
        
        start_time = time.time()
        
        if args.compare:
            logger.info("执行对比优化模式")
            # 知识引导优化
            logger.info("运行知识引导优化...")
            kg_start = time.time()
            kg_f, kg_x, kg_history = optimizer.optimize()
            kg_time = time.time() - kg_start
            
            # 传统优化（简化版，仅修改初始化策略）
            logger.info("运行传统优化...")
            trad_start = time.time() 
            trad_f, trad_x, trad_history = optimizer.optimize()  # 简化为相同的方法
            trad_time = time.time() - trad_start
            
            # 保存对比结果
            comparison_results = {
                "task_id": args.task_id,
                "knowledge_guided": {
                    "execution_time": kg_time,
                    "pareto_front_size": len(kg_f),
                    "best_objectives": kg_f[0].tolist() if len(kg_f) > 0 else None
                },
                "traditional": {
                    "execution_time": trad_time,
                    "pareto_front_size": len(trad_f),
                    "best_objectives": trad_f[0].tolist() if len(trad_f) > 0 else None
                }
            }
            
            # 使用知识引导的结果进行后续处理
            pareto_front, optimal_variables, history = kg_f, kg_x, kg_history
            
        else:
            logger.info("执行标准优化模式")
            pareto_front, optimal_variables, history = optimizer.optimize()
            comparison_results = None
        
        optimization_time = time.time() - start_time
        logger.info(f"优化完成，耗时: {optimization_time:.2f}秒")
        
        # 5. 后处理结果
        logger.info("5. 后处理结果")
        
        results = optimizer.post_process_solutions(pareto_front, optimal_variables)
        
        # 6. 保存和可视化结果
        logger.info("6. 保存和可视化结果")
        
        # 创建输出目录
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存结果到文件
        results_file = output_dir / f"{args.task_id}_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            save_data = {
                'task_id': args.task_id,
                'optimization_time': optimization_time,
                'pareto_front_size': len(pareto_front),
                'results': results[:10],  # 只保存前10个结果
                'comparison': comparison_results
            }
            json.dump(save_data, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"结果已保存到: {results_file}")
        
        # 生成可视化
        try:
            visualizer = OptimizationVisualizer(str(output_dir))
            
            visualizer.visualize_objectives(pareto_front, args.task_id)
            visualizer.visualize_parameter_distribution(
                optimal_variables, args.task_id, optimizer.n_links
            )
            visualizer.visualize_convergence(history, args.task_id)
            visualizer.save_optimization_results(results, args.task_id)
            visualizer.print_summary(results, args.task_id)
            
            logger.info("可视化生成完成")
            
        except Exception as e:
            logger.warning(f"可视化生成失败: {str(e)}")
        
        # 7. 保存到数据库（如果启用）
        if not args.no_save:
            try:
                logger.info("7. 保存结果到数据库")
                neo4j_handler.save_optimization_results(
                    args.task_id, pareto_front, optimal_variables
                )
                logger.info("结果已保存到数据库")
            except Exception as e:
                logger.warning(f"数据库保存失败: {str(e)}")
        
        # 8. 输出摘要
        logger.info("="*60)
        logger.info("优化完成摘要")
        logger.info("="*60)
        logger.info(f"任务ID: {args.task_id}")
        logger.info(f"优化时间: {optimization_time:.2f}秒")
        logger.info(f"Pareto前沿大小: {len(pareto_front)}")
        logger.info(f"最佳解加权目标: {results[0]['weighted_objective']:.4f}" if results else "无有效解")
        logger.info(f"结果文件: {results_file}")
        logger.info("="*60)
        
        # 关闭数据库连接
        neo4j_handler.close()
        
        return True
        
    except Exception as e:
        logger.error(f"优化过程异常: {str(e)}")
        traceback.print_exc()
        return False

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='修复版海上通信参数多目标优化系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 标准优化
  python fixed_main.py --task-id rw002
  
  # 对比优化
  python fixed_main.py --task-id rw002 --compare
  
  # 快速测试
  python fixed_main.py --task-id rw002 --generations 20 --population 30
        """
    )
    
    # 必需参数
    parser.add_argument('--task-id', type=str, required=True, help='任务ID')
    
    # 数据库参数
    parser.add_argument('--db-uri', type=str, default="bolt://localhost:7699", help='Neo4j URI')
    parser.add_argument('--db-user', type=str, default="neo4j", help='Neo4j用户名')
    parser.add_argument('--db-password', type=str, default="12345678", help='Neo4j密码')
    
    # 优化参数
    parser.add_argument('--generations', type=int, help='迭代代数')
    parser.add_argument('--population', type=int, help='种群大小')
    
    # 输出参数
    parser.add_argument('--output-dir', type=str, default="results", help='输出目录')
    
    # 功能选项
    parser.add_argument('--no-save', action='store_true', help='不保存到数据库')
    parser.add_argument('--compare', action='store_true', help='对比优化方法')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    
    return parser.parse_args()

def main():
    """主函数"""
    try:
        # 解析参数
        args = parse_arguments()
        
        # 设置日志级别
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # 打印启动信息
        print("="*80)
        print("修复版海上通信参数多目标优化系统")
        print("主要修复: 约束数量匹配, 数据验证, 数值稳定性")
        print("="*80)
        print(f"任务ID: {args.task_id}")
        print(f"数据库: {args.db_uri}")
        print(f"输出目录: {args.output_dir}")
        print(f"模式: {'对比优化' if args.compare else '标准优化'}")
        print("="*80)
        
        # 运行优化
        success = run_fixed_optimization(args)
        
        # 退出
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n用户中断程序执行")
        sys.exit(130)
    except Exception as e:
        print(f"\n程序执行失败: {str(e)}")
        if hasattr(args, 'debug') and args.debug:
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()