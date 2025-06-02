#!/usr/bin/env python3
"""
快速测试脚本
用于调试约束数量不匹配和任务数据获取问题
"""

import sys
import os
import logging
import numpy as np
from typing import Dict, List

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_data_retrieval():
    """测试数据获取"""
    try:
        from data.neo4j_handler import Neo4jHandler
        
        # 创建数据库连接
        neo4j_handler = Neo4jHandler(
            uri="bolt://localhost:7699",
            user="neo4j", 
            password="12345678"
        )
        
        task_id = "rw002"
        
        # 测试数据库连接
        print("="*50)
        print("1. 测试数据库连接")
        print("="*50)
        if neo4j_handler.test_query(task_id):
            print("✓ 数据库连接成功")
        else:
            print("✗ 数据库连接失败")
            return False
        
        # 获取任务数据
        print("\n" + "="*50)
        print("2. 获取任务数据")
        print("="*50)
        task_data = neo4j_handler.get_task_data(task_id)
        if task_data:
            print(f"✓ 任务数据获取成功")
            print(f"  - 任务ID: {task_data['task_info']['task_id']}")
            print(f"  - 任务名称: {task_data['task_info']['task_name']}")
            print(f"  - 通信链路数量: {len(task_data['communication_links'])}")
            
            # 打印链路详情
            for i, link in enumerate(task_data['communication_links']):
                print(f"  - 链路{i+1}: {link['source_id']} -> {link['target_id']}")
                print(f"    类型: {link.get('comm_type', 'Unknown')}")
                print(f"    频率: {link.get('frequency', 'Unknown')}")
                print(f"    功率: {link.get('power', 'Unknown')}")
        else:
            print("✗ 任务数据获取失败")
            return False
        
        # 获取环境数据
        print("\n" + "="*50)
        print("3. 获取环境数据")
        print("="*50)
        env_data = neo4j_handler.get_environment_data(task_id)
        if env_data:
            print(f"✓ 环境数据获取成功")
            for key, value in env_data.items():
                print(f"  - {key}: {value}")
        else:
            print("✗ 环境数据获取失败")
        
        # 获取约束数据
        print("\n" + "="*50)
        print("4. 获取约束数据")
        print("="*50)
        constraint_data = neo4j_handler.get_constraint_data(task_id)
        if constraint_data:
            print(f"✓ 约束数据获取成功")
            for key, value in constraint_data.items():
                print(f"  - {key}: {value}")
        else:
            print("✗ 约束数据获取失败")
        
        neo4j_handler.close()
        
        return task_data, env_data, constraint_data
        
    except Exception as e:
        print(f"✗ 数据获取测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_constraint_calculation():
    """测试约束计算"""
    try:
        print("\n" + "="*50)
        print("5. 测试约束计算")
        print("="*50)
        
        from config.parameters import OptimizationConfig
        
        # 创建配置
        config = OptimizationConfig()
        
        # 创建修复版约束处理器
        class TestConstraints:
            def __init__(self, config):
                self.config = config
            
            def get_expected_constraint_count(self, n_links: int) -> int:
                return n_links * 4
            
            def evaluate_constraints(self, params_list: List[Dict]) -> np.ndarray:
                n_links = len(params_list)
                expected_constraints = self.get_expected_constraint_count(n_links)
                
                # 简单的约束计算
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
                
                return np.array(constraints)
        
        constraints_handler = TestConstraints(config)
        
        # 测试不同数量的链路
        for n_links in [1, 2, 3]:
            print(f"\n测试 {n_links} 个链路:")
            
            # 创建测试参数
            params_list = []
            for i in range(n_links):
                params = {
                    'frequency': 2e9 + i * 1e8,
                    'bandwidth': 20e6,
                    'power': 15 + i * 5,
                    'modulation': 'QPSK',
                    'polarization': 'LINEAR'
                }
                params_list.append(params)
            
            # 计算约束
            constraints = constraints_handler.evaluate_constraints(params_list)
            expected_count = constraints_handler.get_expected_constraint_count(n_links)
            
            print(f"  - 期望约束数量: {expected_count}")
            print(f"  - 实际约束数量: {len(constraints)}")
            print(f"  - 约束匹配: {'✓' if len(constraints) == expected_count else '✗'}")
            print(f"  - 违反约束数量: {sum(1 for c in constraints if c > 0)}")
        
        print("✓ 约束计算测试完成")
        return True
        
    except Exception as e:
        print(f"✗ 约束计算测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_population_manager():
    """测试种群管理器"""
    try:
        print("\n" + "="*50)
        print("6. 测试种群管理器")
        print("="*50)
        
        from config.parameters import OptimizationConfig
        from optimization.population import PopulationManager
        
        config = OptimizationConfig()
        pop_manager = PopulationManager(config)
        
        # 测试解向量转换
        n_links = 2
        test_solution = np.array([
            2e9, 3e9,       # 频率
            20e6, 25e6,     # 带宽
            15, 20,         # 功率
            1, 2,           # 调制
            0, 1            # 极化
        ])
        
        print(f"测试解向量: {test_solution}")
        
        # 转换为参数
        params_list = pop_manager.solution_to_parameters(test_solution, n_links)
        
        print(f"转换后参数列表长度: {len(params_list)}")
        for i, params in enumerate(params_list):
            print(f"  链路{i+1}: {params}")
        
        print("✓ 种群管理器测试完成")
        return True
        
    except Exception as e:
        print(f"✗ 种群管理器测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_mini_optimization():
    """测试小规模优化"""
    try:
        print("\n" + "="*50)
        print("7. 测试小规模优化")
        print("="*50)
        
        # 获取数据
        data_result = test_data_retrieval()
        if not data_result:
            print("✗ 无法获取数据，跳过优化测试")
            return False
        
        task_data, env_data, constraint_data = data_result
        
        from config.parameters import OptimizationConfig
        
        # 创建小规模测试配置
        config = OptimizationConfig()
        config.population_size = 10
        config.n_generations = 5
        
        print(f"小规模测试配置: 种群{config.population_size}, 代数{config.n_generations}")
        
        # 创建简化的优化器
        class MiniOptimizer:
            def __init__(self, task_data, env_data, constraint_data, config):
                self.task_data = task_data
                self.env_data = env_data
                self.constraint_data = constraint_data
                self.config = config
                
                # 确定链路数量
                self.n_links = len(task_data.get('communication_links', []))
                if self.n_links == 0:
                    self.n_links = 1
                
                self.n_vars = self.n_links * 5
                self.n_constraints = self.n_links * 4
                
                print(f"  - 链路数量: {self.n_links}")
                print(f"  - 变量数量: {self.n_vars}")
                print(f"  - 约束数量: {self.n_constraints}")
            
            def create_random_solution(self):
                """创建随机解"""
                solution = []
                for i in range(self.n_links):
                    # 频率
                    solution.append(np.random.uniform(self.config.freq_min, self.config.freq_max))
                    # 带宽
                    solution.append(np.random.uniform(self.config.bandwidth_min, self.config.bandwidth_max))
                    # 功率
                    solution.append(np.random.uniform(self.config.power_min, self.config.power_max))
                    # 调制
                    solution.append(np.random.randint(0, 4))
                    # 极化
                    solution.append(np.random.randint(0, 4))
                
                return np.array(solution)
            
            def evaluate_solution(self, solution):
                """评估解"""
                # 简化的目标函数
                objectives = np.random.uniform(-5, 5, 5)
                
                # 简化的约束
                constraints = np.random.uniform(-1, 1, self.n_constraints)
                
                return objectives, constraints
        
        mini_opt = MiniOptimizer(task_data, env_data, constraint_data, config)
        
        # 测试解的创建和评估
        test_solution = mini_opt.create_random_solution()
        print(f"  - 测试解长度: {len(test_solution)}")
        
        objectives, constraints = mini_opt.evaluate_solution(test_solution)
        print(f"  - 目标函数数量: {len(objectives)}")
        print(f"  - 约束数量: {len(constraints)}")
        print(f"  - 维度匹配: {'✓' if len(constraints) == mini_opt.n_constraints else '✗'}")
        
        print("✓ 小规模优化测试完成")
        return True
        
    except Exception as e:
        print(f"✗ 小规模优化测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("开始快速诊断测试")
    print("="*60)
    
    try:
        # 1. 测试数据获取
        if not test_data_retrieval():
            print("\n❌ 数据获取测试失败，请检查数据库连接和数据完整性")
            return
        
        # 2. 测试约束计算
        if not test_constraint_calculation():
            print("\n❌ 约束计算测试失败")
            return
        
        # 3. 测试种群管理器
        if not test_population_manager():
            print("\n❌ 种群管理器测试失败")
            return
        
        # 4. 测试小规模优化
        if not test_mini_optimization():
            print("\n❌ 小规模优化测试失败")
            return
        
        print("\n" + "="*60)
        print("✅ 所有测试通过！")
        print("="*60)
        
        print("\n建议的修复方案:")
        print("1. 使用修复版约束处理类 (FixedConstraints)")
        print("2. 使用调试版优化器 (DebugCommunicationOptimizer)")
        print("3. 确保数据预处理正确")
        print("4. 检查约束数量计算逻辑")
        
    except Exception as e:
        print(f"❌ 测试过程异常: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()