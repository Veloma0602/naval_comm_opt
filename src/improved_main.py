"""
改进的海上通信参数多目标优化系统主程序

主要改进：
1. 集成所有改进模块
2. 增强的错误处理和日志记录
3. 性能监控和诊断
4. 自动化结果分析
5. 详细的进度报告

@author: Improved NSGA-II Team
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
from typing import Dict, Any, Optional

# 导入改进的模块
try:
    from optimization.nsga2_optimizer_improved import ImprovedCommunicationOptimizer
    from data.neo4j_handler import Neo4jHandler
    from config.parameters import OptimizationConfig
    from optimization.visualization import OptimizationVisualizer
    from utils.logging_config import setup_logging
    from optimization.numerical_stability import get_stability_manager
    from utils.validators import DataValidator
except ImportError as e:
    print(f"模块导入失败：{str(e)}")
    print("请确保所有依赖模块都已正确安装")
    sys.exit(1)

# 设置日志
logger = setup_logging()



class ImprovedOptimizationRunner:
    """
    改进的优化运行器
    负责协调整个优化流程，包括数据验证、性能监控、结果分析等
    """
    
    def __init__(self, args: argparse.Namespace):
        """
        初始化优化运行器
        
        参数:
        args: 命令行参数
        """
        self.args = args
        self.config = None
        self.neo4j_handler = None
        self.visualizer = None
        self.stability_manager = get_stability_manager()
        
        # 性能监控
        self.performance_stats = {
            'start_time': None,
            'end_time': None,
            'data_loading_time': 0,
            'optimization_time': 0,
            'post_processing_time': 0,
            'total_evaluations': 0,
            'memory_usage': {},
            'errors': []
        }
        
        # 结果存储
        self.optimization_results = {}
        
        logger.info(f"优化运行器初始化完成，任务ID: {args.task_id}")

    def run(self) -> bool:
        """
        运行完整的优化流程
        
        返回:
        bool: 是否成功完成
        """
        self.performance_stats['start_time'] = time.time()
        
        try:
            logger.info("=" * 60)
            logger.info("启动改进的海上通信参数多目标优化系统")
            logger.info("=" * 60)
            
            # 1. 系统环境检查
            if not self._check_system_environment():
                return False
            
            # 2. 初始化组件
            if not self._initialize_components():
                return False
            
            # 3. 数据加载和验证
            task_data, env_data, constraint_data = self._load_and_validate_data()
            if not task_data:
                return False
            
            # 4. 执行优化
            success = self._execute_optimization(task_data, env_data, constraint_data)
            if not success:
                return False
            
            # 5. 后处理和分析
            self._post_process_results()
            
            # 6. 生成报告
            self._generate_comprehensive_report()
            
            self.performance_stats['end_time'] = time.time()
            self._log_performance_summary()
            
            logger.info("优化流程成功完成！")
            return True
            
        except KeyboardInterrupt:
            logger.warning("用户中断优化过程")
            return False
        except Exception as e:
            logger.error(f"优化流程异常终止：{str(e)}")
            logger.error(traceback.format_exc())
            self.performance_stats['errors'].append(str(e))
            return False
        finally:
            self._cleanup_resources()

    def _check_system_environment(self) -> bool:
        """检查系统环境和依赖"""
        try:
            logger.info("检查系统环境...")
            
            # 检查Python版本
            python_version = sys.version_info
            if python_version < (3, 8):
                logger.error(f"Python版本过低：{python_version}，需要3.8+")
                return False
            
            # 检查必要的库
            required_packages = [
                'numpy', 'pymoo', 'neo4j', 'matplotlib', 'scipy'
            ]
            
            missing_packages = []
            for package in required_packages:
                try:
                    __import__(package)
                except ImportError:
                    missing_packages.append(package)
            
            if missing_packages:
                logger.error(f"缺少必要的包：{missing_packages}")
                return False
            
            # 检查输出目录
            output_dir = Path(self.args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 检查磁盘空间（简单检查）
            if not self._check_disk_space(output_dir):
                logger.warning("磁盘空间可能不足，请注意")
            
            logger.info("系统环境检查通过")
            return True
            
        except Exception as e:
            logger.error(f"系统环境检查失败：{str(e)}")
            return False

    def _check_disk_space(self, path: Path) -> bool:
        """检查磁盘空间"""
        try:
            import shutil
            total, used, free = shutil.disk_usage(path)
            free_gb = free // (1024**3)
            return free_gb > 1  # 至少1GB空闲空间
        except:
            return True  # 检查失败时假设足够

    def _initialize_components(self) -> bool:
        """初始化各个组件"""
        try:
            logger.info("初始化系统组件...")
            
            # 初始化优化配置
            self.config = OptimizationConfig()
            
            # 应用命令行参数覆盖
            if self.args.generations:
                self.config.set_generations(self.args.generations)
            if self.args.population:
                self.config.set_population_size(self.args.population)
            
            logger.info(f"优化配置：种群={self.config.population_size}, 代数={self.config.n_generations}")
            
            # 初始化数据库连接
            logger.info(f"连接Neo4j数据库：{self.args.db_uri}")
            self.neo4j_handler = Neo4jHandler(
                uri=self.args.db_uri,
                user=self.args.db_user,
                password=self.args.db_password
            )
            
            # 测试数据库连接
            if not self._test_database_connection():
                return False
            
            # 初始化可视化工具
            self.visualizer = OptimizationVisualizer(self.args.output_dir)
            
            logger.info("组件初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"组件初始化失败：{str(e)}")
            return False

    def _test_database_connection(self) -> bool:
        """测试数据库连接"""
        try:
            logger.info("测试数据库连接...")
            test_result = self.neo4j_handler.test_query(self.args.task_id)
            if test_result:
                logger.info("数据库连接测试成功")
                return True
            else:
                logger.error("数据库连接测试失败")
                return False
        except Exception as e:
            logger.error(f"数据库连接测试异常：{str(e)}")
            return False

    def _load_and_validate_data(self) -> tuple:
        """加载和验证数据"""
        data_start_time = time.time()
        
        try:
            logger.info(f"加载任务数据：{self.args.task_id}")
            
            # 加载任务数据
            task_data = self.neo4j_handler.get_task_data_improved(self.args.task_id)
            if not task_data:
                logger.error(f"无法获取任务 {self.args.task_id} 的数据")
                return None, None, None
            
            # 加载环境数据
            env_data = self.neo4j_handler.get_environment_data(self.args.task_id)
            if not env_data:
                logger.error(f"无法获取任务 {self.args.task_id} 的环境数据")
                return None, None, None
            
            # 加载约束数据
            constraint_data = self.neo4j_handler.get_constraint_data(self.args.task_id)
            if not constraint_data:
                logger.error(f"无法获取任务 {self.args.task_id} 的约束数据")
                return None, None, None
            
            # # 数据验证
            # if not self._validate_loaded_data(task_data, env_data, constraint_data):
            #     return None, None, None
            
            task_data, msg = self.neo4j_handler.validate_and_fix_task_data(task_data)
            if task_data is None:
                logger.error(f"数据验证失败: {msg}")
                return
            
            # 数据统计
            self._log_data_statistics(task_data, env_data, constraint_data)
            
            self.performance_stats['data_loading_time'] = time.time() - data_start_time
            logger.info(f"数据加载完成，耗时：{self.performance_stats['data_loading_time']:.2f}秒")
            
            return task_data, env_data, constraint_data
            
        except Exception as e:
            logger.error(f"数据加载失败：{str(e)}")
            return None, None, None

    def validate_task_data(self, task_data: Dict, env_data: Dict, 
                            constraint_data: Dict) -> bool:
        """验证加载的数据"""
        try:
            logger.info("验证数据完整性...")
            
            # 使用数据验证器
            validator = DataValidator()
            
            # 验证任务数据
            if not validator.validate_task_data(task_data.get('task_info', {})):
                logger.error("任务数据验证失败")
                return False
            
            # 验证环境数据
            if not validator.validate_environment_data(env_data):
                logger.error("环境数据验证失败")
                return False
            
            # 验证约束数据
            if not validator.validate_constraint_data(constraint_data):
                logger.error("约束数据验证失败")
                return False
            
            # 检查通信链路
            comm_links = task_data.get('communication_links', [])
            if not comm_links:
                logger.warning("没有找到通信链路，将创建默认链路")
            else:
                logger.info(f"发现 {len(comm_links)} 个通信链路")
            
            logger.info("数据验证通过")
            return True
            
        except Exception as e:
            logger.error(f"数据验证异常：{str(e)}")
            return False

    def _log_data_statistics(self, task_data: Dict, env_data: Dict, 
                           constraint_data: Dict):
        """记录数据统计信息"""
        try:
            logger.info("数据统计：")
            
            # 任务信息
            task_info = task_data.get('task_info', {})
            logger.info(f"  任务名称：{task_info.get('task_name', 'Unknown')}")
            logger.info(f"  任务区域：{task_info.get('task_area', 'Unknown')}")
            logger.info(f"  兵力组成：{task_info.get('force_composition', 'Unknown')}")
            
            # 通信链路
            comm_links = task_data.get('communication_links', [])
            logger.info(f"  通信链路数量：{len(comm_links)}")
            
            # 环境条件
            logger.info(f"  海况等级：{env_data.get('海况等级', 'Unknown')}")
            logger.info(f"  电磁干扰强度：{env_data.get('电磁干扰强度', 'Unknown')}")
            
            # 约束条件
            logger.info(f"  最小可靠性要求：{constraint_data.get('最小可靠性要求', 'Unknown')}")
            logger.info(f"  最大时延要求：{constraint_data.get('最大时延要求', 'Unknown')}")
            
        except Exception as e:
            logger.warning(f"数据统计记录失败：{str(e)}")

    def _execute_optimization(self, task_data: Dict, env_data: Dict, 
                            constraint_data: Dict) -> bool:
        """执行优化过程"""
        opt_start_time = time.time()
        
        try:
            logger.info("开始优化过程...")
            
            # 初始化优化器
            optimizer = ImprovedCommunicationOptimizer(
                task_data=task_data,
                env_data=env_data,
                constraint_data=constraint_data,
                config=self.config,
                neo4j_handler=self.neo4j_handler
            )
            
            # 执行优化
            if self.args.compare:
                logger.info("执行优化方法对比")
                comparison_results = self._run_comparison_optimization(optimizer)
                self.optimization_results['comparison'] = comparison_results
            else:
                logger.info("执行标准优化")
                optimization_results = self._run_standard_optimization(optimizer)
                self.optimization_results['standard'] = optimization_results
            
            self.performance_stats['optimization_time'] = time.time() - opt_start_time
            logger.info(f"优化完成，耗时：{self.performance_stats['optimization_time']:.2f}秒")
            
            return True
            
        except Exception as e:
            logger.error(f"优化执行失败：{str(e)}")
            logger.error(traceback.format_exc())
            return False

    def _run_standard_optimization(self, optimizer) -> Dict:
        """运行标准优化"""
        try:
            # 记录开始状态
            logger.info("运行改进的NSGA-II算法...")
            
            # 执行优化
            pareto_front, optimal_variables, history = optimizer.optimize()
            
            # 后处理结果
            processed_results = optimizer.post_process_solutions(pareto_front, optimal_variables)
            
            # 保存结果到数据库
            if self.args.no_save:
                logger.info("已禁用保存到Neo4j")
            else:
                logger.info("保存优化结果到Neo4j数据库")
                self.neo4j_handler.save_optimization_results(
                    task_id=self.args.task_id,
                    pareto_front=pareto_front,
                    optimal_variables=optimal_variables
                )
            
            # 构建结果字典
            results = {
                'pareto_front': pareto_front,
                'optimal_variables': optimal_variables,
                'history': history,
                'processed_results': processed_results,
                'optimizer_stats': optimizer.optimization_stats
            }
            
            logger.info(f"标准优化完成，找到 {len(pareto_front)} 个非支配解")
            
            return results
            
        except Exception as e:
            logger.error(f"标准优化失败：{str(e)}")
            raise

    def _run_comparison_optimization(self, optimizer) -> Dict:
        """运行对比优化"""
        try:
            logger.info("执行知识引导 vs 传统NSGA-II对比")
            
            # 知识引导优化
            logger.info("运行知识引导NSGA-II...")
            kg_start = time.time()
            kg_pareto, kg_variables, kg_history = optimizer.optimize()
            kg_time = time.time() - kg_start
            kg_results = optimizer.post_process_solutions(kg_pareto, kg_variables)
            
            # 传统优化
            logger.info("运行传统NSGA-II...")
            trad_start = time.time()
            trad_pareto, trad_variables, trad_history = optimizer.optimize_traditional()
            trad_time = time.time() - trad_start
            trad_results = optimizer.post_process_solutions(trad_pareto, trad_variables)
            
            # 构建对比结果
            comparison_results = {
                "task_id": self.args.task_id,
                "knowledge_guided": {
                    "execution_time": kg_time,
                    "pareto_front_size": len(kg_pareto),
                    "best_weighted_objective": kg_results[0]["weighted_objective"] if kg_results else None,
                    "convergence_history": kg_history,
                    "results": kg_results[:5]  # 只保存前5个结果
                },
                "traditional": {
                    "execution_time": trad_time,
                    "pareto_front_size": len(trad_pareto),
                    "best_weighted_objective": trad_results[0]["weighted_objective"] if trad_results else None,
                    "convergence_history": trad_history,
                    "results": trad_results[:5]
                },
                "improvement": {
                    "time_ratio": kg_time / trad_time if trad_time > 0 else 1.0,
                    "quality_improvement": 0.0,
                    "convergence_improvement": 0.0
                }
            }
            
            # 计算改进指标
            if kg_results and trad_results:
                kg_best = kg_results[0]["weighted_objective"]
                trad_best = trad_results[0]["weighted_objective"]
                if trad_best != 0:
                    comparison_results["improvement"]["quality_improvement"] = \
                        (kg_best - trad_best) / abs(trad_best) * 100
            
            logger.info("对比优化完成")
            logger.info(f"知识引导：{kg_time:.2f}秒，{len(kg_pareto)}个解")
            logger.info(f"传统方法：{trad_time:.2f}秒，{len(trad_pareto)}个解")
            
            return comparison_results
            
        except Exception as e:
            logger.error(f"对比优化失败：{str(e)}")
            raise

    def _post_process_results(self):
        """后处理和分析结果"""
        post_start_time = time.time()
        
        try:
            logger.info("开始后处理结果...")
            
            if 'standard' in self.optimization_results:
                self._post_process_standard_results()
            
            if 'comparison' in self.optimization_results:
                self._post_process_comparison_results()
            
            self.performance_stats['post_processing_time'] = time.time() - post_start_time
            logger.info(f"后处理完成，耗时：{self.performance_stats['post_processing_time']:.2f}秒")
            
        except Exception as e:
            logger.error(f"后处理失败：{str(e)}")

    def _post_process_standard_results(self):
        """后处理标准结果"""
        try:
            results = self.optimization_results['standard']
            
            # 生成可视化
            logger.info("生成结果可视化...")
            
            self.visualizer.visualize_objectives(
                results['pareto_front'], self.args.task_id
            )
            
            self.visualizer.visualize_parameter_distribution(
                results['optimal_variables'], 
                self.args.task_id,
                len(results['processed_results'][0]['parameters']) if results['processed_results'] else 1
            )
            
            self.visualizer.visualize_convergence(
                results['history'], self.args.task_id
            )
            
            # 保存结果报告
            self.visualizer.save_optimization_results(
                results['processed_results'], self.args.task_id
            )
            
            # 打印摘要
            self.visualizer.print_summary(
                results['processed_results'], self.args.task_id
            )
            
        except Exception as e:
            logger.error(f"标准结果后处理失败：{str(e)}")

    def _post_process_comparison_results(self):
        """后处理对比结果"""
        try:
            comparison = self.optimization_results['comparison']
            
            # 保存对比结果
            comparison_file = os.path.join(
                self.args.output_dir, 
                f"{self.args.task_id}_method_comparison.json"
            )
            
            with open(comparison_file, 'w', encoding='utf-8') as f:
                json.dump(comparison, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"对比结果已保存到：{comparison_file}")
            
            # 生成对比可视化
            self._create_comparison_visualization(comparison)
            
        except Exception as e:
            logger.error(f"对比结果后处理失败：{str(e)}")

    def _create_comparison_visualization(self, comparison: Dict):
        """创建对比可视化"""
        try:
            import matplotlib.pyplot as plt
            
            # 性能对比图
            methods = ['Knowledge-Guided', 'Traditional']
            times = [
                comparison['knowledge_guided']['execution_time'],
                comparison['traditional']['execution_time']
            ]
            qualities = [
                comparison['knowledge_guided']['best_weighted_objective'] or 0,
                comparison['traditional']['best_weighted_objective'] or 0
            ]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # 执行时间对比
            ax1.bar(methods, times, color=['blue', 'red'])
            ax1.set_ylabel('Execution Time (s)')
            ax1.set_title('Execution Time Comparison')
            ax1.grid(True, alpha=0.3)
            
            # 解质量对比
            ax2.bar(methods, qualities, color=['blue', 'red'])
            ax2.set_ylabel('Best Weighted Objective')
            ax2.set_title('Solution Quality Comparison')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            comparison_plot_file = os.path.join(
                self.args.output_dir,
                f"{self.args.task_id}_method_comparison.png"
            )
            plt.savefig(comparison_plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"对比可视化已保存到：{comparison_plot_file}")
            
        except Exception as e:
            logger.warning(f"对比可视化创建失败：{str(e)}")

    def _generate_comprehensive_report(self):
        """生成综合报告"""
        try:
            logger.info("生成综合优化报告...")
            
            report_file = os.path.join(
                self.args.output_dir,
                f"{self.args.task_id}_comprehensive_report.md"
            )
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(self._create_report_content())
            
            logger.info(f"综合报告已保存到：{report_file}")
            
        except Exception as e:
            logger.error(f"综合报告生成失败：{str(e)}")

    def _create_report_content(self) -> str:
        """创建报告内容"""
        try:
            total_time = self.performance_stats['end_time'] - self.performance_stats['start_time']
            
            report = f"""# 海上通信参数优化综合报告

## 任务信息
- **任务ID**: {self.args.task_id}
- **优化时间**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.performance_stats['start_time']))}
- **总耗时**: {total_time:.2f} 秒

## 性能统计
- **数据加载时间**: {self.performance_stats['data_loading_time']:.2f} 秒
- **优化执行时间**: {self.performance_stats['optimization_time']:.2f} 秒
- **后处理时间**: {self.performance_stats['post_processing_time']:.2f} 秒

## 优化配置
- **种群大小**: {self.config.population_size}
- **迭代代数**: {self.config.n_generations}
- **变异概率**: {self.config.mutation_prob}
- **交叉概率**: {self.config.crossover_prob}

"""
            
            # 添加标准优化结果
            if 'standard' in self.optimization_results:
                report += self._add_standard_results_to_report()
            
            # 添加对比结果
            if 'comparison' in self.optimization_results:
                report += self._add_comparison_results_to_report()
            
            # 添加改进建议
            report += self._add_improvement_suggestions()
            
            return report
            
        except Exception as e:
            logger.error(f"报告内容创建失败：{str(e)}")
            return f"# 报告生成失败\n\n错误信息：{str(e)}"

    def _add_standard_results_to_report(self) -> str:
        """添加标准结果到报告"""
        try:
            results = self.optimization_results['standard']
            processed = results['processed_results']
            
            if not processed:
                return "\n## 标准优化结果\n无有效结果\n"
            
            best_result = processed[0]
            
            content = f"""
## 标准优化结果

### 最优解
- **加权目标值**: {best_result['weighted_objective']:.4f}
- **可靠性**: {best_result['objectives']['reliability']:.4f}
- **频谱效率**: {best_result['objectives']['spectral_efficiency']:.4f}
- **能量效率**: {best_result['objectives']['energy_efficiency']:.4f}
- **抗干扰性**: {best_result['objectives']['interference']:.4f}
- **环境适应性**: {best_result['objectives']['adaptability']:.4f}

### Pareto前沿统计
- **非支配解数量**: {len(results['pareto_front'])}
- **目标函数维度**: {results['pareto_front'].shape[1]}

"""
            return content
            
        except Exception as e:
            logger.warning(f"标准结果报告添加失败：{str(e)}")
            return "\n## 标准优化结果\n报告生成失败\n"

    def _add_comparison_results_to_report(self) -> str:
        """添加对比结果到报告"""
        try:
            comparison = self.optimization_results['comparison']
            
            content = f"""
## 方法对比结果

### 知识引导NSGA-II
- **执行时间**: {comparison['knowledge_guided']['execution_time']:.2f} 秒
- **Pareto前沿大小**: {comparison['knowledge_guided']['pareto_front_size']}
- **最佳加权目标**: {comparison['knowledge_guided']['best_weighted_objective']:.4f}

### 传统NSGA-II
- **执行时间**: {comparison['traditional']['execution_time']:.2f} 秒
- **Pareto前沿大小**: {comparison['traditional']['pareto_front_size']}
- **最佳加权目标**: {comparison['traditional']['best_weighted_objective']:.4f}

### 改进评估
- **时间比率**: {comparison['improvement']['time_ratio']:.2f}
- **质量提升**: {comparison['improvement']['quality_improvement']:.2f}%

"""
            return content
            
        except Exception as e:
            logger.warning(f"对比结果报告添加失败：{str(e)}")
            return "\n## 方法对比结果\n报告生成失败\n"

    def _add_improvement_suggestions(self) -> str:
        """添加改进建议"""
        suggestions = """
## 改进建议

### 基于本次优化结果的建议：

1. **参数调优**
   - 如果收敛速度较慢，可考虑增加种群大小或调整变异/交叉概率
   - 如果解的多样性不足，可增加变异概率或使用多种群策略

2. **约束优化**
   - 检查约束违反情况，考虑调整约束权重
   - 对于难以满足的约束，可考虑使用软约束机制

3. **目标权重调整**
   - 根据实际需求调整各目标的权重
   - 可考虑使用交互式优化方法让用户参与决策

4. **数据质量**
   - 确保输入数据的准确性和完整性
   - 考虑增加更多历史案例以改善知识引导效果

### 系统性能优化：

1. **算法改进**
   - 可考虑使用NSGA-III处理高维目标问题
   - 实现并行化以提高计算效率

2. **数据处理**
   - 优化数据库查询性能
   - 实现数据缓存机制

## 文件输出说明

本次优化生成的文件包括：
- `{task_id}_optimization_report.txt`: 详细优化结果
- `{task_id}_results.json`: 机器可读的结果数据
- `{task_id}_objectives_boxplot.png`: 目标函数分布图
- `{task_id}_convergence_comparison.png`: 收敛曲线图
- `{task_id}_comprehensive_report.md`: 本综合报告

---
*报告生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}*
"""
        return suggestions

    def _log_performance_summary(self):
        """记录性能摘要"""
        try:
            total_time = self.performance_stats['end_time'] - self.performance_stats['start_time']
            
            logger.info("=" * 60)
            logger.info("性能统计摘要")
            logger.info("=" * 60)
            logger.info(f"总执行时间: {total_time:.2f} 秒")
            logger.info(f"数据加载: {self.performance_stats['data_loading_time']:.2f} 秒 ({self.performance_stats['data_loading_time']/total_time*100:.1f}%)")
            logger.info(f"优化执行: {self.performance_stats['optimization_time']:.2f} 秒 ({self.performance_stats['optimization_time']/total_time*100:.1f}%)")
            logger.info(f"后处理: {self.performance_stats['post_processing_time']:.2f} 秒 ({self.performance_stats['post_processing_time']/total_time*100:.1f}%)")
            
            if self.performance_stats['errors']:
                logger.warning(f"遇到错误数量: {len(self.performance_stats['errors'])}")
            
            logger.info("=" * 60)
            
        except Exception as e:
            logger.warning(f"性能摘要记录失败：{str(e)}")

    def _cleanup_resources(self):
        """清理资源"""
        try:
            if self.neo4j_handler:
                self.neo4j_handler.close()
                self.neo4j_handler = None
            
            logger.info("资源清理完成")
            
        except Exception as e:
            logger.warning(f"资源清理失败：{str(e)}")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='改进的海上通信参数多目标优化系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 标准优化
  python main_improved.py --task-id rw002
  
  # 对比优化方法
  python main_improved.py --task-id rw002 --compare
  
  # 自定义参数
  python main_improved.py --task-id rw002 --generations 500 --population 200
  
  # 指定数据库
  python main_improved.py --task-id rw002 --db-uri bolt://localhost:7687
        """
    )
    
    # 必需参数
    parser.add_argument(
        '--task-id', 
        type=str, 
        required=True,
        help='要优化的任务ID'
    )
    
    # 数据库参数
    parser.add_argument(
        '--db-uri', 
        type=str, 
        default="bolt://localhost:7699",
        help='Neo4j数据库URI (默认: bolt://localhost:7699)'
    )
    parser.add_argument(
        '--db-user', 
        type=str, 
        default="neo4j",
        help='Neo4j用户名 (默认: neo4j)'
    )
    parser.add_argument(
        '--db-password', 
        type=str, 
        default="12345678",
        help='Neo4j密码 (默认: 12345678)'
    )
    
    # 优化参数
    parser.add_argument(
        '--generations', 
        type=int, 
        help='NSGA-II迭代代数 (默认使用配置文件值)'
    )
    parser.add_argument(
        '--population', 
        type=int, 
        help='种群大小 (默认使用配置文件值)'
    )
    
    # 输出参数
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default="results",
        help='结果输出目录 (默认: results)'
    )
    
    # 功能选项
    parser.add_argument(
        '--no-save', 
        action='store_true',
        help='不保存结果到Neo4j数据库'
    )
    parser.add_argument(
        '--compare', 
        action='store_true',
        help='比较知识引导与传统NSGA-II方法'
    )
    
    # 调试选项
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='启用详细日志输出'
    )
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='启用调试模式'
    )
    
    return parser.parse_args()


def setup_environment(args):
    """设置运行环境"""
    try:
        # 设置日志级别
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
        elif args.verbose:
            logging.getLogger().setLevel(logging.INFO)
        else:
            logging.getLogger().setLevel(logging.WARNING)
        
        # 设置数值警告
        import warnings
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        
        # 设置numpy打印选项
        import numpy as np
        np.set_printoptions(precision=4, suppress=True)
        
        return True
        
    except Exception as e:
        print(f"环境设置失败：{str(e)}")
        return False



def main():
    """主函数"""
    try:
        # 解析命令行参数
        args = parse_arguments()
        
        # 设置环境
        if not setup_environment(args):
            sys.exit(1)
        
        # 打印启动信息
        print("=" * 80)
        print("改进的海上通信参数多目标优化系统")
        print("基于NSGA-II算法，集成知识引导和数值稳定性改进")
        print("=" * 80)
        print(f"任务ID: {args.task_id}")
        print(f"数据库: {args.db_uri}")
        print(f"输出目录: {args.output_dir}")
        if args.compare:
            print("模式: 方法对比")
        else:
            print("模式: 标准优化")
        print("=" * 80)
        
        # 创建并运行优化器
        runner = ImprovedOptimizationRunner(args)
        success = runner.run()
        
        # 退出代码
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n用户中断程序执行")
        sys.exit(130)
    except Exception as e:
        print(f"\n程序执行失败：{str(e)}")
        if args.debug if 'args' in locals() else False:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
   