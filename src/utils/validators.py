import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class DataValidator:
    """改进的数据验证器，增强容错性和详细日志"""
    
    @staticmethod
    def validate_task_data(data: Dict) -> bool:
        """验证任务数据完整性 - 改进版"""
        try:
            if not isinstance(data, dict):
                logger.error("任务数据不是字典类型")
                return False
            
            # 检查基本结构
            task_info = data.get('task_info', {})

            # 打印所有获得的task_info信息 - 用于调试
            logger.info("获取的task_info信息:")
            for key, value in task_info.items():
                # 对于值，如果长度超过100则截断显示
                if isinstance(value, str) and len(value) > 100:
                    logger.info(f"  {key}: {value[:100]}...")
                else:
                    logger.info(f"  {key}: {value}")
            
            # 只检查最关键的字段
            if 'task_id' not in task_info or not task_info['task_id']:
                logger.error("缺少必需的任务ID")
                return False
            
            # 检查可选字段
            optional_fields = ['task_name', 'task_area', 'task_time', 'force_composition']
            missing_optional = []
            for field in optional_fields:
                if field not in task_info or not task_info[field]:
                    missing_optional.append(field)
            
            if missing_optional:
                logger.warning(f"缺少可选的任务信息字段: {missing_optional}")
            
            # 检查通信链路
            comm_links = data.get('communication_links', [])
            if not comm_links:
                logger.warning("没有通信链路，将在后续创建默认链路")
            else:
                logger.info(f"找到 {len(comm_links)} 个通信链路")
            
            # 检查节点信息
            nodes = data.get('nodes', {})
            if not nodes:
                logger.warning("没有节点信息")
            else:
                logger.info(f"节点信息: {list(nodes.keys())}")
            
            logger.info("任务数据验证通过")
            return True
            
        except Exception as e:
            logger.error(f"任务数据验证异常: {str(e)}")
            return False
    
    @staticmethod
    def validate_environment_data(data: Dict) -> bool:
        """验证环境数据完整性 - 改进版"""
        try:
            if not isinstance(data, dict):
                logger.error("环境数据不是字典类型")
                return False
            
            if not data:
                logger.error("环境数据为空")
                return False
            
            # 检查关键环境参数，但允许缺失
            required_fields = ['海况等级', '电磁干扰强度']
            optional_fields = ['地理特征', '背景噪声', '多径效应', '温度', '盐度', '深度']
            
            missing_required = []
            for field in required_fields:
                if field not in data:
                    missing_required.append(field)
            
            if missing_required:
                logger.warning(f"缺少环境字段: {missing_required}，使用默认值")
                # 不直接返回False，而是警告并继续
            
            # 检查可选字段
            missing_optional = []
            for field in optional_fields:
                if field not in data:
                    missing_optional.append(field)
            
            if missing_optional:
                logger.warning(f"缺少可选的环境字段: {missing_optional}")
            
            # 验证数值合理性
            try:
                sea_state = float(str(data.get('海况等级', 3)).replace('级', ''))
                if not (0 <= sea_state <= 9):
                    logger.warning(f"海况等级超出合理范围: {sea_state}")
                
                emi_intensity = float(str(data.get('电磁干扰强度', 0.5)))
                if not (0 <= emi_intensity <= 1):
                    logger.warning(f"电磁干扰强度超出合理范围: {emi_intensity}")
                    
            except (ValueError, TypeError) as e:
                logger.warning(f"环境数据数值转换失败: {str(e)}")
            
            logger.info("环境数据验证通过")
            return True
            
        except Exception as e:
            logger.error(f"环境数据验证异常: {str(e)}")
            return False
    
    @staticmethod
    def validate_constraint_data(data: Dict) -> bool:
        """验证约束条件数据完整性 - 改进版"""
        try:
            if not isinstance(data, dict):
                logger.error("约束数据不是字典类型")
                return False
            
            if not data:
                logger.error("约束数据为空")
                return False
            
            # 检查关键约束参数，但允许缺失
            required_fields = ['最小可靠性要求', '最大时延要求']
            optional_fields = [
                '频谱最小频率', '频谱最大频率', '最小信噪比',
                '带宽限制', '发射功率限制'
            ]
            
            missing_required = []
            for field in required_fields:
                if field not in data:
                    missing_required.append(field)
            
            if missing_required:
                logger.warning(f"缺少约束字段: {missing_required}，使用默认值")
                # 不直接返回False，而是警告并继续
            
            # 检查可选字段
            missing_optional = []
            for field in optional_fields:
                if field not in data:
                    missing_optional.append(field)
            
            if missing_optional:
                logger.warning(f"缺少可选的约束字段: {missing_optional}")
            
            # 验证数值合理性
            try:
                reliability = float(str(data.get('最小可靠性要求', 0.95)))
                if not (0 <= reliability <= 1):
                    logger.warning(f"可靠性要求超出合理范围: {reliability}")
                
                delay = float(str(data.get('最大时延要求', 100)).replace('ms', '').replace('毫秒', ''))
                if delay <= 0:
                    logger.warning(f"时延要求不合理: {delay}")
                    
            except (ValueError, TypeError) as e:
                logger.warning(f"约束数据数值转换失败: {str(e)}")
            
            logger.info("约束数据验证通过")
            return True
            
        except Exception as e:
            logger.error(f"约束数据验证异常: {str(e)}")
            return False
    
    @staticmethod
    def validate_solution(solution, config: Dict[str, Any]) -> bool:
        """验证解的有效性 - 改进版"""
        try:
            if solution is None:
                logger.error("解为空")
                return False
            
            if not isinstance(solution, (list, tuple, np.ndarray)):
                logger.error("解的格式不正确")
                return False
            
            solution = np.array(solution)
            
            # 基本形状检查
            if len(solution) == 0:
                logger.error("解向量为空")
                return False
            
            if len(solution) % 5 != 0:
                logger.warning(f"解向量长度 {len(solution)} 不是5的倍数")
                # 不直接返回False，继续验证
            
            # 检查数值有效性
            if not np.all(np.isfinite(solution)):
                logger.error("解包含无效数值 (NaN或Inf)")
                return False
            
            # 如果有配置信息，检查参数范围
            if config:
                n_links = len(solution) // 5
                
                # 检查频率
                frequencies = solution[:n_links] if len(solution) >= n_links else []
                if len(frequencies) > 0 and 'freq_min' in config and 'freq_max' in config:
                    freq_valid = np.all(
                        (frequencies >= config['freq_min']) &
                        (frequencies <= config['freq_max'])
                    )
                    if not freq_valid:
                        logger.warning("部分频率值超出约束范围")
                
                # 检查功率（如果有足够的解向量长度）
                if len(solution) >= 3 * n_links:
                    powers = solution[2*n_links:3*n_links]
                    if 'power_min' in config and 'power_max' in config:
                        power_valid = np.all(
                            (powers >= config['power_min']) &
                            (powers <= config['power_max'])
                        )
                        if not power_valid:
                            logger.warning("部分功率值超出约束范围")
                
                # 检查带宽（如果有足够的解向量长度）
                if len(solution) >= 2 * n_links:
                    bandwidths = solution[n_links:2*n_links]
                    if 'bandwidth_min' in config and 'bandwidth_max' in config:
                        bandwidth_valid = np.all(
                            (bandwidths >= config['bandwidth_min']) &
                            (bandwidths <= config['bandwidth_max'])
                        )
                        if not bandwidth_valid:
                            logger.warning("部分带宽值超出约束范围")
            
            logger.info("解验证通过")
            return True
            
        except Exception as e:
            logger.error(f"解验证异常: {str(e)}")
            return False
