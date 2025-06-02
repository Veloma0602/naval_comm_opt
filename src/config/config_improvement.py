"""
改进的优化配置参数
应该放在 config/improved_parameters.py

这个文件扩展了现有的 parameters.py，添加了更多的验证和安全性功能
"""

from .parameters import OptimizationConfig

class ImprovedOptimizationConfig(OptimizationConfig):
    """
    改进的优化配置类，继承自现有的 OptimizationConfig
    添加了验证、安全性和调试功能
    """
    
    def __init__(self):
        """初始化改进的优化配置参数"""
        super().__init__()
        
        # 添加新的配置参数
        self.constraint_penalty = 1000.0      # 约束违反惩罚系数
        self.constraint_tolerance = 1e-6      # 约束容忍度
        self.numerical_epsilon = 1e-12        # 数值精度
        self.max_objective_value = 1e6        # 最大目标函数值
        self.convergence_tolerance = 1e-8     # 收敛容忍度
        
        # 调试和安全选项
        self.debug_mode = False
        self.verbose_logging = True
        self.safe_mode = True                 # 安全模式，启用额外检查
        
        # 调整默认参数为更稳定的值
        self.population_size = 50             # 减小默认种群大小
        self.n_generations = 100              # 减少默认迭代次数
        self.mutation_prob = 0.1              # 降低变异概率
    
    def validate_configuration(self):
        """
        验证配置参数的有效性
        返回: (is_valid: bool, errors: list)
        """
        errors = []
        
        # 检查基本参数
        if self.population_size <= 0:
            errors.append("种群大小必须大于0")
        elif self.population_size > 500:
            errors.append("种群大小不应超过500（性能考虑）")
        
        if self.n_generations <= 0:
            errors.append("迭代代数必须大于0")
        elif self.n_generations > 1000:
            errors.append("迭代代数不应超过1000（性能考虑）")
        
        if not (0 <= self.mutation_prob <= 1):
            errors.append("变异概率必须在0-1之间")
        
        if not (0 <= self.crossover_prob <= 1):
            errors.append("交叉概率必须在0-1之间")
        
        # 检查频率约束
        if self.freq_min >= self.freq_max:
            errors.append("最小频率必须小于最大频率")
        
        if self.freq_min <= 0:
            errors.append("最小频率必须大于0")
        
        if self.freq_max > 20e9:
            errors.append("最大频率不应超过20GHz（实际限制）")
        
        # 检查功率约束
        if self.power_min >= self.power_max:
            errors.append("最小功率必须小于最大功率")
        
        if self.power_min <= 0:
            errors.append("最小功率必须大于0")
        
        if self.power_max > 1000:
            errors.append("最大功率不应超过1000W（安全考虑）")
        
        # 检查带宽约束
        if self.bandwidth_min >= self.bandwidth_max:
            errors.append("最小带宽必须小于最大带宽")
        
        if self.bandwidth_min <= 0:
            errors.append("最小带宽必须大于0")
        
        # 检查权重
        total_weight = (self.reliability_weight + self.spectral_weight + 
                       self.energy_weight + self.interference_weight + 
                       self.adaptability_weight)
        
        if abs(total_weight - 1.0) > 1e-6:
            errors.append(f"目标权重总和应为1.0，当前为{total_weight:.6f}")
        
        # 检查新增的参数
        if self.constraint_penalty <= 0:
            errors.append("约束惩罚系数必须大于0")
        
        if self.constraint_tolerance <= 0:
            errors.append("约束容忍度必须大于0")
        
        return len(errors) == 0, errors
    
    def get_safe_config(self):
        """
        获取安全的配置（确保所有参数都在合理范围内）
        """
        safe_config = ImprovedOptimizationConfig()
        
        # 确保参数在安全范围内
        safe_config.population_size = max(10, min(100, self.population_size))
        safe_config.n_generations = max(10, min(300, self.n_generations))
        safe_config.mutation_prob = max(0.01, min(0.5, self.mutation_prob))
        safe_config.crossover_prob = max(0.5, min(1.0, self.crossover_prob))
        
        # 确保频率范围合理
        safe_config.freq_min = max(50e6, min(1e9, self.freq_min))
        safe_config.freq_max = max(safe_config.freq_min * 2, min(15e9, self.freq_max))
        
        # 确保功率范围合理
        safe_config.power_min = max(1.0, min(10.0, self.power_min))
        safe_config.power_max = max(safe_config.power_min * 2, min(100.0, self.power_max))
        
        # 确保带宽范围合理
        safe_config.bandwidth_min = max(1e6, min(10e6, self.bandwidth_min))
        safe_config.bandwidth_max = max(safe_config.bandwidth_min * 2, min(100e6, self.bandwidth_max))
        
        # 复制其他参数
        safe_config.snr_min = self.snr_min
        safe_config.delay_max = self.delay_max
        safe_config.reliability_min = self.reliability_min
        safe_config.min_freq_separation = self.min_freq_separation
        
        # 复制权重（应该已经验证过）
        safe_config.reliability_weight = self.reliability_weight
        safe_config.spectral_weight = self.spectral_weight
        safe_config.energy_weight = self.energy_weight
        safe_config.interference_weight = self.interference_weight
        safe_config.adaptability_weight = self.adaptability_weight
        
        return safe_config
    
    def auto_adjust_for_quick_test(self):
        """
        自动调整参数以进行快速测试
        """
        self.population_size = 20
        self.n_generations = 30
        self.mutation_prob = 0.15
        self.crossover_prob = 0.85
        
        print("配置已调整为快速测试模式:")
        print(f"  种群大小: {self.population_size}")
        print(f"  迭代代数: {self.n_generations}")
        print(f"  变异概率: {self.mutation_prob}")
        print(f"  交叉概率: {self.crossover_prob}")
    
    def auto_adjust_for_production(self):
        """
        自动调整参数以进行生产环境优化
        """
        self.population_size = 100
        self.n_generations = 200
        self.mutation_prob = 0.05
        self.crossover_prob = 0.95
        
        print("配置已调整为生产环境模式:")
        print(f"  种群大小: {self.population_size}")
        print(f"  迭代代数: {self.n_generations}")
        print(f"  变异概率: {self.mutation_prob}")
        print(f"  交叉概率: {self.crossover_prob}")
    
    def to_dict(self):
        """将配置转换为字典，包含新增的参数"""
        config_dict = super().to_dict()
        
        # 添加新的参数
        config_dict.update({
            "numerical_params": {
                "constraint_penalty": self.constraint_penalty,
                "constraint_tolerance": self.constraint_tolerance,
                "numerical_epsilon": self.numerical_epsilon,
                "max_objective_value": self.max_objective_value,
                "convergence_tolerance": self.convergence_tolerance
            },
            "debug_params": {
                "debug_mode": self.debug_mode,
                "verbose_logging": self.verbose_logging,
                "safe_mode": self.safe_mode
            }
        })
        
        return config_dict
    
    def load_from_dict(self, config_dict: dict):
        """从字典加载配置，包含新增的参数"""
        # 先调用父类方法
        success = super().load_from_dict(config_dict)
        
        if not success:
            return False
        
        try:
            # 加载新增的参数
            if "numerical_params" in config_dict:
                num_params = config_dict["numerical_params"]
                self.constraint_penalty = num_params.get("constraint_penalty", self.constraint_penalty)
                self.constraint_tolerance = num_params.get("constraint_tolerance", self.constraint_tolerance)
                self.numerical_epsilon = num_params.get("numerical_epsilon", self.numerical_epsilon)
                self.max_objective_value = num_params.get("max_objective_value", self.max_objective_value)
                self.convergence_tolerance = num_params.get("convergence_tolerance", self.convergence_tolerance)
            
            if "debug_params" in config_dict:
                debug_params = config_dict["debug_params"]
                self.debug_mode = debug_params.get("debug_mode", self.debug_mode)
                self.verbose_logging = debug_params.get("verbose_logging", self.verbose_logging)
                self.safe_mode = debug_params.get("safe_mode", self.safe_mode)
            
            return True
        except Exception as e:
            print(f"加载改进配置时出错: {str(e)}")
            return False
    
    def get_test_configs(self):
        """
        获取不同场景的测试配置
        返回配置字典
        """
        return {
            "quick_test": {
                "population_size": 20,
                "n_generations": 30,
                "mutation_prob": 0.15,
                "crossover_prob": 0.85,
                "description": "快速测试配置"
            },
            "standard_test": {
                "population_size": 50,
                "n_generations": 100,
                "mutation_prob": 0.1,
                "crossover_prob": 0.9,
                "description": "标准测试配置"
            },
            "thorough_test": {
                "population_size": 100,
                "n_generations": 200,
                "mutation_prob": 0.05,
                "crossover_prob": 0.95,
                "description": "深度测试配置"
            }
        }
    
    def apply_test_config(self, config_name: str):
        """
        应用指定的测试配置
        """
        test_configs = self.get_test_configs()
        
        if config_name not in test_configs:
            available = ", ".join(test_configs.keys())
            raise ValueError(f"未知的配置名称: {config_name}. 可用配置: {available}")
        
        config = test_configs[config_name]
        
        self.population_size = config["population_size"]
        self.n_generations = config["n_generations"]
        self.mutation_prob = config["mutation_prob"]
        self.crossover_prob = config["crossover_prob"]
        
        print(f"已应用 {config['description']} ({config_name}):")
        print(f"  种群大小: {self.population_size}")
        print(f"  迭代代数: {self.n_generations}")
        print(f"  变异概率: {self.mutation_prob}")
        print(f"  交叉概率: {self.crossover_prob}")
    
    def __str__(self):
        """增强的字符串表示"""
        base_str = super().__str__()
        
        additional_info = f"""
额外配置:
  约束惩罚系数: {self.constraint_penalty}
  约束容忍度: {self.constraint_tolerance}
  数值精度: {self.numerical_epsilon}
  调试模式: {self.debug_mode}
  安全模式: {self.safe_mode}"""
        
        return base_str + additional_info

# 便捷函数
def create_quick_test_config():
    """创建快速测试配置"""
    config = ImprovedOptimizationConfig()
    config.auto_adjust_for_quick_test()
    return config

def create_production_config():
    """创建生产环境配置"""
    config = ImprovedOptimizationConfig()
    config.auto_adjust_for_production()
    return config

def validate_and_fix_config(config):
    """
    验证并修复配置
    返回: (fixed_config, is_valid, errors)
    """
    is_valid, errors = config.validate_configuration()
    
    if is_valid:
        return config, True, []
    else:
        # 尝试获取安全配置
        safe_config = config.get_safe_config()
        is_safe_valid, safe_errors = safe_config.validate_configuration()
        
        if is_safe_valid:
            print("原配置有问题，已自动修复为安全配置")
            return safe_config, True, errors
        else:
            print("无法修复配置，使用默认配置")
            default_config = ImprovedOptimizationConfig()
            return default_config, False, errors + safe_errors