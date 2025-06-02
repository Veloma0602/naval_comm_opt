#!/usr/bin/env python3
# neo4j_test_script.py
"""
Neo4j 查询测试脚本

该脚本用于测试 Neo4jHandler 中的 test_query 方法，验证:
1. 数据库连接是否正常
2. 任务数据检索是否正确
3. 节点和关系分析功能是否正常工作
"""

from data.neo4j_handler import Neo4jHandler
import logging
import argparse

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('neo4j_test')

def main():
    parser = argparse.ArgumentParser(description='Neo4j 查询测试工具')
    parser.add_argument('task_id', type=str, help='要测试的任务ID')
    parser.add_argument('--uri', type=str, default='bolt://localhost:7699', 
                       help='Neo4j服务器URI (默认: bolt://localhost:7699)')
    parser.add_argument('--user', type=str, default='neo4j', help='Neo4j用户名 (默认: neo4j)')
    parser.add_argument('--password', type=str, default='12345678', help='Neo4j密码')
    args = parser.parse_args()

    # 创建Neo4j处理器
    logger.info(f"正在连接Neo4j: {args.uri}")
    neo4j_handler = Neo4jHandler(uri=args.uri, user=args.user, password=args.password)
    
    try:
        logger.info(f"开始测试任务: {args.task_id}")
        
        # 运行测试查询
        logger.info("执行测试查询...")
        success = neo4j_handler.test_query(args.task_id)
        
        if success:
            logger.info("测试查询成功完成")
            
            # 获取详细任务数据
            logger.info("获取完整任务数据...")
            task_data = neo4j_handler.get_task_data_improved(args.task_id)
            
            if task_data:
                logger.info("任务数据结构:")
                print(f"任务ID: {task_data['task_info']['task_id']}")
                print(f"任务名称: {task_data['task_info']['task_name']}")
                print(f"任务区域: {task_data['task_info']['task_area']}")
                print(f"兵力组成: {task_data['task_info']['force_composition']}")
                
                # 打印节点信息
                print("\n节点信息:")
                nodes = task_data['nodes']
                if nodes['command_center']:
                    print(f"指挥所: {nodes['command_center'].get('名称', '未知')}")
                if nodes['command_ship']:
                    print(f"指挥舰船: {nodes['command_ship'].get('名称', '未知')}")
                print(f"作战单位: {len(nodes['combat_units'])}个")
                print(f"通信站: {len(nodes['comm_stations'])}个")
                print(f"通信设备: {len(nodes['communication_systems'])}个")
                
                # 打印通信链路
                print("\n通信链路:")
                for i, link in enumerate(task_data['communication_links'], 1):
                    print(f"链路 #{i}:")
                    print(f"  源节点ID: {link['source_id']}")
                    print(f"  目标节点ID: {link['target_id']}")
                    print(f"  通信类型: {link['comm_type']}")
                    print(f"  频段: {link['frequency_min']/1e6:.2f}-{link['frequency_max']/1e6:.2f} MHz")
                    print(f"  带宽: {link['bandwidth']/1e6:.2f} MHz")
                    print(f"  功率: {link['power']} W")
                    print(f"  调制方式: {link['modulation']}")
                print(f"总共发现 {len(task_data['communication_links'])} 条通信链路")
            else:
                logger.error("无法获取任务数据")
        else:
            logger.error("测试查询失败")
            
    except Exception as e:
        logger.error(f"测试过程中发生错误: {str(e)}")
    finally:
        # 关闭连接
        neo4j_handler.close()
        logger.info("Neo4j连接已关闭")

if __name__ == '__main__':
    main()