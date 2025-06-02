from neo4j import GraphDatabase
import json
import time
import sys

class Neo4jImporter:
    def __init__(self, uri="bolt://localhost:7699", username="neo4j", password="12345678"):
        """初始化Neo4j连接"""
        try:
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
            print(f"成功连接到Neo4j数据库: {uri}")
        except Exception as e:
            print(f"连接Neo4j失败: {str(e)}")
            sys.exit(1)
    
    def close(self):
        """关闭数据库连接"""
        if hasattr(self, 'driver'):
            self.driver.close()
            print("Neo4j连接已关闭")
    
    def clear_database(self):
        """清空数据库中的所有数据"""
        print("正在清空数据库...")
        try:
            with self.driver.session() as session:
                # 删除所有关系和节点
                result = session.run("MATCH (n) DETACH DELETE n")
                # 获取删除的节点和关系数量（可选）
                counts = session.run("MATCH (n) RETURN count(n) as nodes")
                node_count = counts.single()["nodes"]
                print(f"成功清空数据库，当前节点数量：{node_count}")
                return True
        except Exception as e:
            print(f"清空数据库出错: {str(e)}")
            return False
    
    def import_json(self, json_file):
        """从JSON文件导入数据"""
        print(f"开始导入数据: {json_file}")
        start_time = time.time()
        
        try:
            # 加载JSON文件
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取节点和关系
            nodes = data.get('nodes', [])
            relationships = data.get('relationships', [])
            
            print(f"读取到 {len(nodes)} 个节点和 {len(relationships)} 个关系")
            
            # 分批处理节点和关系以提高性能
            batch_size = 1000
            
            # 导入节点
            for i in range(0, len(nodes), batch_size):
                batch = nodes[i:i+batch_size]
                self._import_nodes_batch(batch)
                print(f"已导入节点: {min(i+batch_size, len(nodes))}/{len(nodes)}")
            
            # 导入关系
            for i in range(0, len(relationships), batch_size):
                batch = relationships[i:i+batch_size]
                self._import_relationships_batch(batch)
                print(f"已导入关系: {min(i+batch_size, len(relationships))}/{len(relationships)}")
            
            end_time = time.time()
            print(f"导入完成，耗时: {end_time - start_time:.2f}秒")
            
            # 验证导入
            self._verify_import()
            
            return True
        
        except Exception as e:
            print(f"导入数据出错: {str(e)}")
            return False
    
    def _import_nodes_batch(self, nodes_batch):
        """批量导入节点"""
        with self.driver.session() as session:
            for node in nodes_batch:
                # 构建Cypher参数
                node_id = node.get('identity')
                labels = node.get('labels', [])
                properties = node.get('properties', {})
                
                # 构建标签字符串
                label_str = ':'.join(labels) if labels else 'Node'
                
                # 构建查询
                query = f"""
                CREATE (n:{label_str} {{node_id: $node_id}})
                SET n += $properties
                """
                
                # 执行查询
                session.run(query, node_id=node_id, properties=properties)
    
    def _import_relationships_batch(self, rels_batch):
        """批量导入关系"""
        with self.driver.session() as session:
            for rel in rels_batch:
                # 构建Cypher参数
                rel_id = rel.get('identity')
                rel_type = rel.get('type')
                start_id = rel.get('start')
                end_id = rel.get('end')
                properties = rel.get('properties', {})
                
                # 构建查询
                query = f"""
                MATCH (a {{node_id: $start_id}}), (b {{node_id: $end_id}})
                CREATE (a)-[r:{rel_type} {{rel_id: $rel_id}}]->(b)
                SET r += $properties
                """
                
                # 执行查询
                session.run(query, rel_id=rel_id, start_id=start_id, 
                           end_id=end_id, properties=properties)
    
    def _verify_import(self):
        """验证导入结果"""
        with self.driver.session() as session:
            # 获取节点和关系数量
            node_count = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
            
            # 获取标签分布
            label_result = session.run("""
                CALL db.labels() YIELD label
                MATCH (n:`${label}`)
                RETURN label, count(n) as count
                ORDER BY count DESC
            """)
            
            labels = {record["label"]: record["count"] for record in label_result}
            
            # 输出验证结果
            print("\n=== 导入验证 ===")
            print(f"总节点数: {node_count}")
            print(f"总关系数: {rel_count}")
            print("标签分布:")
            for label, count in labels.items():
                print(f"  {label}: {count}")

def main():
    # 配置参数（根据实际情况修改）
    neo4j_uri = "bolt://localhost:7699"
    neo4j_user = "neo4j"
    neo4j_password = "12345678"
    json_file = "naval_knowledge_graph_improved.json"  # JSON文件路径
    
    # 创建导入器
    importer = Neo4jImporter(neo4j_uri, neo4j_user, neo4j_password)
    
    try:
        # 清空数据库
        if importer.clear_database():
            # 导入JSON数据
            importer.import_json(json_file)
    finally:
        # 关闭连接
        importer.close()

if __name__ == "__main__":
    main()