from flask import Flask, request, jsonify, render_template  # ← 注意这里也加上 render_template
from py2neo import Graph
import requests
import json
def gpt_4_call(text, api_key, url="https://api.chatanywhere.tech/v1/chat/completions"):
    payload = json.dumps({
        "model": "gpt-4o-mini",
        "temperature": 1.0,
        "messages": [
            {"role": "system", "content": "你是一个中文知识图谱助手，会根据结构化数据为用户生成简洁自然的描述。"},
            {"role": "user", "content": text}
        ]
    })
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    response = requests.post(url, headers=headers, data=payload)
    if response.status_code == 200:
        try:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except KeyError:
            return "返回格式异常"
    else:
        return f"请求失败：{response.status_code} - {response.text}"
app = Flask(__name__)

# 连接 Neo4j 数据库
graph = Graph("bolt://localhost:7687", auth=("neo4j", "dengzhijie250"))

@app.route('/')
def index():
    return render_template('index.html')  # ← 渲染 index.html 页面
@app.route('/search', methods=['GET'])
def search():
    keyword = request.args.get('q')
    limit = int(request.args.get('limit', 10))

    # 查询图谱
    query = f"""
    MATCH (center)
    WHERE center.name CONTAINS '{keyword}'
    WITH center LIMIT 1
    MATCH (center)-[r]-(neighbor)
    RETURN center.name AS center, type(r) AS relation, neighbor.name AS neighbor, labels(neighbor) AS labels
    LIMIT {limit}
    """
    results = graph.run(query).data()

    # 图谱可视化用节点 & 边
    nodes = []
    links = []
    node_ids = set()
    for record in results:
        center = record['center']
        neighbor = record['neighbor']
        rel = record['relation']

        # 中心节点
        center_id = hash(center)
        if center_id not in node_ids:
            nodes.append({"id": center_id, "label": center, "type": "中心实体"})
            node_ids.add(center_id)

        # 邻居节点
        neighbor_id = hash(neighbor)
        if neighbor_id not in node_ids:
            nodes.append({"id": neighbor_id, "label": neighbor, "type": record['labels'][0]})
            node_ids.add(neighbor_id)

        links.append({
            "source": center_id,
            "target": neighbor_id,
            "label": rel
        })

    # 构造给 GPT 的 prompt
    API_KEY = "sk-**"
    prompt = f"""
用户查询的是：{keyword}
以下是该实体在知识图谱中相关的信息：
{json.dumps(results, indent=2, ensure_ascii=False)}
请根据这些信息，用中文生成简洁自然的描述。
"""
    llm_output = gpt_4_call(prompt, api_key=API_KEY)

    return jsonify({
        "nodes": nodes,
        "links": links,
        "description": llm_output
    })

if __name__ == '__main__':
    app.run(debug=True)