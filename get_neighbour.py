import jsonlines
import json

from torch_geometric.datasets import Planetoid
from torch_geometric.utils import k_hop_subgraph
import torch

raw_data_path = 'C:/Users/86130/PycharmProjects/pythonProject28/GraphProcess/Cora/full_graph.jsonl'

processed_raw_path = 'C:/Users/86130/PycharmProjects/pythonProject28/GraphProcess/Cora/neighbours.json'

cora_path = 'C:/Users/86130/PycharmProjects/pythonProject28/GraphProcess/Cora/cora_raw.json'
# dataset from LLaGA
neighbours = []
with open(raw_data_path, 'r', encoding="utf8") as f:
    for i, line in enumerate(f):
        data = json.loads(line)

        if 'graph' in data:
            top = data['graph'][1:11]
            neighbour = [value for value in top if value != -500]
            print(neighbour)

            elem = {
                'id': data['id'],
                'neighbour': neighbour
            }

            print(elem)
            neighbours.append(elem)

with open(processed_raw_path, 'w', encoding="utf8") as f:
    json.dump(neighbours, f, indent=4)

    print('done')

# raw dataset of cora

# 对节点1-10进行操作
nodes_to_search = range(0, 2707)
node_sequences = {}

# 加载Cora数据集
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

cora_neighbors = []
for node in nodes_to_search:
    # 找到1-hop邻居节点并打印
    neighbors = torch.where(data.edge_index[0] == node)[0]
    hop1_nodes = [data.edge_index[1, neighbor].item() for neighbor in neighbors]

    elem = {
        'id': data.edge_index[0][node].item(),
        'hop1': hop1_nodes,
    }
    cora_neighbors.append(elem)
    print(elem)

with open(cora_path, 'w', encoding="utf8") as f:
    json.dump(neighbours, f, indent=4)

    print('done')







