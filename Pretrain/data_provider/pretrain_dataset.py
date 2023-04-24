import torch
from torch_geometric.data import Data, Dataset
import torch_geometric
from utils.GraphAug import drop_nodes, permute_edges, subgraph, mask_nodes
from copy import deepcopy
import numpy as np
import os
import random
from transformers import BertTokenizer


class GINPretrainDataset(Dataset):
    def __init__(self, root, text_max_len):
        super(GINPretrainDataset, self).__init__(root)
        self.root = root
        self.text_max_len = text_max_len
        self.graph_name_list = os.listdir(root+'graph/')
        self.graph_name_list.sort()
        self.text_name_list = os.listdir(root+'text/')
        self.text_name_list.sort()
        self.smiles_name_list = os.listdir(root+'smiles/')
        self.smiles_name_list.sort()
        self.tokenizer = BertTokenizer.from_pretrained('bert_pretrained/')

    def __len__(self):
        return len(self.graph_name_list)

    def __getitem__(self, index):
        graph_name, text_name, smiles_name = self.graph_name_list[index], self.text_name_list[index], self.smiles_name_list[index]
        # print(graph_name)
        # print(text_name)

        # load and process graph
        graph_path = os.path.join(self.root, 'graph', graph_name)
        data_graph = torch.load(graph_path)

        # load and process text
        text_path = os.path.join(self.root, 'text', text_name)
        text_list = []
        count = 0
        for line in open(text_path, 'r', encoding='utf-8'):
            count += 1
            text_list.append(line)
            if count > 500:
                break
        # print(text_list)
        if len(text_list) < 2:
            two_text_list = [text_list[0], text_list[0]]
        else:
            two_text_list = random.sample(text_list, 2)
        text_list.clear()

        text1, mask1 = self.tokenizer_func(two_text_list[0])
        text2, mask2 = self.tokenizer_func(two_text_list[1])

        # load and process smiles
        smiles_path = os.path.join(self.root, 'smiles', smiles_name)
        with open(smiles_path, 'r') as f:
            smiles = f.readline().rstrip()
        smiles_tokens, smiles_mask = self.tokenizer_func(smiles)

        return data_graph, smiles_tokens.squeeze(0), smiles_mask.squeeze(0), text1.squeeze(0), mask1.squeeze(0), text2.squeeze(0), mask2.squeeze(0)

    def augment(self, data, graph_aug):
        # node_num = data.edge_index.max()
        # sl = torch.tensor([[n, n] for n in range(node_num)]).t()
        # data.edge_index = torch.cat((data.edge_index, sl), dim=1)

        if graph_aug == 'dnodes':
            data_aug = drop_nodes(deepcopy(data))
        elif graph_aug == 'pedges':
            data_aug = permute_edges(deepcopy(data))
        elif graph_aug == 'subgraph':
            data_aug = subgraph(deepcopy(data))
        elif graph_aug == 'mask_nodes':
            data_aug = mask_nodes(deepcopy(data))
        elif graph_aug == 'random2':  # choose one from two augmentations
            n = np.random.randint(2)
            if n == 0:
                data_aug = drop_nodes(deepcopy(data))
            elif n == 1:
                data_aug = subgraph(deepcopy(data))
            else:
                print('sample error')
                assert False
        elif graph_aug == 'random3':  # choose one from three augmentations
            n = np.random.randint(3)
            if n == 0:
                data_aug = drop_nodes(deepcopy(data))
            elif n == 1:
                data_aug = permute_edges(deepcopy(data))
            elif n == 2:
                data_aug = subgraph(deepcopy(data))
            else:
                print('sample error')
                assert False
        elif graph_aug == 'random4':  # choose one from four augmentations
            n = np.random.randint(4)
            if n == 0:
                data_aug = drop_nodes(deepcopy(data))
            elif n == 1:
                data_aug = permute_edges(deepcopy(data))
            elif n == 2:
                data_aug = subgraph(deepcopy(data))
            elif n == 3:
                data_aug = mask_nodes(deepcopy(data))
            else:
                print('sample error')
                assert False
        else:
            data_aug = deepcopy(data)
            data_aug.x = torch.ones((data.edge_index.max()+1, 1))

        return data_aug

    def tokenizer_func(self, text):
        sentence_token = self.tokenizer(text=text,
                                        truncation=True,
                                        padding='max_length',
                                        add_special_tokens=False,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True)
        input_ids = sentence_token['input_ids']  # [176,398,1007,0,0,0]
        attention_mask = sentence_token['attention_mask']  # [1,1,1,0,0,0]
        return input_ids, attention_mask


if __name__ == '__main__':
    # mydataset = GraphTextDataset()
    # train_loader = torch_geometric.loader.DataLoader(
    #     mydataset,
    #     batch_size=16,
    #     shuffle=True,
    #     num_workers=4
    # )
    # for i, (graph, text1, mask1, text2, mask2) in enumerate(train_loader):
    #     print(aug1.edge_index.shape)
    #     print(aug1.x.shape)
    #     print(aug1.ptr.size(0))
    #     print(aug2.edge_index.dtype)
    #     print(aug2.x.dtype)
    #     print(text1.shape)
    #     print(mask1.shape)
    #     print(text2.shape)
    #     print(mask2.shape)
    # mydataset = GraphormerPretrainDataset(root='data/', text_max_len=128, graph_aug1='dnodes', graph_aug2='subgraph')
    # from functools import partial
    # from data_provider.collator import collator_text
    # train_loader = torch.utils.data.DataLoader(
    #         mydataset,
    #         batch_size=8,
    #         num_workers=4,
    #         collate_fn=partial(collator_text,
    #                            max_node=128,
    #                            multi_hop_max_dist=5,
    #                            spatial_pos_max=1024),
    #     )
    # graph, text1, mask1, text2, mask2 = mydataset[0]
    mydataset = GINPretrainDataset(root='data/', text_max_len=128, graph_aug1='dnodes', graph_aug2='subgraph')
    train_loader = torch_geometric.loader.DataLoader(
            mydataset,
            batch_size=32,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
            # persistent_workers = True
        )

    for i, (graph, text1, mask1, text2, mask2) in enumerate(train_loader):
        print(graph)
        # print(graph.x.shape)
        # print(graph)
        # print(graph.x.dtype)
        # print(text1.shape)
        # print(mask1.shape)
        # print(text2.shape)
        # print(mask2.shape)