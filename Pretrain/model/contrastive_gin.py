import torch
import torch.nn as nn
from model.gin_model import GNN
from model.bert import TextEncoder
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import optim


class GINSimclr(pl.LightningModule):
    def __init__(
            self,
            temperature,
            gin_hidden_dim,
            gin_num_layers,
            drop_ratio,
            graph_pooling,
            bert_hidden_dim,
            bert_pretrain,
            projection_dim,
            lr,
            weight_decay,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.temperature = temperature
        self.gin_hidden_dim = gin_hidden_dim
        self.gin_num_layers = gin_num_layers
        self.drop_ratio = drop_ratio
        self.graph_pooling = graph_pooling

        self.bert_hidden_dim = bert_hidden_dim
        self.bert_pretrain = bert_pretrain

        self.projection_dim = projection_dim

        self.lr = lr
        self.weight_decay = weight_decay

        self.graph_encoder = GNN(
            num_layer=self.gin_num_layers,
            emb_dim=self.gin_hidden_dim,
            gnn_type='gin',
            # virtual_node=True,
            # residual=False,
            drop_ratio=self.drop_ratio,
            JK='last',
            # graph_pooling=self.graph_pooling,
        )
        # print(self.graph_encoder.state_dict().keys())
        ckpt = torch.load('gin_pretrained/graphcl_80.pth')
        # print(ckpt.keys())
        missing_keys, unexpected_keys = self.graph_encoder.load_state_dict(ckpt, strict=False)
        print(missing_keys)
        print(unexpected_keys)

        # Text Encoder
        if self.bert_pretrain:
            self.text_encoder = TextEncoder(pretrained=False)
        else:
            self.text_encoder = TextEncoder(pretrained=True)
        
        # Smiles Encoder (same as text encoder)
        if self.bert_pretrain:
            self.smiles_encoder = TextEncoder(pretrained=False)
        else:
            self.smiles_encoder = TextEncoder(pretrained=True)
            
        if self.bert_pretrain:
            print("bert load kvplm")
            ckpt = torch.load('kvplm_pretrained/ckpt_KV_1.pt')
            if 'module.ptmodel.bert.embeddings.word_embeddings.weight' in ckpt:
                pretrained_dict = {"main_model."+k[20:]: v for k, v in ckpt.items()}
            elif 'bert.embeddings.word_embeddings.weight' in ckpt:
                pretrained_dict = {"main_model."+k[5:]: v for k, v in ckpt.items()}
            else:
                pretrained_dict = {"main_model."+k[12:]: v for k, v in ckpt.items()}
            # print(pretrained_dict.keys())
            # print(self.text_encoder.state_dict().keys())
            self.text_encoder.load_state_dict(pretrained_dict, strict=False)
            self.smiles_encoder.load_state_dict(pretrained_dict, strict=False)
            # missing_keys, unexpected_keys = self.text_encoder.load_state_dict(pretrained_dict, strict=False)
            # print(missing_keys)
            # print(unexpected_keys)
        # self.feature_extractor.freeze()


        self.graph_proj_head = nn.Sequential(
          nn.Linear(self.gin_hidden_dim, self.gin_hidden_dim),
          nn.ReLU(inplace=True),
          nn.Linear(self.gin_hidden_dim, self.projection_dim)
        )
        self.text_proj_head = nn.Sequential(
          nn.Linear(self.bert_hidden_dim, self.bert_hidden_dim),
          nn.ReLU(inplace=True),
          nn.Linear(self.bert_hidden_dim, self.projection_dim)
        )
        self.smiles_proj_head = nn.Sequential(
          nn.Linear(self.bert_hidden_dim, self.bert_hidden_dim),
          nn.ReLU(inplace=True),
          nn.Linear(self.bert_hidden_dim, self.projection_dim)
        )

    def forward(self, features_graph, features_text):
        batch_size = features_graph.size(0)

        # normalized features
        features_graph = F.normalize(features_graph, dim=-1)
        features_text = F.normalize(features_text, dim=-1)

        # cosine similarity as logits
        logits_per_graph = features_graph @ features_text.t() / self.temperature
        logits_per_text = logits_per_graph.t()

        labels = torch.arange(batch_size, dtype=torch.long, device=self.device)
        loss_graph = F.cross_entropy(logits_per_graph, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        loss = (loss_graph + loss_text) / 2

        return logits_per_graph, logits_per_text, loss

    def configure_optimizers(self):
        # High lr because of small dataset and small model
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def training_step(self, batch, batch_idx):
        graph, smiles, mask, text1, mask1, text2, mask2 = batch

        graph_rep = self.graph_encoder(graph)
        graph_rep = self.graph_proj_head(graph_rep)

        smiles_rep = self.smiles_encoder(smiles, mask)
        smiles_rep = self.smiles_proj_head(smiles_rep)

        text1_rep = self.text_encoder(text1, mask1)
        text1_rep = self.text_proj_head(text1_rep)

        text2_rep = self.text_encoder(text2, mask2)
        text2_rep = self.text_proj_head(text2_rep)

        _, _, loss1 = self.forward(graph_rep, text1_rep)
        _, _, loss2 = self.forward(graph_rep, text2_rep)
        _, _, loss3 = self.forward(graph_rep, smiles_rep)

        loss = (loss1 + loss2 + loss3) / 3.0

        self.log("train_loss", loss)
        return loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("GINSimclr")
        # train mode
        parser.add_argument('--temperature', type=float, default=0.1, help='the temperature of NT_XentLoss')
        # GIN
        parser.add_argument('--gin_hidden_dim', type=int, default=300)
        parser.add_argument('--gin_num_layers', type=int, default=5)
        parser.add_argument('--drop_ratio', type=float, default=0.0)
        parser.add_argument('--graph_pooling', type=str, default='sum')
        # Bert
        parser.add_argument('--bert_hidden_dim', type=int, default=768, help='')
        parser.add_argument('--bert_pretrain', action='store_false', default=True)
        parser.add_argument('--projection_dim', type=int, default=256)
        # optimization
        parser.add_argument('--lr', type=float, default=0.0001, help='optimizer learning rate')
        parser.add_argument('--weight_decay', type=float, default=1e-5, help='optimizer weight decay')
        return parent_parser

