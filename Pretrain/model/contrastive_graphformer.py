import torch
import torch.nn as nn
from model.graphormer.graphormer_graph_encoder import GraphormerGraphEncoder
from model.bert import TextEncoder
import torch.nn.functional as F
import pytorch_lightning as pl
from utils.lr import PolynomialDecayLR
from torch import optim


class GraphormerSimclr(pl.LightningModule):
    def __init__(
            self,
            temperature,
            num_atoms,
            num_in_degree,
            num_out_degree,
            num_edges,
            num_spatial,
            num_edge_dis,
            edge_type,
            multi_hop_max_dist,
            num_encoder_layers,
            graph_embed_dim,
            graph_ffn_embed_dim,
            graph_attention_heads,
            dropout,
            attention_dropout,
            activation_dropout,
            encoder_normalize_before,
            pre_layernorm,
            apply_graphormer_init,
            activation_fn,
            graphormer_pretrain,
            graph_self,
            bert_hidden_dim,
            bert_pretrain,
            projection_dim,
            warmup_updates,
            tot_updates,
            peak_lr,
            end_lr,
            weight_decay,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.temperature = temperature

        self.graphormer_pretrain = graphormer_pretrain
        self.graph_self = graph_self

        self.bert_hidden_dim = bert_hidden_dim
        self.bert_pretrain = bert_pretrain
        self.projection_dim = projection_dim

        self.warmup_updates = warmup_updates
        self.tot_updates = tot_updates
        self.peak_lr = peak_lr
        self.end_lr = end_lr
        self.weight_decay = weight_decay
                
        self.num_atoms = num_atoms
        self.num_in_degree = num_in_degree
        self.num_out_degree = num_out_degree
        self.num_edges = num_edges
        self.num_spatial = num_spatial
        self.num_edge_dis = num_edge_dis
        self.edge_type = edge_type
        self.multi_hop_max_dist = multi_hop_max_dist
        self.num_encoder_layers = num_encoder_layers
        self.graph_embed_dim = graph_embed_dim
        self.graph_ffn_embed_dim = graph_ffn_embed_dim
        self.graph_attention_heads = graph_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.encoder_normalize_before = encoder_normalize_before
        self.pre_layernorm = pre_layernorm
        self.apply_graphormer_init = apply_graphormer_init
        self.activation_fn = activation_fn

        self.graph_encoder = GraphormerGraphEncoder(
            num_atoms=self.num_atoms,
            num_in_degree=self.num_in_degree,
            num_out_degree=self.num_out_degree,
            num_edges=self.num_edges,
            num_spatial=self.num_spatial,
            num_edge_dis=self.num_edge_dis,
            edge_type=self.edge_type,
            multi_hop_max_dist=self.multi_hop_max_dist,
            num_encoder_layers=self.num_encoder_layers,
            embedding_dim=self.graph_embed_dim,
            ffn_embedding_dim=self.graph_ffn_embed_dim,
            num_attention_heads=self.graph_attention_heads,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            activation_dropout=self.activation_dropout,
            encoder_normalize_before=self.encoder_normalize_before,
            pre_layernorm=self.pre_layernorm,
            apply_graphormer_init=self.apply_graphormer_init,
            activation_fn=self.activation_fn,
        )

        if self.graphormer_pretrain:  # load pretrained graphormer
            ckpt = torch.load('graphormer_pretrained/checkpoint_best_pcqm4mv2.pt',
                              map_location=self.device)
            state_dict = ckpt['model']
            # # print(state_dict.keys())
            for k in list(state_dict.keys()):
                if k.startswith('encoder.graph_encoder.'):
                    # remove prefix module.
                    state_dict[k[len("encoder.graph_encoder."):]] = state_dict[k]
                    del state_dict[k]
            # print(state_dict.keys())
            # print(self.graph_encoder.state_dict().keys())
            self.graph_encoder.load_state_dict(state_dict, strict=False)
            # missing_keys, unexpected_keys = self.graph_encoder.load_state_dict(state_dict, strict=False)
            # print(missing_keys)
            # print(unexpected_keys)

        self.text_encoder = TextEncoder(pretrained=self.bert_pretrain)
        if self.bert_pretrain:
            ckpt = torch.load('bert_pretrained/ckpt_KV_1.pt')
            if 'module.ptmodel.bert.embeddings.word_embeddings.weight' in ckpt:
                pretrained_dict = {"main_model."+k[20:]: v for k, v in ckpt.items()}
            elif 'bert.embeddings.word_embeddings.weight' in ckpt:
                pretrained_dict = {"main_model."+k[5:]: v for k, v in ckpt.items()}
            else:
                pretrained_dict = {"main_model."+k[12:]: v for k, v in ckpt.items()}
            # print(pretrained_dict.keys())
            # print(self.text_encoder.state_dict().keys())
            self.text_encoder.load_state_dict(pretrained_dict, strict=False)
            # missing_keys, unexpected_keys = self.text_encoder.load_state_dict(pretrained_dict, strict=False)
            # print(missing_keys)
            # print(unexpected_keys)
        # self.feature_extractor.freeze()

        self.graph_proj_head = nn.Sequential(
          nn.Linear(self.graph_embed_dim, self.graph_embed_dim),
          nn.ReLU(inplace=True),
          nn.Linear(self.graph_embed_dim, self.projection_dim)
        )
        self.text_proj_head = nn.Sequential(
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

        labels = torch.arange(batch_size, dtype=torch.long, device=self.device)  # 大小为B
        loss_graph = F.cross_entropy(logits_per_graph, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        loss = (loss_graph + loss_text) / 2

        return logits_per_graph, logits_per_text, loss

    def configure_optimizers(self):
        # High lr because of small dataset and small model
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.peak_lr,
            weight_decay=self.weight_decay
            )

        lr_scheduler = {
            'scheduler': PolynomialDecayLR(
                optimizer,
                warmup_updates=self.warmup_updates,
                tot_updates=self.tot_updates,
                lr=self.peak_lr,
                end_lr=self.end_lr,
                power=1.0,
            ),
            'name': 'learning_rate',
            'interval': 'step',
            'frequency': 1,
        }
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        aug1, aug2, text1, mask1, text2, mask2 = batch

        _, graph1_rep = self.graph_encoder(aug1)
        graph1_rep = self.graph_proj_head(graph1_rep)

        _, graph2_rep = self.graph_encoder(aug2)
        graph2_rep = self.graph_proj_head(graph2_rep)

        text1_rep = self.text_encoder(text1, mask1)
        text1_rep = self.text_proj_head(text1_rep)

        text2_rep = self.text_encoder(text2, mask2)
        text2_rep = self.text_proj_head(text2_rep)

        _, _, loss11 = self.forward(graph1_rep, text1_rep)
        _, _, loss12 = self.forward(graph1_rep, text2_rep)
        _, _, loss21 = self.forward(graph2_rep, text1_rep)
        _, _, loss22 = self.forward(graph2_rep, text2_rep)

        if self.graph_self:
            _, _, loss_graph_self = self.forward(graph1_rep, graph2_rep)
            loss = (loss11 + loss12 + loss21 + loss22 + loss_graph_self) / 5.0
        else:
            loss = (loss11 + loss12 + loss21 + loss22) / 4.0

        self.log("train_loss", loss)
        return loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("GraphormerSimclr")
        # train mode
        parser.add_argument('--temperature', type=float, default=0.1, help='the temperature of NT_XentLoss')
        # Graphormer
        parser.add_argument('--num_atoms', type=int, default=512*9, help='number of atom types')
        parser.add_argument('--num_in_degree', type=int, default=512, help='number of in degree types')
        parser.add_argument('--num_out_degree', type=int, default=512, help='number of out degree types')
        parser.add_argument('--num_edges', type=int, default=512*3, help='number of edge types')
        parser.add_argument('--num_spatial', type=int, default=512, help='number of spatial types')
        parser.add_argument('--num_edge_dis', type=int, default=128, help='number of edge dis types')
        parser.add_argument('--edge_type', type=str, default='multi_hop', help='edge type')
        parser.add_argument('--num_encoder_layers', type=int, default=12, help='encoder layer num')
        parser.add_argument('--graph_embed_dim', type=int, default=768, help='d_model')
        parser.add_argument('--graph_ffn_embed_dim', type=int, default=768, help='d_ff')
        parser.add_argument('--graph_attention_heads', type=int, default=32, help='head nums')
        parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate in FFN')
        parser.add_argument('--attention_dropout', type=float, default=0.1, help='dropout rate in MSA')
        parser.add_argument('--activation_dropout', type=float, default=0.1, help='dropout rate')
        parser.add_argument('--encoder_normalize_before', action='store_false', default=True)
        parser.add_argument('--pre_layernorm', action='store_true', default=False)
        parser.add_argument('--apply_graphormer_init', action='store_false', default=True)
        parser.add_argument('--activation_fn', type=str, default='gelu', help='activation function')

        parser.add_argument('--graphormer_pretrain', action='store_false', default=True)
        parser.add_argument('--graph_self', action='store_true', help='use graph self-supervise or not', default=False)

        # Bert
        parser.add_argument('--bert_hidden_dim', type=int, default=768, help='')
        parser.add_argument('--bert_pretrain', action='store_false', default=True)
        parser.add_argument('--projection_dim', type=int, default=256)
        # optimization
        parser.add_argument('--warmup_updates', type=int, default=60000)
        parser.add_argument('--tot_updates', type=int, default=1000000)
        parser.add_argument('--peak_lr', type=float, default=2e-4)
        parser.add_argument('--end_lr', type=float, default=1e-9)
        parser.add_argument('--weight_decay', type=float, default=0.0, help='optimizer weight decay')
        return parent_parser

