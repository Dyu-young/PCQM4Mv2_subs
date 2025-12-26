import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, Sequential, global_add_pool, global_mean_pool, DeepGCNLayer
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import TransformerConv as Glayer
from constants import YMIN,YMAX
from torch.nn import Dropout, Linear, ReLU, LayerNorm
import torch.nn.functional as F
import torch
import utils
from muon import Muon

class DGCN(pl.LightningModule):

    def __init__(self, max_node_fea, num_feas, config, accumulate_grad_batches=1):
        super(DGCN, self).__init__()
        
        # Enable manual optimization to handle multiple optimizers correctly
        self.automatic_optimization = False

        self.num_feas = num_feas
        self.lr = config.lr
        self.wd = config.wd
        self.epochs = config.epochs
        
        # Accumulation steps
        self.acc_steps = accumulate_grad_batches

        # hidden layer node features
        self.hidden = config.H1
        self.En = config.En
        Ee = config.Ee
        self.max_node_fea = max_node_fea + 1
        self.emb = torch.nn.Embedding(self.max_node_fea, self.En)
        heads = config.heads
        H = self.hidden // heads
        total_hidden = H * heads

        act = None if config.act == 'None' else config.act
        act = eval(config.act)
        
        # First layer
        self.l1 = Glayer(self.num_feas*self.En, H, heads, edge_dim=3, beta=config.beta)
        self.ln1 = LayerNorm(total_hidden)
        
        # Deep layers
        ls = []
        lns = []
        for i in range(config.layers):
            ls.append(DeepGCNLayer(Glayer(total_hidden, H, heads, edge_dim=3, beta=config.beta)))
            lns.append(LayerNorm(total_hidden))
            
        self.layers = torch.nn.ModuleList(ls)
        self.lns = torch.nn.ModuleList(lns)
        
        self.w = torch.nn.Sequential(Linear(total_hidden, 4), ReLU(), Linear(4,1))
        self.out = torch.nn.Sequential(Linear(total_hidden, 1),
                                       Linear(1, 1))      

    def forward(self, x, edge_index, edge_attr, batch_index):
        x = self.emb(x).view(-1,self.num_feas*self.En)
        
        # First layer + LN
        x = self.l1(x,edge_index,edge_attr.float()/4.0)
        x = self.ln1(x)
        x = F.relu(x)
        
        # Deep layers + LN
        for i, (l, ln) in enumerate(zip(self.layers, self.lns)):
            x = l(x,edge_index,edge_attr.float()/4.0)
            x = ln(x)
            x = F.relu(x)
        
        # Numerical stable attention pooling
        a = self.w(x)
        # Subtract max for stability within each graph
        from torch_scatter import scatter_max
        a_max, _ = scatter_max(a, batch_index, dim=0)
        a = torch.exp(a - a_max[batch_index])
        
        x = global_add_pool(x*a, batch_index)
        a = global_add_pool(a, batch_index)
        x = x/(a + 1e-8) # Avoid division by zero
        x1 = self.out(x)
        x2 = torch.clip(x1,YMIN,YMAX)
        x_out = (x1+x2)/2
        return x_out.squeeze()
    
    def _f(self, batch, batch_index):
        x, edge_index = batch.x, batch.edge_index
        edge_attr = batch.edge_attr
        batch_index = batch.batch
        x_out = self.forward(x, edge_index, edge_attr, batch_index)
        return x_out
    
    def _loss(self, batch, batch_index, tag):
        x_out = self._f(batch, batch_index)
        loss = F.smooth_l1_loss(x_out, batch.y, beta=0.1)
        x_out = torch.clip(x_out,YMIN,YMAX)
        mae = F.l1_loss(x_out, batch.y)
        self.log(f"{tag}_mae", mae, batch_size = batch.y.shape[0], prog_bar=True)
        self.log(f"{tag}_loss", loss, batch_size = batch.y.shape[0], prog_bar=True)
        return loss

    def training_step(self, batch, batch_index):
        # Manual optimization logic
        opt_muon, opt_adam = self.optimizers()
        
        # Calculate loss
        loss = self._loss(batch, batch_index, 'train')
        
        # Accumulation factor
        # Scale loss for accumulation
        loss = loss / self.acc_steps
        
        # Manual backward
        self.manual_backward(loss)
        
        # Step optimizer every acc_steps
        if (batch_index + 1) % self.acc_steps == 0:
            # Gradient clipping (Tightened to 0.5)
            self.clip_gradients(opt_muon, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
            self.clip_gradients(opt_adam, gradient_clip_val=0.5, gradient_clip_algorithm="norm")

            opt_muon.step()
            opt_adam.step()
            
            opt_muon.zero_grad()
            opt_adam.zero_grad()
        
        return loss

    def on_train_epoch_end(self):
        # Step schedulers at the end of epoch
        sch_muon, sch_adam = self.lr_schedulers()
        
        # If the scheduler is based on epoch (like CosineAnnealingLR), step it here
        # Note: Some schedulers might need metrics (ReduceLROnPlateau), CosineAnnealingLR does not.
        sch_muon.step()
        sch_adam.step()

    def validation_step(self, batch, batch_index):
        return self._loss(batch, batch_index, 'valid')
        
    def predict_step(self, batch, batch_index):
        x_out = self._f(batch, batch_index)
        return torch.clip(x_out,YMIN,YMAX)

    def configure_optimizers(self):
        # Filter parameters for Muon (>= 2D) and AdamW (< 2D or Embeddings)
        muon_params = []
        adamw_params = []
        
        for name, p in self.named_parameters():
            if p.requires_grad:
                if p.ndim >= 2 and 'emb' not in name:
                    muon_params.append(p)
                else:
                    adamw_params.append(p)

        # Lower Muon LR to 0.01 for better stability in deep networks
        muon_lr = 0.01 if self.lr < 0.01 else self.lr
        
        # Create optimizers
        opt_muon = Muon(muon_params, lr=muon_lr, momentum=0.95)
        opt_adam = torch.optim.AdamW(adamw_params, lr=self.lr, weight_decay=self.wd)
        
        # Create schedulers
        sch_muon = torch.optim.lr_scheduler.CosineAnnealingLR(opt_muon, self.epochs)
        sch_adam = torch.optim.lr_scheduler.CosineAnnealingLR(opt_adam, self.epochs)
             
        # Return as lists
        return [opt_muon, opt_adam], [sch_muon, sch_adam]

if __name__ == "__main__":
    config = utils.load_yaml('../yaml/gnn.yaml')
    model = DGCN(36,9,config)
    # 5 layers: 0.2008 val mae
    # 10 layers: 0.172 val mae
    # 10 lyaers, smooth l1 loss beta=0.1: 0.169
