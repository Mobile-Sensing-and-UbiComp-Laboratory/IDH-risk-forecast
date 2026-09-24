from src.models.layers import *
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader

from torch.distributions.normal import Normal

class PreRealAutoReg(nn.Module):
    def __init__(
        self,
        emb_size=48,
        num_heads=12,
        num_layers=1, 
        num_classes=2, 
        dropout=0.1,
        # data related
        in_channel=8,
        pre_size=18,
        seq_length=64,
        # other setting
        use_cls=False,
        late_fuse=True,
        seg_embed=False
    ):
        super().__init__()

        self.late_fuse = late_fuse
        self.seg_embed = seg_embed
        self.emb_size = emb_size
        ### Special Token
        self.use_cls = use_cls
        if use_cls:
            self.cls_embed = nn.Parameter(torch.rand(1, 1, emb_size), requires_grad=True)
        if self.seg_embed:
            self.segs = {
                "lab": nn.Parameter(torch.rand(1, 1, emb_size), requires_grad=True),
                "med": nn.Parameter(torch.rand(1, 1, emb_size), requires_grad=True),
                "disease": nn.Parameter(torch.rand(1, 1, emb_size), requires_grad=True),
                "realtime": nn.Parameter(torch.rand(1, 1, emb_size), requires_grad=True),
                "summaries": nn.Parameter(torch.rand(1, 1, emb_size), requires_grad=True),
            }
            self.segs = nn.ModuleDict(self.segs)

        ### Real-time encoder
        self.realtime_encoder = nn.Sequential( # (N, L, C)
            # Patching
            EmbConvBlock(in_channel, 8, emb_size, hidden_size=emb_size*4),
            # position embedding
            tAPE(emb_size, dropout=dropout, max_len=seq_length),
        ) # (N, L, E)

        ### Pre-session encoder
        self.presession_encoder = nn.Sequential(
            nn.Linear(pre_size, emb_size*4),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(emb_size*4, emb_size),
        ) # (N, L, E)

        ### transformer backbone
        if use_cls: # use [CLS] as single representation 
            self.fusion_layer = nn.Sequential(
                CheckShape(None, key=lambda x: x[:, 0, :])
            )
        else: # use average pooling
            self.fusion_layer = nn.Sequential(
                CheckShape(None, key=lambda x: x.permute(0, 2, 1)), # to N, E, L
                nn.AdaptiveAvgPool1d(1), # N, E, 1
                CheckShape(None, key=lambda x: x.squeeze(2)), 
            )

        self.transformer_backbone = nn.Sequential(
            Transformer(
                num_layers=num_layers,
                emb_size=emb_size,
                num_heads=num_heads,
                hidden_size=emb_size*4,
                dropout=dropout,
            ),
            self.fusion_layer
        )

        # output heads
        latent_size = emb_size # +emb_size if concat
        self.out_head = VAE_Latent(latent_size, 4)
        self.hemo_out = nn.Linear(latent_size, 2)
        self.kdoqi_out = nn.Linear(latent_size, 2)
        
        # loss functions and weights
        self.loss_funcs = {
            "fall20": (0.8, nn.CrossEntropyLoss(weight=torch.Tensor([0.15, 0.85]).to(DEVICE))),
            "fall30":  (0.8, nn.CrossEntropyLoss(weight=torch.Tensor([0.1, 0.9]).to(DEVICE))),
            "nadir90": (0.8, nn.CrossEntropyLoss(weight=torch.Tensor([0.05, 0.95]).to(DEVICE))),
            "nadir100": (0.8, nn.CrossEntropyLoss(weight=torch.Tensor([0.1, 0.9]).to(DEVICE))),
            "hemo": (0.8, nn.CrossEntropyLoss(weight=torch.Tensor([0.1, 0.9]).to(DEVICE))),
            "kdoqi": (0.8, nn.CrossEntropyLoss(weight=torch.Tensor([0.05, 0.95]).to(DEVICE))),
            # autoregressive helper task
            "next_measures": (0.8, nn.L1Loss()),
            "next_measures_normal": (1.0, self.autoreg_normal_loss)
        }
    
    def autoreg_normal_loss(self, y_pred, y_true, pd):
        y_pred_prob = pd.cdf(y_pred)
        y_true_prob = pd.cdf(y_true)
        return torch.mean(torch.maximum(y_pred_prob, y_true_prob) - torch.minimum(y_pred_prob, y_true_prob))
    
    def loss_func(self, out_pack, in_pack):
        final_loss = 0
        for task in out_pack:
            if task == 'pd':
                continue

            task_weight, task_loss_f = self.loss_funcs[task]

            if task == "next_measures_normal":
                final_loss = final_loss + task_weight*task_loss_f(out_pack[task], in_pack["next_measures"][:, 0], out_pack["pd"])
            else:
                final_loss = final_loss + task_weight*task_loss_f(out_pack[task], in_pack[task])
        return final_loss
    
    def get_prob(self, pd, upper_bound):
        pos = pd.cdf(upper_bound).unsqueeze(1)
        neg = 1 - pos
        return torch.cat((pos, neg), dim=-1)

    def prob_out(self, x, pd, z, out): # (N, L, C)
        # loss during training
        out_preds = {
            "hemo": self.hemo_out(out),
            "kdoqi": self.kdoqi_out(out),
            "next_measures": z,
            "next_measures_normal": z[:, 0],
            "pd": pd
        }

        curr_val = x[:, -1, 0]
        if not self.training:
            out_preds["fall20"] = self.get_prob(pd, curr_val - (20/128))
            out_preds["fall30"] = self.get_prob(pd, curr_val - (30/128))
            out_preds["nadir90"] = self.get_prob(pd, torch.tensor([90/128]).expand(curr_val.shape).to(DEVICE))
            out_preds["nadir100"] = self.get_prob(pd, torch.tensor([100/128]).expand(curr_val.shape).to(DEVICE))
        return out_preds
        
    
    def forward(self, x, pre_x, decays): # Input shape: (N, C, L)
        # encoding
        N, C, L = x.shape
        _, C_P, L_P = pre_x.shape
        x = x.permute(0, 2, 1) # (N, L, C)
        realtime_out = self.realtime_encoder(x)
        presession_out = self.presession_encoder(pre_x) + decays.unsqueeze(2)
        
        # cls token
        if self.use_cls:
            cls_token = self.cls_embed.expand(N, 1, self.emb_size)
            realtime_out = torch.cat((cls_token, realtime_out), dim=1)
            presession_out = torch.cat((cls_token, presession_out), dim=1)
        # segment embedding
        if self.seg_embed:
            realtime_out = realtime_out + self.segs["realtime"].expand(N, L, self.emb_size)
            presession_out = presession_out + self.segs["summaries"].expand(N, L_P, self.emb_size)

        # transformer backbone
        if self.late_fuse:
            realtime_out = self.transformer_backbone(realtime_out)
            presession_out = self.transformer_backbone(presession_out)
            out = 0.5*realtime_out + 0.5*presession_out
        else: # early fuse
            all_out = torch.cat((realtime_out, presession_out), dim=1)
            out = self.transformer_backbone(all_out)
        
        z, mu, var = self.out_head(out, latent_only=False)
        return self.prob_out(x, Normal(mu[:, 0], var[:, 0]), z, out)
    
    def fit(
            self, 
            train_dataset, 
            eval_dataset,
            batch_size=512,
            epochs=50,
            lr=1e-3,
            weight_decay=1e-5,
        ):

        # main train pipeline
        print("Construct data loader")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

        # construct optimizer
        optimizer = optim.Adam(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.997)

        # stored variables
        train_losses, eval_losses = list(), list()
        train_i, eval_i = list(), list()
        iteration = 0
        last_pred = None
        print("Start training")
        for e in tqdm(range(epochs)):
            self.train()
            for data_pack in tqdm(train_loader):
                iteration += 1
                # forward
                out_pack = self(data_pack["bodies"], data_pack["summaries"], data_pack["decays"])
                loss = self.loss_func(out_pack, data_pack)

                # backprop
                optimizer.zero_grad() # clear cache
                loss.backward() # calculate gradient
                optimizer.step() # update parameters
                scheduler.step() # update learning rate

                # update record
                train_losses.append(loss.detach().cpu().item())
                train_i.append(iteration)

                torch.cuda.empty_cache()

            # validation
            if e%2 == 0 or e == epochs-1:
                print("Eval")
                last_pred = None
                self.eval()
                eval_loss = 0
                total_val_num = 0
                y_preds, y_trues = dict(), dict()
                for task in self.loss_funcs:
                    if task in ['next_measures', 'next_measures_normal']:
                        continue
                    y_preds[task] = list()
                    y_trues[task] = list()

                with torch.no_grad():
                    for data_pack in tqdm(eval_loader):
                        out_pack = self(data_pack["bodies"], data_pack["summaries"], data_pack["decays"])
                        loss = self.loss_func(out_pack, data_pack)

                        # update record
                        eval_loss += loss.detach().cpu().item() * len(data_pack["bodies"])
                        total_val_num +=  len(data_pack["bodies"])

                        # update stored record
                        for task in y_preds:
                            y_preds[task] += torch.softmax(out_pack[task], dim=1)[:, 1].detach().cpu().numpy().tolist()
                            y_trues[task] += data_pack[task].detach().cpu().numpy().tolist()
                
                last_pred = {"y_preds": y_preds, "y_trues": y_trues}
                
                eval_loss /= total_val_num
                eval_losses.append(eval_loss)
                eval_i.append(iteration)
                print("Check Val Loss:", eval_losses[-1])

                torch.cuda.empty_cache()
        
        return {
            "train_losses": train_losses, # list
            "train_i": train_i,
            "eval_losses": eval_losses, # list
            "eval_i": eval_i,
            "last_pred": last_pred # dict{list, list}
        }


