from src.models.layers import *
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader

# Import required by the optional pretrained session-encoder variant below.
# from src.models.pre_session_base import *


class PreRealBaseMulti(nn.Module):
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

        # Historical encoder variant: SiT with latent projection; adapt the VAE_Latent constructor and forward interface before reuse.
        # ### Original encoder
        # self.encoder = nn.Sequential( # (N, L, E)
        #     SiT(
        #         num_layers=num_layers,
        #         emb_size=emb_size,
        #         num_heads=num_heads,
        #         dropout=dropout,
        #         in_channel=in_channel,
        #         seq_length=seq_length
        #     ),
        #     # fuse
        #     CheckShape(None, key=lambda x: x.permute(0, 2, 1)), # to N, E, L
        #     nn.AdaptiveAvgPool1d(1), # N, E, 1
        #     CheckShape(None, key=lambda x: x.squeeze(2)), 
        #     VAE_Latent(emb_size)
        # ) # (N, E)
        # ####

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

        # # output heads
        latent_size = emb_size # +emb_size if concat
        # Fusion variant: add a separately trained session encoder; paired with the optional pre_latent block below.
        # self.pre_session_encoder = PreSessionBase(
        #     in_channel=pre_size,
        #     emb_size=emb_size,
        #     num_classes=num_classes,
        #     num_heads=num_heads,
        #     dropout=dropout
        # )

        self.vae_layer = VAE_Latent(emb_size, emb_size)

        # multitask downstream tasks
        self.out_heads = {
            "fall20": nn.Linear(latent_size, num_classes),
            "fall30": nn.Linear(latent_size, num_classes),
            "nadir90": nn.Linear(latent_size, num_classes),
            "nadir100": nn.Linear(latent_size, num_classes),
            "hemo": nn.Linear(latent_size, num_classes),
            "kdoqi": nn.Linear(latent_size, num_classes),
            "next_measures": nn.Linear(latent_size, 4)
        }
        self.out_heads = nn.ModuleDict(self.out_heads)
        # Historical single-task output variant: also adapt forward, targets, and loss to a single head.
        # self.out_heads = nn.Linear(latent_size, num_classes)
        
        # loss functions and weights
        self.loss_funcs = {
            "fall20": (0.8, nn.CrossEntropyLoss(weight=torch.Tensor([0.15, 0.85]).to(DEVICE))),
            "fall30":  (0.8, nn.CrossEntropyLoss(weight=torch.Tensor([0.1, 0.9]).to(DEVICE))),
            "nadir90": (0.8, nn.CrossEntropyLoss(weight=torch.Tensor([0.05, 0.95]).to(DEVICE))),
            "nadir100": (0.8, nn.CrossEntropyLoss(weight=torch.Tensor([0.1, 0.9]).to(DEVICE))),
            "hemo": (0.8, nn.CrossEntropyLoss(weight=torch.Tensor([0.1, 0.9]).to(DEVICE))),
            "kdoqi": (0.8, nn.CrossEntropyLoss(weight=torch.Tensor([0.05, 0.95]).to(DEVICE))),
            # autoregressive helper task
            "next_measures": (0.4, nn.L1Loss())
        }

        # Class-weight variant: scalar positive-class weights; paired with the loss-construction block below.
        # self.loss_funcs = {
        #     "fall20": (0.8, 0.714),
        #     "fall30":  (0.8, 0.5),
        #     "nadir90": (0.8,0.9),
        #     "nadir100": (0.8, 0.8),
        #     "hemo": (0.8, 0.8),
        #     "kdoqi": (0.8, 0.9),
        #     # autoregressive helper task
        #     "next_measures": (0.4, -1)
        # }
    
    def loss_func(self, out_pack, in_pack):
        final_loss = 0
        for task in self.loss_funcs:
            task_weight, task_loss_f = self.loss_funcs[task]

            # get loss functions and weights
            curr_loss_f = task_loss_f

            # Class-weight variant: construct losses from the scalar weights above rather than stored loss modules.
            # if task_loss_f > 0:
            #     curr_loss_f =  nn.CrossEntropyLoss(weight=torch.Tensor([1-task_loss_f, task_loss_f]).to(DEVICE))
            # else:
            #     curr_loss_f = nn.L1Loss()
            
            # calculate loss
            final_loss = final_loss + task_weight*curr_loss_f(out_pack[task], in_pack[task])
        return final_loss
    
    def forward(self, x, pre_x, decays): # Input shape: (N, C, L)
        # encoding
        N, C, L = x.shape
        _, C_P, L_P = pre_x.shape
        x = x.permute(0, 2, 1) # (N, 1, L, C)
        realtime_out = self.realtime_encoder(x)
        presession_out = self.presession_encoder(pre_x) + decays.unsqueeze(2)
        
        # cls token
        if self.use_cls:
            cls_token = self.cls_embed.expand(N, 1, self.emb_size)
            realtime_out = torch.cat((cls_token, realtime_out), dim=1)
            presession_out = torch.cat((cls_token, presession_out), dim=1)
        # segment embedding
        if self.seg_embed:
            realtime_out = realtime_out + self.segs["realtime"]
            presession_out = presession_out + self.segs["summaries"]

        # transformer backbone
        if self.late_fuse:
            realtime_out = self.transformer_backbone(realtime_out)
            presession_out = self.transformer_backbone(presession_out)
            out = 0.5*realtime_out + 0.5*presession_out
        else: # early fuse
            all_out = torch.cat((realtime_out, presession_out), dim=1)
            out = self.transformer_backbone(all_out)
        
        out = self.vae_layer(out, latent_only=True)

        
        # Fusion variants: add a separate session embedding or concatenate modality embeddings; adapt head dimensions.
        # get pre latent
        # session_level_pred, pre_latent = self.pre_session_encoder(pre_x, decays, latent_state=True)
        # out = out + pre_latent
        # out = torch.cat((realtime_out, presession_out), dim=1)

        # output
        pred = dict()
        for task in self.out_heads:
            pred[task] = self.out_heads[task](out)

        return pred
    
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
                    if task == 'next_measures':
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


