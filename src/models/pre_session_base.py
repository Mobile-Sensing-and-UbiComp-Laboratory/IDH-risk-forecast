from src.models.layers import *
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader

class PreSessionBase(nn.Module):
    def __init__(
        self, 
        emb_size=16,
        num_classes=2,
        num_heads=8,
        dropout=0.1,
        # data related
        in_channel=18,
    ):
        super().__init__()
        
        # Input shape: (Series_length, Channel, 1)
        self.embedding = nn.Sequential(
            nn.Linear(in_channel, emb_size*4),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(emb_size*4, emb_size),
        )

        # Backbone variant: explicit attention/feed-forward blocks; check constructor compatibility before reuse.
        # self.transformer = nn.Sequential(
        #     Attention(emb_size, num_heads, seq_len=seq_length, dropout=dropout, add_norm=add_norm),
        #     FeedForward(emb_size, hidden_size, dropout=0.5, add_norm=add_norm),
        # )
        self.transformer = Transformer(
            num_layers=1,
            emb_size=emb_size,
            num_heads=num_heads,
            hidden_size=emb_size*4,
            dropout=dropout,
        )

        self.fuse = nn.Sequential(
            CheckShape(None, key=lambda x: x.permute(0, 2, 1)), # to N, E, L
            nn.AdaptiveAvgPool1d(1), # N, E, 1
            CheckShape(None, key=lambda x: x.squeeze(2))
        )

        self.output = nn.Linear(emb_size, num_classes)

    def forward(self, x, decays, latent_state=False):
        # Input shape: (N, L, C)
        # decays: (N, L)

        # History-weighting variant: normalize decay weights across sessions; handle all-zero padding before enabling.
        # decays = decays / torch.sum(decays, dim=1, keepdim=True)
        decays = decays.unsqueeze(2) # (N, L, 1)

        # embedding
        emb = decays + self.embedding(x) # (N, L, E)

        # transformer (attention)
        out = self.transformer(emb) # (N, L, E)

        # fuse and out
        out = self.fuse(out) # (N, E)
        final_out = self.output(out) # (N, num_class)

        if latent_state:
            return final_out, out
        return final_out

    def fit(
            self, 
            train_dataset, 
            eval_dataset,
            batch_size=512,
            epochs=50,
            lr=1e-3,
            weight_decay=1e-5,
            i=0
        ):

        print("Construct data loader")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)

        # declare loss
        loss_func = nn.CrossEntropyLoss(weight=torch.Tensor([0.644, 0.356]).to(DEVICE))

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
                out = self(data_pack["summaries"], data_pack["decays"])
                loss = loss_func(out, data_pack["session_level_labels"])

                # backprop
                optimizer.zero_grad() # clear cache
                loss.backward() # calculate gradient
                optimizer.step() # update parameters
                scheduler.step() # update learning rate

                # update record
                train_losses.append(loss.detach().cpu().item())
                train_i.append(iteration)

            # validation
            if e%2 == 0 or e == epochs-1:
                print("Eval")
                self.eval()
                eval_loss = 0
                total_val_num = 0
                y_preds, y_trues = list(), list()
                with torch.no_grad():
                    for data_pack in tqdm(eval_loader):
                        out = self(data_pack["summaries"], data_pack["decays"])
                        loss = loss_func(out, data_pack["session_level_labels"])
                        eval_loss += loss.detach().cpu().item() * len(data_pack["session_level_labels"])
                        total_val_num +=  len(data_pack["session_level_labels"])

                        # update stored record
                        y_preds += torch.softmax(out, dim=1).detach().cpu().numpy().tolist()
                        y_trues += data_pack["session_level_labels"].detach().cpu().numpy().tolist()
                last_pred = {"y_preds": y_preds, "y_trues": y_trues}
                eval_loss /= total_val_num
                eval_losses.append(eval_loss)
                eval_i.append(iteration)
        
        return {
            "train_losses": train_losses, # list
            "train_i": train_i,
            "eval_losses": eval_losses, # list
            "eval_i": eval_i,
            "last_pred": last_pred # dict{list, list}
        }


