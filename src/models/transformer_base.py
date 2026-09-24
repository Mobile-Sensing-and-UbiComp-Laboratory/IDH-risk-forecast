from src.models.layers import *
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader

class TransformerBase(nn.Module):
    def __init__(
        self, 
        emb_size=16,
        num_classes=2,
        num_heads=8,
        dropout=0.1,
        hidden_size=256,
        add_norm=True,
        # data related
        in_channel=4,
        seq_length=256,
    ):
        super().__init__()
        
        # Encoder variant: convolutional patches and positional encoding instead of the active RNN baseline.
        # # Input shape: (Series_length, Channel, 1)
        # self.encoder = nn.Sequential(
        #     EmbConvBlock(in_channel, 8, emb_size, hidden_size=emb_size*4),
        #     CheckShape(None, key=lambda x: x.squeeze(1)),
        #     tAPE(emb_size, dropout=dropout, max_len=seq_length),
        # )
        self.encoder = RNNEncoder(in_channel, emb_size) # if rnn baseline

        # Backbone variant: enable attention/feed-forward blocks with the matching forward call below; check constructor arguments.
        # self.transformer = nn.Sequential(
        #     Attention(emb_size, num_heads, seq_len=seq_length, dropout=dropout, add_norm=add_norm),
        #     FeedForward(emb_size, hidden_size, dropout=0.5, add_norm=add_norm),
        # )

        self.fuse = nn.Sequential(
            CheckShape(None, key=lambda x: x.permute(0, 2, 1)), # to N, E, L
            nn.AdaptiveAvgPool1d(1), # N, E, 1
            CheckShape(None, key=lambda x: x.squeeze(2))
        )

        # Historical single-task head: requires matching single-task forward, targets, and loss.
        # self.output = nn.Linear(emb_size, num_classes)
        # multitask downstream tasks
        self.out_heads = {
            "fall20": nn.Linear(emb_size, num_classes),
            "fall30": nn.Linear(emb_size, num_classes),
            "nadir90": nn.Linear(emb_size, num_classes),
            "nadir100": nn.Linear(emb_size, num_classes),
            "hemo": nn.Linear(emb_size, num_classes),
            "kdoqi": nn.Linear(emb_size, num_classes),
            "next_measures": nn.Linear(emb_size, 4)
        }
        self.out_heads = nn.ModuleDict(self.out_heads)
        
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
    
    def loss_func(self, out_pack, in_pack):
        final_loss = 0
        for task in self.loss_funcs:
            task_weight, task_loss_f = self.loss_funcs[task]

            # get loss functions and weights
            curr_loss_f = task_loss_f
            
            # calculate loss
            final_loss = final_loss + task_weight*curr_loss_f(out_pack[task], in_pack[task].to(DEVICE))
        return final_loss

    def forward(self, x):
        # Input shape: (N, C, L)
        x = x.unsqueeze(1).permute(0, 1, 3, 2) # (N, 1, L, C)

        # embedding
        out = self.encoder(x) # (N, L, E)

        # Backbone variant: paired with the optional self.transformer definition above.
        # # transformer (attention)
        # out = self.transformer(out) # (N, L, E)

        # fuse and out
        out = self.fuse(out) # (N, E)

        # output
        pred = dict()
        for task in self.out_heads:
            pred[task] = self.out_heads[task](out)

        return pred
        out = self.output(out) # (N, num_class)
        return out

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
        loss_func = nn.CrossEntropyLoss(weight=torch.Tensor([0.286, 0.714]).to(DEVICE))

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
                out = self(data_pack["bodies"])
                loss = self.loss_func(out, data_pack)

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
                y_preds, y_trues = dict(), dict()
                for task in self.loss_funcs:
                    if task == 'next_measures':
                        continue
                    y_preds[task] = list()
                    y_trues[task] = list()
                    
                with torch.no_grad():
                    for data_pack in tqdm(eval_loader):
                        out = self(data_pack["bodies"])
                        loss = self.loss_func(out, data_pack)
                        eval_loss += loss.detach().cpu().item() * len(data_pack["bodies"])
                        total_val_num +=  len(data_pack["bodies"])

                        # update stored record
                        for task in y_preds:
                            y_preds[task] += torch.softmax(out[task], dim=1)[:, 1].detach().cpu().numpy().tolist()
                            y_trues[task] += data_pack[task].detach().cpu().numpy().tolist()
                last_pred = {"y_preds": y_preds, "y_trues": y_trues}
                eval_loss /= total_val_num
                eval_losses.append(eval_loss)
                eval_i.append(iteration)
                print("Check Val Loss:", eval_losses[-1])
        
        return {
            "train_losses": train_losses, # list
            "train_i": train_i,
            "eval_losses": eval_losses, # list
            "eval_i": eval_i,
            "last_pred": last_pred # dict{list, list}
        }

if __name__ == '__main__':
    num_sample = 5
    seq_length = 32
    in_channel = 3

    x = torch.rand((num_sample, in_channel, seq_length))
    layer = TransformerBase(
        emb_size=16,
        num_classes=2,
        num_heads=8,
        dropout=0.1,
        hidden_size=256,
        add_norm=True,
        # data related
        in_channel=in_channel,
        seq_length=seq_length,
    )
    y = layer(x) # expect (5, 2)
    print("Output shape:", y.shape)
