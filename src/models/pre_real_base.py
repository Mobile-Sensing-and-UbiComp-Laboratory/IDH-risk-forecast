from src.config import DATA_ROOT, FIGURE_ROOT, SEQ_LENGTH
from src.models.layers import *
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.models.pre_session_base import *

class PreRealBase(nn.Module):
    def __init__(
        self,
        emb_size=16,
        num_heads=8,
        num_classes=2, 
        dropout=0.1,
        in_channel=4,
        seq_length=256,
        pre_size=18,
        pretrain_and_freeze=False
    ):
        super().__init__()

        ### Real-time blocks
        self.encoder = nn.Sequential( # (N, L, E)
            # Encoder variants: direct attention or a Transformer block instead of SiT; adapt input dimensions and constructor arguments.
            # Attention(emb_size, num_heads, seq_len=seq_length, dropout=dropout),
            # Transformer(num_layers=1, emb_size=emb_size, num_heads=num_heads, hidden_size=hidden_size, dropout=0.1),
            SiT(
                num_layers=1,
                emb_size=emb_size,
                num_heads=num_heads,
                dropout=dropout,
                in_channel=in_channel,
                seq_length=seq_length
            ),
            # fuse
            CheckShape(None, key=lambda x: x.permute(0, 2, 1)), # to N, E, L
            nn.AdaptiveAvgPool1d(1), # N, E, 1
            CheckShape(None, key=lambda x: x.squeeze(2)), 
            # Latent-mapping variant: deterministic linear projection; adapt the forward method's latent tuple unpacking.
            # # normal fc map
            # nn.Linear(emb_size, emb_size)
            VAE_Latent(emb_size)
        ) # (N, E)
        ####

        # output heads
        latent_size = emb_size+emb_size # +emb_size if concat
        self.pre_session_encoder = PreSessionBase(
            in_channel=pre_size,
            emb_size=emb_size,
            num_classes=num_classes,
            num_heads=num_heads,
            dropout=dropout
        )
        self.downstream_pred = nn.Linear(latent_size, num_classes) # main objective output
        self.pred_head = nn.Linear(latent_size, 4)
        
        # loss functions
        self.class_loss_f = nn.CrossEntropyLoss(weight=torch.Tensor([0.286, 0.714]).to(DEVICE))
        self.rec_loss_f = nn.L1Loss()
        self.session_class_f = nn.CrossEntropyLoss(weight=torch.Tensor([0.644, 0.356]).to(DEVICE))
        self.kl_loss_f = None # TBD

        # other variables
        self.pretrain_and_freeze = pretrain_and_freeze
    
    def forward(self, x, pre_x, decays): # Input shape: (N, C, L)
        # encoding
        x = x.permute(0, 2, 1) # (N, 1, L, C)
        out, mu, var = self.encoder(x) # if using VAE

        # if during test time, take the expected output
        if not self.training:
            out = mu
        
        # get pre latent
        session_level_pred, pre_latent = self.pre_session_encoder(pre_x, decays, latent_state=True)
        # Fusion variant: add the two embeddings instead of concatenating; also adjust downstream head dimensions.
        # out = out + pre_latent
        out = torch.cat((out, pre_latent), dim=1)

        # downstream
        pred = self.downstream_pred(out) # main output
        next_measure_pred = self.pred_head(out) # next measure prediction

        # fuse
        pre_w = 0
        pred = (1-pre_w)*pred + pre_w*session_level_pred

        return pred, next_measure_pred, mu, var, session_level_pred

    def loss_func(self, out_pack, in_pack):
        class_loss = self.class_loss_f(out_pack["pred"], in_pack["labels"])
        pred_loss = self.rec_loss_f(out_pack["next_measure_pred"], in_pack["next_measures"])
        if not self.pretrain_and_freeze:
            session_level_loss = self.session_class_f(out_pack["session_level_pred"], in_pack["session_level_labels"])
            return 0.8*class_loss + 0.5*session_level_loss + 0.3*pred_loss 
        return 1.0*class_loss + 0.4*pred_loss 
    
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

        if self.pretrain_and_freeze:
            # Pretraining variant: fit the session encoder here before freezing, instead of loading a saved checkpoint.
            # # fit presession eval model
            # self.pre_session_encoder.fit(
            #     train_dataset, 
            #     eval_dataset,
            #     batch_size=batch_size,
            #     epochs=epochs,
            #     lr=lr,
            #     weight_decay=weight_decay
            # )
            model_record_name = (DATA_ROOT + '/exp_res/pre_session_base_model_{}.pt').format(i)
            self.pre_session_encoder.load_state_dict(torch.load(model_record_name))

            # freeze the parameters
            for param in self.pre_session_encoder.parameters():
                param.requires_grad = False

        # main train pipeline
        print("Construct data loader")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)

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
                pred, next_measure_pred, mu, var, session_level_pred = self(data_pack["bodies"], data_pack["summaries"], data_pack["decays"])
                out_pack = {"pred": pred, "next_measure_pred": next_measure_pred, "mu": mu, "var": var, "session_level_pred": session_level_pred}
                loss = self.loss_func(out_pack, data_pack)

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
                        pred, next_measure_pred, mu, var, session_level_pred = self(data_pack["bodies"], data_pack["summaries"], data_pack["decays"])
                        out_pack = {"pred": pred, "next_measure_pred": next_measure_pred, "mu": mu, "var": var, "session_level_pred": session_level_pred}
                        loss = self.loss_func(out_pack, data_pack)

                        # update record
                        eval_loss += loss.detach().cpu().item() * len(data_pack["labels"])
                        total_val_num +=  len(data_pack["labels"])

                        # update stored record
                        y_preds += torch.softmax(pred, dim=1).detach().cpu().numpy().tolist()
                        y_trues += data_pack["labels"].detach().cpu().numpy().tolist()
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
    layer = VAEBase(
        emb_size=16,
        num_heads=8,
        num_classes=2,
        dropout=0.1,
        hidden_size=256,
        add_norm=True,
        # data related
        in_channel=in_channel,
        seq_length=seq_length,
    )
    pred, next_measure_pred, mu, var = layer(x)
    print("In shape", x.shape) # expect (5, 3, 32)
    print("Pred shape:", pred.shape) # expect(5, 2)
    print("Latent mu shape:", mu.shape) # expect (5, 16)
    print("Latent var shape:", var.shape) # expect (5, 16)
    print("Pred shape:", next_measure_pred.shape) # expect (5, 3)
