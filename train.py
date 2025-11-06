from tps_flow.parsing import parse_train_args
args = parse_train_args()
from tps_flow.logger import get_logger
logger = get_logger(__name__)

import torch, os, wandb
from tps_flow.dataset import tps_flowDataset
from tps_flow.wrapper import TPS_Flow
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
import pytorch_lightning as pl


torch.set_float32_matmul_precision('medium')

if args.wandb:
    wandb.init(
        entity=os.environ["WANDB_ENTITY"],
        settings=wandb.Settings(start_method="fork"),
        project="tps_flow",
        name=args.run_name,
        config=args,
    )


trainset = tps_flowDataset(args, split=args.train_split)

if args.overfit:
    valset = trainset    
else:
    valset = tps_flowDataset(args, split=args.val_split, repeat=args.val_repeat)

train_loader = torch.utils.data.DataLoader(
    trainset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=True,
)

val_loader = torch.utils.data.DataLoader(
    valset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
)
model = TPS_Flow(args)
    
trainer = pl.Trainer(
    accelerator="gpu" if torch.cuda.is_available() else 'auto',
    max_epochs=args.epochs,
    limit_train_batches=args.train_batches or 1.0,
    limit_val_batches=0.0 if args.no_validate else (args.val_batches or 1.0),
    num_sanity_val_steps=0,
    precision=args.precision,
    enable_progress_bar=not args.wandb or os.getlogin() == 'hstark',
    gradient_clip_val=args.grad_clip,
    default_root_dir=os.environ["MODEL_DIR"], 
    callbacks=[
        ModelCheckpoint(
            dirpath=os.environ["MODEL_DIR"], 
            save_top_k=-1,
            every_n_epochs=args.ckpt_freq,
        ),
        #  ModelCheckpoint(
        #     dirpath=os.environ["MODEL_DIR"], 
        #     # save_top_k=-1,
        #     save_top_k=5, 
        #     monitor="val_loss",
        #     mode="min", 
        #     every_n_epochs=args.ckpt_freq,
        #     filename="{epoch}-{val_loss:.4f}",
        # ),
        ModelSummary(max_depth=2),
    ],
    accumulate_grad_batches=args.accumulate_grad,
    val_check_interval=args.val_freq,
    check_val_every_n_epoch=args.val_epoch_freq,
    logger=False,
    # gradient_clip_val=5.0,
    # gradient_clip_algorithm="norm"
)

# torch.manual_seed(137)
# np.random.seed(137)

# if args.pretrained:
#     model.load_state_dict(torch.load(args.pretrained))

if args.validate:
    trainer.validate(model, val_loader, ckpt_path=args.ckpt)
else:
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.ckpt)


# torch.set_printoptions(precision=8, sci_mode=False)