import warnings
warnings.simplefilter("ignore", UserWarning)

import logging
import torch
import numpy as np
from modules.tokenizers import Tokenizer
from modules.dataloaders import LADataLoader
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.trainer import Trainer
from modules.loss import compute_loss
import models
from models.baseline import LAMRGModel as BaselineModel
from config import opts


def main():
    # parse arguments
    # args = parse_agrs()
    args = opts.parse_opt()
    logging.info(str(args))

    # fix random seeds
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # create tokenizer
    tokenizer = Tokenizer(args)

    # create data loader
    train_dataloader = LADataLoader(args, tokenizer, split='train', shuffle=True)
    val_dataloader = LADataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = LADataLoader(args, tokenizer, split='test', shuffle=False)

    # build model architecture
    if str(args.version) == "900":
        model_name = "BaselineModel"
        model = BaselineModel(args, tokenizer)
    else:
        model_name = f"LAMRGModel_v{args.version}"
        model_cls = getattr(models, model_name, None)
        if model_cls is None:
            raise ValueError(f"Unsupported model version: {args.version}")
        model = model_cls(args, tokenizer)
    logging.info(f"Model name: {model_name} \tModel Layers:{args.num_layers}")

    # get function handles of loss and metrics
    criterion = compute_loss
    metrics = compute_scores

    # build optimizer, learning rate scheduler
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    # build trainer and start to train
    trainer = Trainer(model, criterion, metrics, optimizer, args, lr_scheduler, train_dataloader, val_dataloader, test_dataloader)
    trainer.train()
    logging.info(str(args))


if __name__ == '__main__':
    main()
