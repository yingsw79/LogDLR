import os
import time
import torch
import lightning as L
from argparse import ArgumentParser
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sentence_transformers import SentenceTransformer
from model.logdlr.logdlr import LogDLR
from utils.data import LogDataModule, CycleZip, Chain
from utils.utils import freeze, timer


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset_S', type=str, default='Thunderbird')
    parser.add_argument('--dataset_T', type=str, default='BGL')
    parser.add_argument('--train_size_S', type=int, default=100000)
    parser.add_argument('--train_size_T', type=int, default=10000)
    parser.add_argument('--log_file_S', type=str, default='dataset/Thunderbird/Thunderbird.log')
    parser.add_argument('--log_file_T', type=str, default='dataset/BGL/BGL.log')

    parser.add_argument('--num_fields', type=int, default=10)
    parser.add_argument('--nrows', type=int, default=10**7)
    parser.add_argument('--window_size', type=int, default=20)
    parser.add_argument('--step_size', type=int, default=4)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--random_seed', type=int, default=8)

    parser.add_argument('--trained_model_path', type=str, default='')
    parser.add_argument('--embedding_model_name', type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--embed_dim', type=int, default=384)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--num_warmup_steps', type=int, default=4000)  # 2000
    parser.add_argument('--latent_dim', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--output_dim', type=int, default=2)

    return parser.parse_args()


@timer
def main():
    args = parse_args()
    kwargs = vars(args)
    L.seed_everything(args.random_seed)

    model = LogDLR(
        num_training_steps=args.max_epochs * args.train_size_S / args.batch_size,
        **kwargs,
    )
    sentence_embedding_model = SentenceTransformer(args.embedding_model_name, device='cuda')
    freeze(sentence_embedding_model)

    data_S = LogDataModule(
        log_file=args.log_file_S,
        sentence_embedding_model=sentence_embedding_model,
        train_size=args.train_size_S,
        **kwargs,
    )
    data_T = LogDataModule(
        log_file=args.log_file_T,
        sentence_embedding_model=sentence_embedding_model,
        train_size=args.train_size_T,
        **kwargs,
    )

    checkpoint_path = os.path.join(
        f'saved_model/{args.dataset_S}-{args.dataset_T}',
        time.strftime('%Y-%m-%d_%H-%M-%S'),
    )

    checkpoint = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename='{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        save_weights_only=True,
        monitor='val_loss',
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=True)

    trainer = L.Trainer(
        logger=CSVLogger(save_dir=checkpoint_path),
        callbacks=[checkpoint, early_stopping],
        max_epochs=args.max_epochs,
        enable_progress_bar=False,
    )

    if args.trained_model_path:
        model_path = args.trained_model_path
    else:
        trainer.fit(
            model=model,
            train_dataloaders=CycleZip(data_S.data_loader('train'), data_T.data_loader('train')),
            val_dataloaders=Chain(data_S.data_loader('eval'), data_T.data_loader('eval')),
        )
        model_path = checkpoint.best_model_path

    print(f'Test results for the target: {args.dataset_T}')
    model.load_state_dict(torch.load(model_path)['state_dict'])
    trainer.predict(model, data_T.data_loader('val'), return_predictions=False)
    trainer.test(model, data_T.data_loader('test'))


if __name__ == '__main__':
    main()
