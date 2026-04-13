import argparse
import torch

from train.train import Trainer
from train.eval import Evaluator


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Signet Project'
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'eval', 'test'],
        default='train'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None
    )

    args = parser.parse_args()

    # ====================================
    # TRAIN MODE
    # ====================================
    if args.mode == 'train':

        trainer = Trainer(args.config)

        trainer.train()

    # ====================================
    # EVAL MODE (验证集)
    # ====================================
    elif args.mode == 'eval':

        if args.checkpoint is None:

            raise ValueError(
                "eval模式必须提供checkpoint"
            )

        trainer = Trainer(args.config)

        trainer.model.load_state_dict(
            torch.load(args.checkpoint)
        )

        evaluator = Evaluator(
            trainer.model,
            trainer.val_loader,
            trainer.device
        )

        evaluator.evaluate()

    # ====================================
    # TEST MODE (测试集)
    # ====================================
    elif args.mode == 'test':

        if args.checkpoint is None:

            raise ValueError(
                "test模式必须提供checkpoint"
            )

        trainer = Trainer(args.config)

        trainer.model.load_state_dict(
            torch.load(args.checkpoint)
        )

        evaluator = Evaluator(
            trainer.model,
            trainer.test_loader,
            trainer.device
        )

        evaluator.evaluate()