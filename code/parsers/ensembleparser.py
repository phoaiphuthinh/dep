# -*- coding: utf-8 -*-

import os
from datetime import datetime, timedelta

import code
import torch
import torch.distributed as dist
from code.utils import Config, Dataset
from code.utils.field import Field
from code.utils.logging import init_logger, logger
from code.utils.metric import Metric
from code.utils.parallel import DistributedDataParallel as DDP
from code.utils.parallel import is_master
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR


class EnsembleParser(object):

    NAME = None
    MODEL = None

    def __init__(self, args, model, origin, addition=None, optimizer=None, scheduler=None):
        self.args = args
        self.model = model
        self.origin = origin
        self.addition = addition
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train_2_time(self, train, dev, test, buckets=32, batch_size=5000, clip=5.0, epochs=5000, epochs_add=1000, patience=100, **kwargs):
        args = self.args.update(locals())
        init_logger(logger)

        self.origin.train()
        self.addition.train()

        if dist.is_initialized():
            args.batch_size = args.batch_size // dist.get_world_size()

        logger.info("Loading the data")
        train = Dataset(self.origin, args.train, **args)
        dev = Dataset(self.origin, args.dev)
        test = Dataset(self.origin, args.test)

        train.build(args.batch_size, args.buckets, True, dist.is_initialized())

        dev.build(args.batch_size, args.buckets)
        test.build(args.batch_size, args.buckets)
        
        if self.addition:
            train_add = Dataset(self.addition, args.train_add, **args)
            dev_add = Dataset(self.addition, args.dev_add)
            test_add = Dataset(self.addition, args.test_add)

            buck_sizes = train_add.build(args.batch_size, args.buckets, True, dist.is_initialized())
            dev_add.build(args.batch_size, args.buckets)
            test_add.build(args.batch_size, args.buckets)

            logger.info(f"\n{'train:':6} {train_add}\n")
            logger.info(f"\n{'train:':6} {dev_add}\n")
            logger.info(f"\n{'train:':6} {test_add}\n")

        logger.info(f"\n{'train:':6} {train}\n{'dev:':6} {dev}\n{'test:':6} {test}\n")
        
        if dist.is_initialized():
            self.model = DDP(self.model, device_ids=[args.local_rank], find_unused_parameters=True)

        strategy = 2

        if strategy == 1:
            #train source
            elapsed = timedelta()
            best_e, best_metric = 1, Metric()
            epoch = 0
            self.model.train_model_(True)
            while epoch < args.epochs_add:
                start = datetime.now()
                epoch += 1
                logger.info(f"Epoch {epoch} / {args.epochs_add}:")
                self._train_2_time(train_add.loader, source_train=True)
                loss, dev_metric = self._evaluate(dev_add.loader, source_train=True)
                logger.info(f"{'dev:':5} loss: {loss:.4f} - {dev_metric}")
                loss, test_metric = self._evaluate(test_add.loader, source_train=True)
                logger.info(f"{'test:':5} loss: {loss:.4f} - {test_metric}")

                t = datetime.now() - start
                logger.info(f"{t}s elapsed\n")


            from transformers import AdamW, get_linear_schedule_with_warmup

            param_optimizer = list(self.model.named_parameters())
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
                {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
            ]
            num_train_optimization_steps = int(args.epochs * len(train.loader))
            self.optimizer = AdamW(
                optimizer_grouped_parameters, lr=args.lr, correct_bias=False
            ) 
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=5, num_training_steps=num_train_optimization_steps
            )

            elapsed = timedelta()
            best_e, best_metric = 1, Metric()
            epoch = 0
            self.model.train_model_(False)
            while epoch < args.epochs:
                start = datetime.now()
                epoch += 1
                logger.info(f"Epoch {epoch} / {args.epochs}:")
                self._train_2_time(train.loader)
                loss, dev_metric = self._evaluate(dev.loader, source_train=False)
                logger.info(f"{'dev:':5} loss: {loss:.4f} - {dev_metric}")
                loss, test_metric = self._evaluate(test.loader)
                logger.info(f"{'test:':5} loss: {loss:.4f} - {test_metric}")

                t = datetime.now() - start
                # save the model if it is the best so far
                if dev_metric > best_metric:
                    best_e, best_metric = epoch, dev_metric
                    if is_master():
                        self.save(args.path)
                    logger.info(f"{t}s elapsed (saved)\n")
                else:
                    logger.info(f"{t}s elapsed\n")
                elapsed += t
                if epoch - best_e >= args.patience:
                    break
        elif strategy == 2:
            
            rate = 5

            from transformers import AdamW, get_linear_schedule_with_warmup

            param_optimizer = list(self.model.named_parameters())
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
                {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
            ]
            num_train_optimization_steps = int(args.epochs * len(train.loader))
            self.optimizer = AdamW(
                optimizer_grouped_parameters, lr=args.lr, correct_bias=False
            ) 
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=5, num_training_steps=num_train_optimization_steps
            )

            elapsed = timedelta()
            best_e, best_metric = 1, Metric()
            epoch = 0
            while epoch < args.epochs:
                start = datetime.now()
                epoch += 1
                logger.info(f"Epoch {epoch} / {args.epochs}:")
                #train source
                self.model.train_model_(True)
                for k in range(1, rate+1):
                    logger.info(f"sub epoch {k} / {rate}:")
                    self._train_2_time(train_add.loader, source_train=True)
                    loss, dev_metric = self._evaluate(dev_add.loader, source_train=True)
                    logger.info(f"{'dev:':5} loss: {loss:.4f} - {dev_metric}")
                    loss, test_metric = self._evaluate(test_add.loader, source_train=True)
                    logger.info(f"{'test:':5} loss: {loss:.4f} - {test_metric}")

                #train target
                self.model.train_model_(False)
                self._train_2_time(train.loader)
                loss, dev_metric = self._evaluate(dev.loader, source_train=False)
                logger.info(f"{'dev:':5} loss: {loss:.4f} - {dev_metric}")
                loss, test_metric = self._evaluate(test.loader)
                logger.info(f"{'test:':5} loss: {loss:.4f} - {test_metric}")

                t = datetime.now() - start
                # save the model if it is the best so far
                if dev_metric > best_metric:
                    best_e, best_metric = epoch, dev_metric
                    if is_master():
                        self.save(args.path)
                    logger.info(f"{t}s elapsed (saved)\n")
                else:
                    logger.info(f"{t}s elapsed\n")
                elapsed += t
                if epoch - best_e >= args.patience:
                    break

        #test phase
        loss, metric = self.load(**args)._evaluate(test.loader)

        logger.info(f"Epoch {best_e} saved")
        logger.info(f"{'dev:':5} {best_metric}")
        logger.info(f"{'test:':5} {metric}")
        logger.info(f"{elapsed}s elapsed, {elapsed / epoch}s/epoch")

        

    def train(self, train, dev, test, buckets=32, batch_size=5000, clip=5.0, epochs=5000, epochs_add=100, patience=100, **kwargs):

        self.train_2_time(train, dev, test, buckets, batch_size, clip, epochs, epochs_add, patience, **kwargs)
        # args = self.args.update(locals())
        # init_logger(logger)

        # self.origin.train()
        # self.addition.train()

        # if dist.is_initialized():
        #     args.batch_size = args.batch_size // dist.get_world_size()

        # logger.info("Loading the data")
        # train = Dataset(self.origin, args.train, **args)
        # dev = Dataset(self.origin, args.dev)
        # test = Dataset(self.origin, args.test)

        # train.build(args.batch_size, args.buckets, True, dist.is_initialized())
        # dev.build(args.batch_size, args.buckets)
        # test.build(args.batch_size, args.buckets)
        
        # if self.addition:
        #     train_add = Dataset(self.addition, args.train_add, **args)
        #     #dev_add = DatasetPos(self.addition, args.dev_add)
        #     #test_add = DatasetPos(self.addition, args.test_add)

        #     buck_sizes = train_add.build(args.batch_size, args.buckets, True, dist.is_initialized())
        #     #dev_add.build(args.batch_size, args.buckets)
        #     #test_add.build(args.batch_size, args.buckets)

        #     logger.info(f"\n{'train:':6} {train_add}\n")

        # logger.info(f"\n{'train:':6} {train}\n{'dev:':6} {dev}\n{'test:':6} {test}\n")

        # if args.encoder == 'bert':
        #     from transformers import AdamW, get_linear_schedule_with_warmup
        #     steps = len(train.loader) * epochs
        #     self.optimizer = AdamW(
        #         [{'params': c.parameters(), 'lr': args.lr * (1 if n == 'encoder' else args.lr_rate)}
        #          for n, c in self.model.named_children()],
        #         args.lr)
        #     self.scheduler = get_linear_schedule_with_warmup(self.optimizer, int(steps*args.warmup), steps)
        

        # if dist.is_initialized():
        #     self.model = DDP(self.model, device_ids=[args.local_rank], find_unused_parameters=True)

        # elapsed = timedelta()
        # best_e, best_metric = 1, Metric()

        # #for epoch in range(1, args.epochs + 1):
        # epoch = 0
        # while epoch < args.epochs:
        #     start = datetime.now()
        #     epoch += 1
        #     logger.info(f"Epoch {epoch} / {args.epochs}:")
        #     self._train(train.loader, train_add.loader)
        #     loss, dev_metric = self._evaluate(dev.loader)
        #     logger.info(f"{'dev:':5} loss: {loss:.4f} - {dev_metric}")
        #     loss, test_metric = self._evaluate(test.loader)
        #     logger.info(f"{'test:':5} loss: {loss:.4f} - {test_metric}")

        #     t = datetime.now() - start
        #     # save the model if it is the best so far
        #     if dev_metric > best_metric:
        #         best_e, best_metric = epoch, dev_metric
        #         if is_master():
        #             self.save(args.path)
        #         logger.info(f"{t}s elapsed (saved)\n")
        #     else:
        #         logger.info(f"{t}s elapsed\n")
        #     elapsed += t
        #     if epoch - best_e >= args.patience:
        #         break
        # loss, metric = self.load(**args)._evaluate(test.loader)

        # logger.info(f"Epoch {best_e} saved")
        # logger.info(f"{'dev:':5} {best_metric}")
        # logger.info(f"{'test:':5} {metric}")
        # logger.info(f"{elapsed}s elapsed, {elapsed / epoch}s/epoch")

    def evaluate(self, data, buckets=8, batch_size=5000, **kwargs):
        args = self.args.update(locals())
        init_logger(logger, verbose=True)
    

        self.origin.train()
        logger.info("Loading the data")
        dataset = Dataset(self.origin, data)
        dataset.build(args.batch_size, args.buckets)
        logger.info(f"\n{dataset}")

        logger.info("Evaluating the dataset")
        start = datetime.now()
        loss, metric = self._evaluate(dataset.loader)
        elapsed = datetime.now() - start
        logger.info(f"loss: {loss:.4f} - {metric}")
        logger.info(f"{elapsed}s elapsed, {len(dataset)/elapsed.total_seconds():.2f} Sents/s")

        return loss, metric

    def predict(self, data, pred=None, buckets=8, batch_size=5000, prob=False, **kwargs):
        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        self.origin.eval()
        if args.prob:
            self.origin.append(Field('probs'))

        logger.info("Loading the data")
        dataset = Dataset(self.origin, data)
        dataset.build(args.batch_size, args.buckets)
        logger.info(f"\n{dataset}")

        logger.info("Making predictions on the dataset")
        start = datetime.now()
        preds = self._predict(dataset.loader)
        elapsed = datetime.now() - start

        for name, value in preds.items():
            setattr(dataset, name, value)
        if pred is not None and is_master():
            logger.info(f"Saving predicted results to {pred}")
            self.origin.save(pred, dataset)
        logger.info(f"{elapsed}s elapsed, {len(dataset) / elapsed.total_seconds():.2f} Sents/s")

        return dataset

    def _train(self, loader):
        raise NotImplementedError

    def _train_2_time(self, loader):
        raise NotImplementedError

    @torch.no_grad()
    def _evaluate(self, loader):
        raise NotImplementedError
    
    @torch.no_grad()
    def _evaluate_print(self, loader):
        raise NotImplementedError

    @torch.no_grad()
    def _predict(self, loader):
        raise NotImplementedError

    @classmethod
    def build(cls, path, **kwargs):
        raise NotImplementedError

    @classmethod
    def load(cls, path, **kwargs):
        r"""
        Loads a parser with data fields and pretrained model parameters.

        Args:
            path (str):
                - a string with the shortcut name of a pretrained parser defined in ``code.PRETRAINED``
                  to load from cache or download, e.g., ``'crf-dep-en'``.
                - a path to a directory containing a pre-trained parser, e.g., `./<path>/model`.
            kwargs (dict):
                A dict holding the unconsumed arguments that can be used to update the configurations and initiate the model.

        Examples:
            >>> from code import Parser
            >>> parser = Parser.load('biaffine-dep-en')
            >>> parser = Parser.load('./ptb.biaffine.dependency.char')
        """

        args = Config(**locals())
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if os.path.exists(path):
            state = torch.load(path)
        else:
            state = torch.hub.load_state_dict_from_url(code.PRETRAINED[path] if path in code.PRETRAINED else path)
        cls = code.PARSER[state['name']] if cls.NAME is None else cls
        args = state['args'].update(args)
        model = cls.MODEL(**args)
        model.load_pretrained(state['pretrained'])
        model.load_state_dict(state['state_dict'], False)
        model.to(args.device)
        origin = state['origin']
        addition = state['addition']
        return cls(args, model, origin, addition)

    def save(self, path):
        model = self.model
        if hasattr(model, 'module'):
            model = self.model.module
        args = model.args
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        pretrained = state_dict.pop('pretrained.weight', None)
        state = {'name': self.NAME,
                 'args': args,
                 'state_dict': state_dict,
                 'pretrained': pretrained,
                 'origin': self.origin,
                 'addition': self.addition}
        torch.save(state, path)
