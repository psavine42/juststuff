import unittest
from src.regimes import *
from src.model.arguments import *
from arg_config import *
from src.algo.recursives import *
import  src.problem.dataset as ds
from src.layout import *


class TestNN(unittest.TestCase):
    def test_super1(self):
        args = super_args('supervised1')
        zdim = 64

        enc = EncodeState4(args.inst.num_spaces, zdim)
        dec = DecodeState4(args.inst.num_spaces, zdim)
        model = GBP(enc, dec, 5, zdim, 12, shared_size=12)

        trainer = GBPTrainer(None, model=model, argobj=args)

        dataset = ds.build_dataset(args.ds, ProbStack, args.inst, args.env)
        print(len(dataset))
        trainer.train_vanilla_supervised(dataset, args.train)

    def test_dataset(self):
        args = super_args('supervised1')
        dataset = ds.build_dataset(args.ds, ProbStack, args.inst, args.env)
        print(len(dataset))
