# -*- coding: utf-8 -*-
# @File    : test_NCFDataConvertor.py
# @Author  : Hua Guo
# @Time    : 2021/10/28 下午4:07
# @Disc    :
from unittest import TestCase

from src.DataConvert.NCFDataConvertor import NCFDataConvertor


class TestNCFDataConvertor(TestCase):
    def setUp(self) -> None:
        self.raw_dir = 'data/debug/raw'
        self.output_dir = 'logs/debug/cleaned'

    def test_smoke(self):
        convertor = NCFDataConvertor(input_dir=self.raw_dir, output_dir=self.output_dir)
        convertor.convert()