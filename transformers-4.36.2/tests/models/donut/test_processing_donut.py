# coding=utf-8
# Copyright 2022 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import unittest

from transformers import DonutProcessor


DONUT_PRETRAINED_MODEL_NAME = "naver-clova-ix/donut-base"


class DonutProcessorTest(unittest.TestCase):
    def setUp(self):
        self.processor = DonutProcessor.from_pretrained(DONUT_PRETRAINED_MODEL_NAME)

    def test_token2json(self):
        expected_json = {
            "name": "John Doe",
            "age": "99",
            "city": "Atlanta",
            "state": "GA",
            "zip": "30301",
            "phone": "123-4567",
            "nicknames": [{"nickname": "Johnny"}, {"nickname": "JD"}],
        }

        sequence = (
            "<s_name>John Doe</s_name><s_age>99</s_age><s_city>Atlanta</s_city>"
            "<s_state>GA</s_state><s_zip>30301</s_zip><s_phone>123-4567</s_phone>"
            "<s_nicknames><s_nickname>Johnny</s_nickname>"
            "<sep/><s_nickname>JD</s_nickname></s_nicknames>"
        )
        actual_json = self.processor.token2json(sequence)

        self.assertDictEqual(actual_json, expected_json)
