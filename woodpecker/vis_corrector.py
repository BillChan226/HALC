import sys
sys.path.append("/data/xyq/bill/MiniGPT-4/woodpecker")
from models.preprocessor import PreProcessor
from models.entity_extractor import EntityExtractor
from models.detector import Detector
from models.questioner import Questioner
from models.answerer import Answerer
from models.claim_generator import ClaimGenerator
from models.refiner import Refiner
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
import time

LLaMA_MODEL_PATH = "/home/czr/contrast_decoding_LVLMs/model_checkpoints/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93"

class Corrector:
    def __init__(self, args) -> None:
        # init all the model
        
        if args.llm_backbone == "llama":
            self.LLM_backbone = AutoModelForCausalLM.from_pretrained(
        LLaMA_MODEL_PATH, 
        load_in_8bit=True, device_map={"": Accelerator().process_index}
        )
            self.LLM_tokenizer = AutoTokenizer.from_pretrained(LLaMA_MODEL_PATH)

        self.LLM_MODEL = {"model": self.LLM_backbone, "tokenizer": self.LLM_tokenizer}

        self.preprocessor = PreProcessor(args, self.LLM_MODEL)
        self.entity_extractor = EntityExtractor(args)
        self.detector = Detector(args)
        self.questioner = Questioner(args)
        self.answerer = Answerer(args)
        self.claim_generator = ClaimGenerator(args)
        self.refiner = Refiner(args)
        
        print("Finish loading models.")

    
    def correct(self, sample: Dict):
        '''
        sample is Dict containing at least two fields:
            'input_desc': A passage that contains a description of the image.
            'input_img': Path to a local image 
        '''
        print("\n pre sample:", sample)
        sample = self.preprocessor.generate_sentences(sample)
        print("\n generate_sentences", sample)
        sample = self.entity_extractor.extract_entity(sample)
        print("\n extract_entity", sample)
        # sample['named_entity'] = ['man.book']
        sample = self.detector.detect_objects(sample)
        print("\n detect_objects", sample)
        sample = self.questioner.generate_questions(sample)
        sample = self.answerer.generate_answers(sample)
        sample = self.claim_generator.generate_claim(sample)
        sample = self.refiner.generate_output(sample)
        
        return sample

    def batch_correct(self, samples: List[Dict]):

        return [self.correct(sample) for sample in tqdm(samples, total=len(samples))]