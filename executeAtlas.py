import requests
import time
import subprocess
respuesta = input("¿Escribir pregunta manual? (y/n) ")

if respuesta.lower() == "y":
    print("Introduce una pregunta")
    question=input()
    print("Introduce instancia")
    subject=input()
elif respuesta.lower() == "n":
    question="How much money does Jeff Bezos has?"
    subject="Jeff Bezos"
    print("La pregunta es: " + question)
    print("La instancia es: " + subject)
else:
    print("Respuesta inválida. Por favor ingrese 'y' o 'n'.")
t1=time.time()
url = 'https://en.wikipedia.org/w/api.php'
params = {
        'action': 'query',
        'format': 'json',
        'titles': subject,
        'prop': 'extracts',
        'exintro': True,
        'explaintext': True,
    }
 
response = requests.get(url, params=params)
data = response.json()
 
page = next(iter(data['query']['pages'].values()))
#print(page['extract'])
filtered=page['extract'].replace('"', '')
passageWiki=[{"id": 0, "title": subject, "text": filtered}]
t2=time.time()
tiempoRetrieve=t2-t1
print(dict)
print("El tiempo de retrieve de passage ha sido de " + str(tiempoRetrieve) + " s")

#A partir de aquí es evaluate.py
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.cuda
import torch.distributed as dist

from src import dist_utils, slurm, util
from src.index_io import load_or_initialize_index, save_embeddings_and_index
from src.model_io import create_checkpoint_directories, load_or_initialize_atlas_model
from src.options import get_options
from src.tasks import get_task
os.environ["TOKENIZERS_PARALLELISM"] = "true"
class Options:
    def __init__(self):
        size="large"
        model_path= './atlas_data/models/atlas_nq/'+ size
        reader_model_type='google/t5-'+size+'-lm-adapt'
        self.accumulation_steps = 1
        self.alpha = 1.0
        self.beta2 = 0.999
        self.checkpoint_dir = './atlas_data/experiments/'
        self.clip = 1.0
        self.closed_book = False
        self.compute_crossattention_stats = False
        self.decoder_format = None
        self.decoder_prompt_format = None
        self.device = torch.device(type='cuda')
        self.dont_write_passages = False
        self.dropout = 0.1
        self.encoder_format = '{query} title: {title} context: {text}'
        self.epsilon = 1e-06
        self.eval_data = ['./input.json']
        self.eval_freq = 500
        self.faiss_code_size = None
        self.faiss_index_type = 'flat'
        self.filtering_overretrieve_ratio = 2
        self.freeze_retriever_steps = -1
        self.generation_length_penalty = 1.0
        self.generation_max_length = 200
        self.generation_min_length = None
        self.generation_num_beams = 1
        self.global_rank = 0
        self.gold_score_mode = 'ppmean'
        self.index_mode = 'flat'
        self.is_distributed = False
        self.is_main = True
        self.is_slurm_job = False
        self.load_index_path = None
        self.local_rank = -1
        self.log_freq = 100
        self.lr = 0.0001
        self.lr_retriever = 1e-05
        self.main_port = 15482
        self.max_lm_context_ratio = 0.5
        self.max_passages = -1
        self.min_lm_context_ratio = 0.5
        self.min_words_per_lm_instance = None
        self.mlm_mean_noise_span_length = 3
        self.mlm_noise_density = 0.15
        self.model_path =model_path
        self.multi_gpu = False
        self.multi_node = False
        self.multiple_choice_eval_permutations = 'single'
        self.multiple_choice_num_options = 4
        self.multiple_choice_train_permutations = 'single'
        self.n_context = 40
        self.n_gpu_per_node = 1
        self.n_nodes = 1
        self.n_to_rerank_with_retrieve_with_rerank = 128
        self.name = 'pruebaUse'
        self.node_id = 0
        self.passages = ['./atlas_data/usePassage.jsonl']
        self.per_gpu_batch_size = 1
        self.per_gpu_embedder_batch_size = 512
        self.precision = 'fp32'
        self.qa_prompt_format = 'question: {question} answer: <extra_id_0>'
        self.query_side_retriever_training = False
        self.reader_model_type = reader_model_type
        self.refresh_index = '-1'
        self.retrieve_only = False
        self.retrieve_with_rerank=False
        self.retrieve_with_rerank=False
        self.retriever_format='{title} {text}'
        self.retriever_model_path='facebook/contriever'
        self.retriever_n_context=40
        self.save_freq=5000
        self.save_index_n_shards=128
        self.save_index_path=None
        self.save_optimizer=False
        self.scheduler='cosine'
        self.scheduler_steps=None
        self.seed=0
        self.shard_grads=False
        self.shard_optim=False
        self.shuffle=False
        self.target_maxlength=None
        self.task='qa'
        self.temperature_gold=0.01
        self.temperature_score=0.01
        self.text_maxlength=100
        self.total_steps=1000
        self.train_data=[]
        self.train_retriever=False
        self.use_file_passages=True
        self.use_gradient_checkpoint_reader=False
        self.use_gradient_checkpoint_retriever=False
        self.warmup_steps=1000
        self.weight_decay=0.1
        self.world_size=1
        self.write_results=True
def evaluate(model, opt,passage, step=None):
    model.eval()
    metrics = defaultdict(lambda: [])
    dataset_wpred = []
    unwrapped_model = util.get_unwrapped_model_if_wrapped(model)
    reader_tokenizer = unwrapped_model.reader_tokenizer
    task = get_task(opt, reader_tokenizer)
    strQuestion="question:" + question + " answer: <extra_id_0>"  
    query=[strQuestion]
    retrieved_passages=passage
    reader_tokens, _ = unwrapped_model.tokenize_passages(query, retrieved_passages)
    generation = unwrapped_model.generate(
        reader_tokens, query, None
    )
    for k, g in enumerate(generation):
        if opt.decoder_prompt_format is not None:
            query_ids = reader_tokenizer.encode(
                opt.decoder_prompt_format.format_map({"query": query[k]}), add_special_tokens=False
            )
            g = g[len(query_ids) + 1 :]
        pred = reader_tokenizer.decode(g, skip_special_tokens=True)
    return pred



opt = Options()

torch.manual_seed(opt.seed)
slurm.init_distributed_mode(opt)
slurm.init_signal_handler()

logger = util.init_logger(opt.is_main, opt.is_distributed, os.path.join(opt.checkpoint_dir, "run.log"))

logger.info(f"world size: {dist_utils.get_world_size()}")

#index, passages = load_or_initialize_index(opt)
model, _, _, _, _, opt, step = load_or_initialize_atlas_model(opt, eval_only=True)

logger.info("Start Evaluation")
dist_utils.barrier()
t3=time.time()
#result = evaluate2(model, index, opt, data_path, passageWiki, step)
result = evaluate(model, opt, passageWiki, step)
t4=time.time()
print("Answer")
print(result)
print("El tiempo de evaluación ha sido de " + str(t4-t3) + " s")





