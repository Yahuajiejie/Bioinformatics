import torch
import numpy as np
from numba import jit, prange, types
from numba.typed import Dict, List as NumbaList
import cython
from typing import List, Tuple, Set
from numba.typed import Dict
from numba import types
from collections import defaultdict
from Bio.Seq import Seq
import random
import itertools

# 全局常量和预计算
BASE_TO_INT = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 0}
INT_TO_BASE = ['A', 'T', 'G', 'C']

@jit(nopython=True, cache=True)
def generate_all_kmers(k):
    """生成所有可能的k-mer整数编码"""
    total_kmers = 4 ** k
    kmers = np.empty(total_kmers, dtype=np.int32)
    for i in range(total_kmers):
        kmers[i] = i
    return kmers

@jit(nopython=True, cache=True)
def int_to_kmer_string(kmer_int, k):
    """将整数编码的k-mer转换为字符串"""
    bases = np.empty(k, dtype=np.int32)
    temp = kmer_int
    for i in range(k-1, -1, -1):
        bases[i] = temp % 4
        temp //= 4
    return bases

def build_kmer_to_token_mapping(tokenizer, k=6):
    """预构建k-mer到token ID的映射表"""
    print("Building k-mer to token mapping...")
    kmer_to_token = {}
    
    # 生成所有可能的k-mer
    bases = ['A', 'T', 'G', 'C']
    all_kmers = [''.join(p) for p in itertools.product(bases, repeat=k)]
    
    # 批量tokenize所有k-mer
    batch_size = 1000
    for i in range(0, len(all_kmers), batch_size):
        batch_kmers = all_kmers[i:i+batch_size]
        batch_tokens = tokenizer(batch_kmers, padding=False, truncation=False, add_special_tokens=False)
        
        for j, kmer in enumerate(batch_kmers):
            # 将k-mer编码为整数
            kmer_int = 0
            for base in kmer:
                kmer_int = kmer_int * 4 + BASE_TO_INT[base]
            kmer_to_token[kmer_int] = batch_tokens['input_ids'][j][0]  # 假设每个k-mer对应一个token
    
    print(f"Built mapping for {len(kmer_to_token)} k-mers")
    return kmer_to_token

@jit(nopython=True, parallel=True, cache=True)
def fast_sequence_to_int_array(seq_str, base_map):
    """快速将DNA序列转换为整数数组"""
    seq_len = len(seq_str)
    result = np.empty(seq_len, dtype=np.int32)
    
    for i in prange(seq_len):
        char = seq_str[i]
        if char == 'A':
            result[i] = 0
        elif char == 'T':
            result[i] = 1
        elif char == 'G':
            result[i] = 2
        elif char == 'C':
            result[i] = 3
        else:  # N or other
            result[i] = 0
    return result

@jit(nopython=True, parallel=True, cache=True)
def fast_kmer_to_ids(seq_array, kmer_size, window_size, flank):
    """
    直接从序列数组生成k-mer的整数编码
    返回每个滑动窗口的所有k-mer编码
    """
    seq_len = len(seq_array)
    valid_positions = seq_len - 2 * flank
    kmers_per_window = window_size - kmer_size + 1
    
    # 预分配结果数组
    result = np.empty((valid_positions, kmers_per_window), dtype=np.int32)
    
    for i in prange(valid_positions):
        window_start = i
        window_end = i + window_size
        
        for j in range(kmers_per_window):
            kmer_start = window_start + j
            kmer_val = 0
            
            # 计算k-mer的整数编码
            for k in range(kmer_size):
                if kmer_start + k < seq_len:
                    kmer_val = kmer_val * 4 + seq_array[kmer_start + k]
                else:
                    kmer_val = kmer_val * 4  # N或越界当作A处理
                    
            result[i, j] = kmer_val
    
    return result

@jit(nopython=True, cache=True)
def fast_kmer_ids_to_token_ids(kmer_ids, kmer_to_token_dict):
    """将k-mer整数编码转换为token IDs"""
    rows, cols = kmer_ids.shape
    token_ids = np.empty((rows, cols), dtype=np.int32)
    
    for i in range(rows):
        for j in range(cols):
            kmer_id = kmer_ids[i, j]
            if kmer_id in kmer_to_token_dict:
                token_ids[i, j] = kmer_to_token_dict[kmer_id]
            else:
                token_ids[i, j] = 0  # UNK token
    
    return token_ids

@jit(nopython=True, parallel=True, cache=True)
def fast_coordinate_matching_vectorized(pred_coords, true_coords):
    """向量化的坐标匹配"""
    # 使用集合操作会更快，但numba对set支持有限
    # 这里用一个简化的向量化版本
    tp_mask = np.zeros(len(pred_coords), dtype=np.bool_)
    
    for i in prange(len(pred_coords)):
        for j in range(len(true_coords)):
            if pred_coords[i] == true_coords[j]:
                tp_mask[i] = True
                break
    
    return tp_mask

@jit(nopython=True, cache=True)
def fast_reverse_complement_array(seq_array):
    """快速计算反向互补序列"""
    seq_len = len(seq_array)
    result = np.empty(seq_len, dtype=np.int32)
    
    # 反向互补映射: A(0)->T(1), T(1)->A(0), G(2)->C(3), C(3)->G(2)
    complement_map = np.array([1, 0, 3, 2], dtype=np.int32)
    
    for i in range(seq_len):
        result[seq_len - 1 - i] = complement_map[seq_array[i]]
    
    return result

def sliding_eval_by_splice_anchor_ultra_optimized(
    model, tokenizer, genome, splice_coords_set_donor, splice_coords_set_acceptor,
    device, mode='donor', region_length=1000, flank=50, kmer=6, sample_time=1, batch_size=32,
    use_amp=True, kmer_to_token=None, inference_batch_size=512
):
    """
    超级优化版本的剪接位点评估函数
    
    主要优化：
    1. 预构建k-mer到token映射，跳过tokenizer
    2. 大量使用numba JIT加速
    3. 向量化操作
    4. 支持AMP混合精度
    5. 优化内存分配和批处理
    """
    model.eval()
    results = []
    window_size = flank * 2 + 1
    current_times = 0
    FP_set, FN_set = [], []
    
    # 构建k-mer到token映射（如果没有提供）
    if kmer_to_token is None:
        kmer_to_token = build_kmer_to_token_mapping(tokenizer, kmer)
    
    # 将Python字典转换为numba字典以在JIT函数中使用
    nb_kmer_to_token = Dict.empty(
        key_type=types.int32,
        value_type=types.int32
    )
    for k, v in kmer_to_token.items():
        nb_kmer_to_token[k] = v
    
    # 预编译坐标集合
    splice_coords_set = splice_coords_set_donor if mode == 'donor' else splice_coords_set_acceptor
    coord_to_type = {}
    
    # 构建快速查找字典
    for chrom, coord, strand in splice_coords_set_donor:
        coord_to_type[(chrom, coord, strand)] = 'donor'
    for chrom, coord, strand in splice_coords_set_acceptor:
        coord_to_type[(chrom, coord, strand)] = 'acceptor'
    
    # 预分配一些常用数组
    base_map = np.array([0, 1, 2, 3], dtype=np.int32)
    
    # 启用AMP
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == 'cuda' else None
    
    with torch.no_grad():
        # 批量处理采样的序列
        sample_coords = random.sample(list(splice_coords_set), k=min(20, len(splice_coords_set)))
        
        for batch_start in range(0, len(sample_coords), batch_size):
            if current_times >= sample_time:
                break
                
            batch_coords = sample_coords[batch_start:batch_start + batch_size]
            
            # 预分配批次数据结构
            batch_sequences_int = []  # 存储整数编码的序列
            batch_metadata = []
            
            # 批量准备序列（避免字符串操作）
            for chrom, center, strand in batch_coords:
                if current_times >= sample_time:
                    break
                    
                chrom_seq = genome[chrom].seq
                region_length_pre = random.randint(0, region_length // 2)
                region_length_post = region_length - region_length_pre
                seq_start = center - region_length_pre - flank
                seq_end = center + region_length_post + flank
                
                if seq_start < 0 or seq_end >= len(chrom_seq):
                    continue
                    
                seq_length = seq_end - seq_start
                seq_str = str(chrom_seq[seq_start:seq_end+1]) # 2025/6/12 5:33 + 1
                
                # 快速转换为整数数组
                seq_array = fast_sequence_to_int_array(seq_str, base_map)
                
                if strand == '-':
                    seq_array = fast_reverse_complement_array(seq_array)
                
                batch_sequences_int.append(seq_array)
                batch_metadata.append({
                    'chrom': chrom, 'center': center, 'strand': strand,
                    'seq_start': seq_start, 'seq_end': seq_end, 'seq_length': seq_length
                })
                current_times += 1
            
            if not batch_sequences_int:
                continue
            
            # 批量生成token IDs（跳过字符串tokenization）
            all_token_ids = []
            all_absolute_coords = []
            batch_boundaries = []
            
            for seq_idx, (seq_array, metadata) in enumerate(zip(batch_sequences_int, batch_metadata)):
                start_boundary = len(all_token_ids)
                
                # 使用优化的k-mer生成
                kmer_ids = fast_kmer_to_ids(seq_array, kmer, window_size, flank)
                
                # 直接转换为token IDs
                token_ids_matrix = fast_kmer_ids_to_token_ids(kmer_ids, nb_kmer_to_token)
                
                # 计算绝对坐标
                absolute_coords = []
                for i in range(flank, metadata['seq_length'] - flank):
                    if metadata['strand'] == '+':
                        coord = metadata['seq_start'] + i
                    else:
                        coord = metadata['seq_end'] - i - 1  # 转换为正链坐标
                    absolute_coords.append(coord)
                
                # 将每行token IDs添加到批次中
                for row_idx in range(token_ids_matrix.shape[0]):
                    all_token_ids.append(token_ids_matrix[row_idx])
                
                all_absolute_coords.extend(absolute_coords)
                batch_boundaries.append((start_boundary, len(all_token_ids)))
            
            if not all_token_ids:
                continue
            
            # 将token IDs转换为PyTorch张量
            max_seq_len = max(len(tokens) for tokens in all_token_ids)
            input_ids = np.zeros((len(all_token_ids), max_seq_len), dtype=np.int32)
            attention_mask = np.zeros((len(all_token_ids), max_seq_len), dtype=np.int32)
            
            for i, tokens in enumerate(all_token_ids):
                seq_len = len(tokens)
                input_ids[i, :seq_len] = tokens
                attention_mask[i, :seq_len] = 1
            
            input_ids_tensor = torch.from_numpy(input_ids).to(device)
            attention_mask_tensor = torch.from_numpy(attention_mask).to(device)
            
            # 分批推理（支持AMP）
            all_preds = []
            
            for i in range(0, len(input_ids_tensor), inference_batch_size):
                batch_input_ids = input_ids_tensor[i:i + inference_batch_size]
                batch_attention_mask = attention_mask_tensor[i:i + inference_batch_size]
                
                if use_amp and device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        logits = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
                else:
                    logits = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
                
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
            
            # 处理每个序列的结果（使用向量化操作）
            for seq_idx, (start_idx, end_idx) in enumerate(batch_boundaries):
                metadata = batch_metadata[seq_idx]
                seq_preds = np.array(all_preds[start_idx:end_idx])
                seq_coords = np.array(all_absolute_coords[start_idx:end_idx])
                
                # 向量化提取预测的剪接位点
                splice_mask = (seq_preds == 1) | (seq_preds == 2)
                pred_splice_coords_array = seq_coords[splice_mask]
                pred_splice_labels = seq_preds[splice_mask]
                
                pred_splice_coords = []
                for i, coord in enumerate(pred_splice_coords_array):
                    pred_splice_coords.append({
                        'coord': int(coord),
                        'label': 'donor' if pred_splice_labels[i] == 1 else 'acceptor'
                    })
                
                # 快速提取真实剪接位点
                true_splice_in_region = []
                chrom, strand = metadata['chrom'], metadata['strand']
                
                for coord in seq_coords:
                    coord_key = (chrom, int(coord), strand)
                    if coord_key in coord_to_type:
                        true_splice_in_region.append({
                            'coord': int(coord),
                            'label': coord_to_type[coord_key]
                        })
                
                # 使用集合快速计算TP, FP, FN
                pred_coords_set = {item['coord'] for item in pred_splice_coords}
                true_coords_set = {item['coord'] for item in true_splice_in_region}
                
                tp_coords = pred_coords_set & true_coords_set
                fp_coords = pred_coords_set - true_coords_set
                fn_coords = true_coords_set - pred_coords_set
                
                TP = len(tp_coords)
                FP = len(fp_coords)
                FN = len(fn_coords)
                TN = len(seq_preds) - TP - FP - FN
                
                # 收集FP和FN样本（优化字符串提取）
                chrom_seq = genome[chrom].seq
                for coord in fp_coords:
                    if coord - flank >= 0 and coord + flank + 1 <= len(chrom_seq):
                        FP_set.append({
                            'seq': str(chrom_seq[coord-flank:coord+flank+1]),
                            'label': 'negative'
                        })
                
                for coord in fn_coords:
                    if coord - flank >= 0 and coord + flank + 1 <= len(chrom_seq):
                        true_label = next(item['label'] for item in true_splice_in_region if item['coord'] == coord)
                        FN_set.append({
                            'seq': str(chrom_seq[coord-flank:coord+flank+1]),
                            'label': true_label
                        })
                
                result_entry = {
                    "chrom": chrom,
                    "strand": strand,
                    "predictions": pred_splice_coords,
                    "true_splice_coord": true_splice_in_region,
                    "metrics": {"TP": TP, "TN": TN, "FP": FP, "FN": FN}
                }
                results.append(result_entry)
    
    # 汇总所有批次的指标
    total_TP = sum(r["metrics"]["TP"] for r in results)
    total_TN = sum(r["metrics"]["TN"] for r in results)
    total_FP = sum(r["metrics"]["FP"] for r in results)
    total_FN = sum(r["metrics"]["FN"] for r in results)
    
    return results, FP_set, FN_set, total_TP, total_TN, total_FP, total_FN

# 额外的工具函数
def create_optimized_model_wrapper(model, device, use_compile=True):
    """
    创建优化的模型包装器
    """
    if use_compile and hasattr(torch, 'compile'):
        # PyTorch 2.0的torch.compile加速
        model = torch.compile(model)
    
    return model

# 使用示例和完整的Cython扩展
def create_advanced_cython_extension():
    """
    创建高级Cython扩展用于更复杂的序列操作
    """
    setup_code = """
from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize([
        "fast_sequence_ops.pyx",
        "splice_site_utils.pyx"
    ], compiler_directives={'language_level': "3"}),
    include_dirs=[numpy.get_include()]
)
"""
    
    sequence_ops_pyx = """
# fast_sequence_ops.pyx
import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport malloc, free

@cython.boundscheck(False)
@cython.wraparound(False)
def ultra_fast_kmer_encode(str sequence, int k):
    cdef int seq_len = len(sequence)
    cdef int num_kmers = max(0, seq_len - k + 1)
    cdef np.ndarray[np.int64_t, ndim=1] result = np.empty(num_kmers, dtype=np.int64)
    cdef int i, j
    cdef long kmer_hash
    cdef char* seq_ptr = <char*>sequence.encode('ascii')
    
    cdef int* base_values = <int*>malloc(256 * sizeof(int))
    # 初始化所有值为0
    for i in range(256):
        base_values[i] = 0
    # 设置ATGC的值
    base_values[65] = 0  # A
    base_values[84] = 1  # T
    base_values[71] = 2  # G
    base_values[67] = 3  # C
    base_values[97] = 0  # a
    base_values[116] = 1 # t
    base_values[103] = 2 # g
    base_values[99] = 3  # c
    
    for i in range(num_kmers):
        kmer_hash = 0
        for j in range(k):
            kmer_hash = kmer_hash * 4 + base_values[<int>seq_ptr[i + j]]
        result[i] = kmer_hash
    
    free(base_values)
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
def batch_kmer_tokenize(list sequences, int k, dict kmer_to_token):
    cdef int n_seqs = len(sequences)
    cdef list result = []
    
    for seq in sequences:
        kmer_codes = ultra_fast_kmer_encode(seq, k)
        token_ids = [kmer_to_token.get(code, 0) for code in kmer_codes]
        result.append(token_ids)
    
    return result
"""
    
    splice_utils_pyx = """
# splice_site_utils.pyx
import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def fast_coordinate_analysis(np.ndarray[np.int64_t, ndim=1] pred_coords,
                           np.ndarray[np.int64_t, ndim=1] true_coords):
    cdef int n_pred = len(pred_coords)
    cdef int n_true = len(true_coords)
    cdef set true_set = set(true_coords)
    
    cdef list tp_indices = []
    cdef list fp_indices = []
    cdef int i
    
    for i in range(n_pred):
        if pred_coords[i] in true_set:
            tp_indices.append(i)
        else:
            fp_indices.append(i)
    
    return np.array(tp_indices), np.array(fp_indices)
"""
    
    return setup_code, sequence_ops_pyx, splice_utils_pyx