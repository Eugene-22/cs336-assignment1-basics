# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```
---


## 一、实验概述
### 实验内容
- Section 2：BPE (Byte-pair encoding) 分词器；
- Section 3：基于 Transformer 架构的语言模型；
- Section 4：交叉熵损失函数和 AdamW 优化器；
- Section 5：实现完整的训练循环，并支持模型与优化器状态的保存/加载功能。

### 实验任务
- 在 TinyStories 数据集上实现并训练一个 BPE 分词器；
- 在 TinyStories 数据集上训练一个基于 Transformer 架构的语言模型；
- 使用训练好的模型生成样本并评估困惑度 (Perplexity)；
- 在 OpenWebText 数据集上训练模型，并将达到的困惑度结果提交到排行榜。

## 二、BPE 分词器
### 2.1 unicode 标准
Unicode标准将字符映射到整数code points上。例如"s" 的code point 是115（或写作十六进制的U+0073），“牛”对应29275。
### 2.2 unicode 编码
根据 Unicode 标准定义了三种编码方式：UTF-8, UTF-16, UTF-32。其中，UTF-8 最为常用。在 Python 中可以使用 encode() 和 decode() 实现字符和 UTF-8 之间的编码和解码。
### 2.3 子词分词方案
- 单词级划分：序列短，处理快。词表必须包含所有可能出现的单词。遇到训练时没见过的词，就是 OOV，无法处理。
- 字节级分词：把文本按 UTF-8 拆成单个字节。任何文本都能表示为 0~255 的序列，不会出现 OOV。序列越长，模型每一步的计算量越大，模型更难捕捉远端的依赖关系。
- 子词分词：取上述两者的“中间点”，训练得到更大的词汇表，来换取对输入序列的更好压缩。

#### BPE 算法
既有字节级分词消灭 OOV 的能力，又有子词分词对序列长度的有效压缩。

如何选择这些子词单元子词单元？
- 基础词汇表：256 个单字节（0~255），保证任何文本都能被编码。
- 扩展词汇表：通过 BPE 训练，从语料中学习常见的字节序列合并，压缩序列长度。

### 2.4 训练 BPE 分词器
#### 第一步：初始化词汇表
从字节串（bytes）到整数 ID（int）的映射表。

除了基础的 256 个字节 token，如果用户提供了特殊 token（如 <|endoftext|>），也需要在初始化阶段加入词汇表。

特殊 token：
- 不是通过 BPE 合并得到的，而是手动指定的；
- 会被赋予新的、唯一的 ID（256、257……依次往后）；
- 在训练和编码时会被特殊保护，不会被 BPE 拆分。

所以初始化完成后的词汇表大小为：256（基础单字节） + len(special_tokens)

#### 第二步：预分词
词汇表初始化完成后，目前有：
- 一个包含 256 个单字节 token 的词汇表（加上可能有的特殊 token）；
- 大量原始文本语料（比如几百万个英文句子、文档）。

目标：找出语料中最常相邻出现的字节对，以便把最高频的合并成新 token。

问题：
- 效率极低，语料有几个GB，每做一次合并（整个训练过程可能要合并几千次）就得重新扫描整个语料，更新 token 序列，再统计新的相邻对频率；
- 产生无意义的跨单词合并，文本中 "dog!" 和 "dog."，如果直接**统计字节对**，"dog" 末尾的 g 可能会和 ! 或 . 合并，产生 g! 和 g. 这样两个不同的合并 token。

预分词就是在 BPE 统计之前，先把文本粗切成许多小块（pre-token），然后只在每个小块内部统计和合并。

- 锁定语义边界，确保合并只在单词/标点内部发生，不跨边界。
- 统计效率大幅提高，我们可以先统计每个预分词单元在语料中出现了多少次，然后在每个单元内部独立统计字节对频率，再乘以该单元出现的次数。这样我们只需要一次遍历语料来收集所有预分词单元及其计数，之后的合并迭代完全不需要重新读语料。

如何划分得到预分词单元？

基于正则表达式切分（GPT-2 及之后的现代做法）：

```python
>>> PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

在 BPE 训练中，通过一个**字典数据结构**来存储**预分词单元**及其**频率**：

键是预分词单元的字节串

值是一个对象，包括：
- count：该预分词单元在语料中出现的次数
- tokens：当前该单元的 token ID 列表（初始是各字节的 ID，如 [116, 104, 101]）

BPE 流程：
1. 遍历整个语料，用正则预分词，统计每个预分词单元的出现次数。

2. 对每个预分词单元，转成 UTF-8 字节序列，统计其内部的字节对频率，乘以该单元的出现次数，汇总全局频率。

3. 找出频率最高的字节对，合并成一个新 token，更新词汇表。

4. 在所有预分词单元内部，把该字节对的出现替换为新 token，更新计数（只需在受影响的单元内局部更新，不需重新扫描语料）。

5. 重复步骤 3-4，直到词汇表达到目标大小。

#### 第三步：计算 BPE 合并
现在，我们已经有了：
- 一个由 256 个单字节 token 组成的初始词汇表（加上特殊 token）；
- 一个预分词单元 → (出现次数, token 序列) 的字典结构。

合并流程：
1. 统计所有相邻 token 对的全局频率；
2. 找到全局出现次数最多的那个 pair，如果最高频的 pair 有多个，按字典序最大的那个 pair 进行合并

字典序最大规则：
- 字典序比较是先比第一个元素，如果相同再比第二个。

训练循环结束后，我们拿到：
- vocab：{token_id: bytes}，包含原始 256 个单字节和所有合并产生的新 token。
- merges：[(bytes_A, bytes_B), ...]，按合并顺序排列，越靠前优先级越高（越早合并的 pair 在编码时优先合并）。

#### 特殊 token
特殊 token 是一些人为指定的、有特殊含义的字符串。

必须作为一个整体存在，不能被 BPE 拆分成更小的 token；

必须预先加入词汇表，拥有固定的 token ID，这样编码和解码时可以直接按完整符号处理。

在 BPE 训练时，它们被识别并保护起来，不会被合并；在编码时，它们被优先“挖出来”，对它们不做 BPE 合并。

### 2.5 实验实现 BPE 分词器训练
通过 adapters.run_train_bpe 接口调用，在实现 BPE 训练实现必须做优化。

输入：
- input_path: str — 训练语料的文件路径；
- vocab_size: int — 最终词汇表的最大大小（包括 256 个基础字节、合并产生的新 token、以及特殊 token）；
- special_tokens: list[str] — 特殊 token 列表（如 ["<|endoftext|>"]）。

输出：
- vocab: dict[int, bytes] — token ID 到字节串的映射；
- merges: list[tuple[bytes, bytes]] — 按合并顺序排列的合并规则。

#### 优化一：并行化预分词
优化方法：
- 使用 Python 的 multiprocessing 库并行处理。
- 先把文件按文档边界切分成若干块（chunk），每个进程处理一块。
- 切分时使用课程提供的 find_chunk_boundaries 函数（它保证切分点恰好落在 <|endoftext|> 这样的特殊 token 之后，不会把特殊 token 切成两半）。

#### 优化二：预分词前先移除特殊 token
#### 优化三：优化合并步骤

实验任务：
- 实现 train_bpe；
- 实现完整的 BPE 训练流程，通过 test_train_bpe.py 的 3 个测试。

测试要求：
- 速度测试：在 corpus.en 上训练 500 大小的词汇表，时间不超过 1.5 秒。
- 正确性测试：训练出的 merges 必须与参考合并列表完全一致，vocab 的键和值集合与参考词汇表一致。
- 特殊 token 测试：特殊 token 不被拆分，不参与合并。