from functools import partial

import torch
import torch.nn as nn

ONE_INITIALIZER = torch.nn.init.ones_
NORMAL_INITIALIZER = partial(torch.nn.init.normal_, mean=0, std=0.1)


class EntityNetwork(nn.Module):
    def __init__(
            self, 
            embeddings_size,
            vocab_size,
            answers_vocab_size,
            sentences_length,
            queries_length,
            n_blocks,
            output_module,
            output_inner_size,
            temporal_attention_to_sentence,
            temporal_activation,
            device,
            dropout_prob=0
        ):
        super().__init__()
        # Sanity check
        assert output_module in ("joint", "parallel")
        self.temporal_attention_to_sentence = temporal_attention_to_sentence
        self.embeddings_table = nn.Embedding( # [vocab_sz, embeddings_sz]
            num_embeddings=vocab_size + n_blocks,
            embedding_dim=embeddings_size,
            padding_idx=0)
        self.keys = torch.arange(vocab_size, vocab_size + n_blocks).to(device)
        self.prelu = _PReLU(
            num_parameters=embeddings_size,
            device=device)
        self.input_module = _InputModule(
            embeddings_table=self.embeddings_table,
            sentences_length=sentences_length,
            embeddings_size=embeddings_size,
            dropout_prob=dropout_prob)
        self.dynamic_memory = _DynamicMemory(
            embeddings_table=self.embeddings_table,
            block_size=embeddings_size,
            embeddings_size=embeddings_size,
            n_blocks=n_blocks,
            prelu=self.prelu,
            device=device)
        output_kwargs = {
            "embeddings_table": self.embeddings_table,
            "embeddings_size": embeddings_size,
            "inner_size": output_inner_size,
            "answers_vocab_size": answers_vocab_size,
            "queries_length": queries_length,
            "temporal_attention_to_sentence": temporal_attention_to_sentence,
            "temporal_activation": temporal_activation,
            "device": device,
            "prelu": self.prelu
        }
        if output_module == "joint":
            self.output_module = _JointOutputModule(**output_kwargs)
        elif output_module == "parallel":
            self.output_module = _ParallelOutputModule(**output_kwargs)

    def forward(self, stories, stories_mask, queries, supporting_facts=None):
        """
        Args:
            - stories: a 3-D tensor with shape [batch_sz, stories_len, sentence_len].
            - stories_mask: a 2-D tensor with shape [batch_sz, stories_len].
            - queries: a 2-D tensor with shape [batch_sz, queries_len].
            - supporting_facts: a 2-D tensor with shape [batch_sz, stories_len].
        Returns:
            - answers: a 2-D tensor with shape [batch_sz, answers_vocab_size].
            - attended: a 2-D tensor with shape [batch_sz, stories_len].
        """
        stories_reduced = self.input_module(stories)
        memories = self.dynamic_memory(
            stories_reduced,
            stories_mask,
            self.keys)
        answers, alignment, attention = self.output_module(
            keys=self.keys,
            memories=memories,
            queries=queries,
            stories_mask=stories_mask,
            stories=stories_reduced if self.temporal_attention_to_sentence else None,
            supporting_facts=supporting_facts)
        return answers, alignment, attention


class _PReLU(nn.Module):
    def __init__(self, num_parameters, device, initializer=ONE_INITIALIZER):
        super().__init__()
        self.weights = nn.Parameter(torch.Tensor(num_parameters))
        initializer(self.weights)
        self.zeros = torch.zeros(num_parameters).to(device)
    
    def forward(self, x):
        pos = torch.max(x, self.zeros)
        neg = self.weights * torch.min(x, self.zeros)
        return pos + neg


class _InputModule(nn.Module):
    def __init__(self, embeddings_table, sentences_length, embeddings_size, dropout_prob, initializer=ONE_INITIALIZER):
        super().__init__()
        self.embeddings_table = embeddings_table # [vocab_sz, embeddings_sz]
        self.multiplicative_mask = nn.Parameter( # [stories_len, embeddings_sz]
            torch.Tensor(sentences_length, embeddings_size))
        initializer(self.multiplicative_mask)
        self.dropout = nn.Dropout(p=dropout_prob)
    
    def forward(self, stories):
        """
        Args:
            - stories: a 3-D tensor with shape [batch_sz, stories_len, sentence_len].
        Returns:
            - stories_reduced: a 3-D tensor with shape [batch_sz, stories_len, embeddings_sz].
        """
        stories_embedded = self.embeddings_table(stories) # [batch_sz, stories_len, sentence_len, embeddings_sz]
        stories_dropout = self.dropout(stories_embedded)
        stories_masked = torch.mul( # [batch_sz, stories_len, sentence_len, embeddings_sz]
            stories_dropout, self.multiplicative_mask)
        stories_reduced = torch.sum( # [batch_sz, stories_len, embeddings_sz]
            stories_masked, dim=2)
        return stories_reduced


class _DynamicMemory(nn.Module):
    def __init__(self, embeddings_table, block_size, embeddings_size, n_blocks, prelu, device, initializer=NORMAL_INITIALIZER):
        super().__init__()
        self.embeddings_table = embeddings_table
        self.block_size = block_size
        self.embeddings_size = embeddings_size
        self.n_blocks = n_blocks
        self.prelu = prelu
        self.device = device
        self.initializer = initializer

        self.g_bias = nn.Parameter(torch.Tensor(n_blocks))
        initializer(self.g_bias)
        self.U = nn.Parameter(torch.Tensor(block_size, block_size))
        initializer(self.U)
        self.U_bias = nn.Parameter(torch.Tensor(block_size))
        initializer(self.U_bias)
        self.V = nn.Parameter(torch.Tensor(block_size, block_size))
        initializer(self.V)
        self.W = nn.Parameter(torch.Tensor(embeddings_size, block_size))
        initializer(self.W)
    
    def forward(self, stories, stories_mask, keys):
        """
        Args:
            - stories: a 3-D tensor with shape [batch_sz, stories_len, embeddings_sz].
            - stories_mask: a 2-D tensor with shape [batch_sz, stories_len].
            - keys: a 1-D tensor with shape [n_blocks].
        Returns:
            - memories: a 4-D tensor with shape [batch_sz, story_len, n_blocks, block_sz].
        """
        batch_size, stories_length, _ = stories.size()
        keys_embedded = self.embeddings_table(keys) # [n_blocks, embeddings_sz]
        hidden = keys_embedded.clone().repeat((batch_size, 1, 1)).detach().to(self.device)
        memories = []
        for i in range(stories_length):
            hidden = self.__step(
                hidden,
                stories[:, i, :],
                stories_mask[:, i],
                keys_embedded)
            memories.append(hidden)
        memories = torch.stack(memories, dim=1) # [batch_sz, story_len, n_blocks, block_sz]
        return memories

    
    def __step(self, prev_h, facts, mask, keys):
        """
        Args:
            - prev_h: a 3-D tensor with shape [batch_sz, n_blocks, block_sz].
            - facts: a 2-D tensor with shape [batch_sz, embeddings_sz].
            - mask: a 1-D tensor with shape [batch_sz].
            - keys: a 2-D tensor with shape [n_blocks, embeddings_sz].
        Returns:
            - h: a 3-D tensor with shape [batch_sz, n_blocks, block_sz].
        """
        facts_unsqueezed = torch.unsqueeze( # [batch_sz, 1, embeddings_sz]
            facts, dim=1)
        mask_unsqueezed = torch.unsqueeze( # [batch_sz, 1]
            mask, dim=1)
        g_h = torch.sum( # [batch_sz, n_blocks]
            prev_h * facts_unsqueezed, dim=2)
        g_w = torch.sum( # [batch_sz, n_blocks]
            keys * facts_unsqueezed, dim=2)
        g = torch.sigmoid(g_h + g_w + self.g_bias) # [batch_sz, n_blocks]
        g_masked = torch.mul(g, mask_unsqueezed) # [batch_sz, n_blocks]
        g_unsqueezed = torch.unsqueeze( # [batch_sz, n_blocks, 1]
            g_masked, dim=2)
        
        h_hat_U = torch.tensordot( # [batch_sz, n_blocks, block_sz]
            prev_h, self.U, dims=[[2], [0]])
        h_hat_U = h_hat_U + self.U_bias # [batch_sz, n_blocks, block_sz]
        h_hat_V = torch.matmul( # [n_blocks, block_sz]
            keys, self.V)
        h_hat_V = torch.unsqueeze( # [1, n_blocks, block_sz]
            h_hat_V, dim=0)
        h_hat_W = torch.matmul( # [batch_sz, block_sz]
            facts, self.W)
        h_hat_W = torch.unsqueeze( # [batch_sz, 1, block_sz]
            h_hat_W, dim=1)
        h_hat = self.prelu(h_hat_U + h_hat_V + h_hat_W) # [batch_sz, n_blocks, block_sz]

        new_h = prev_h + (h_hat * g_unsqueezed)
        new_h = nn.functional.normalize(new_h, dim=2)

        return new_h


class _OutputModule(nn.Module):
    def __init__(
            self,
            embeddings_table,
            embeddings_size,
            inner_size,
            answers_vocab_size,
            queries_length,
            temporal_attention_to_sentence,
            temporal_activation,
            device,
            prelu,
            initializer=NORMAL_INITIALIZER,
            mask_initializer=ONE_INITIALIZER
        ):
        super().__init__()
        self.embeddings_table = embeddings_table
        self.embeddings_size = embeddings_size
        self.answers_vocab_size = answers_vocab_size
        self.queries_length = queries_length
        self.temporal_attention_to_sentence = temporal_attention_to_sentence
        assert temporal_activation in ("sigmoid", "softmax")
        self.temporal_activation = temporal_activation
        self.prelu = prelu

        self.multiplicative_mask = nn.Parameter(torch.Tensor(queries_length, embeddings_size))
        mask_initializer(self.multiplicative_mask)
        self.A = nn.Parameter(torch.Tensor(embeddings_size, inner_size))
        initializer(self.A)
        self.B = nn.Parameter(torch.Tensor(embeddings_size, inner_size))
        initializer(self.B)
        self.C = nn.Parameter(torch.Tensor(embeddings_size, inner_size))
        initializer(self.C)
        self.v = nn.Parameter(torch.Tensor(inner_size))
        mask_initializer(self.v) # Initialize it with ones!
        # self.b_1 = nn.Parameter(torch.Tensor(20))
        # initializer(self.b_1)
        self.D = nn.Parameter(torch.Tensor(embeddings_size, inner_size))
        initializer(self.D)
        if temporal_attention_to_sentence:
            self.E = nn.Parameter(torch.Tensor(embeddings_size, inner_size))
            initializer(self.E)
        self.F = nn.Parameter(torch.Tensor(embeddings_size, inner_size))
        initializer(self.F)
        self.w = nn.Parameter(torch.Tensor(inner_size))
        mask_initializer(self.w)
        # self.b_2 = nn.Parameter(torch.Tensor(10))
        # initializer(self.b_2)
        self.H = nn.Parameter(torch.Tensor(embeddings_size, embeddings_size))
        initializer(self.H)
        self.R = nn.Parameter(torch.Tensor(embeddings_size, answers_vocab_size))
        initializer(self.R)
        self.minus_inf = torch.Tensor([float("-inf")]).to(device)

    def _reduce_query(self, queries):
        queries_embedded = self.embeddings_table(queries) # [batch_sz, queries_len, embeddings_sz]
        queries_masked = torch.mul( # [batch_sz, queries_len, embeddings_sz]
            queries_embedded, self.multiplicative_mask)
        queries_reduced = torch.sum( # [batch_sz, embeddings_sz]
            queries_masked, dim=1)
        return queries_reduced

    def _get_temporal_attention(self, memories, keys, queries, stories_mask, stories=None):
        """
        Args:
            - memories: a tensor with shape [batch_sz, story_len, n_blocks, block_sz].
            - keys: a tensor with shape [n_blocks].
            - queries: a tensor with shape [batch_sz, embeddings_sz].
            - stories_mask: a tensor with shape [batch_sz, storieS_len].
            - stories: a tensor with shape [batch_sz, story_len, embeddings_sz].
        """
        # Intratemporal attention
        intratemporal_alignment_A = torch.matmul( # [batch_sz, story_len, n_blocks, inner_sz]
            memories, self.A)
        # batch_size, story_length, _, _ = intratemporal_alignment_A.size()
        keys_embedded = self.embeddings_table(keys)
        intratemporal_alignment_B = torch.matmul( # [n_blocks, inner_sz] -> [1, 1, n_blocks, inner_sz]
            keys_embedded, self.B).unsqueeze(0).unsqueeze(0)
        intratemporal_alignment_C = torch.matmul( # [batch_sz, inner_sz] -> [batch_sz, 1, 1, inner_sz]
            queries, self.C).unsqueeze(1).unsqueeze(1)
        intratemporal_alignment = torch.sum( # [batch_sz, story_len, n_blocks]
            self.v * torch.tanh(intratemporal_alignment_A + intratemporal_alignment_B + intratemporal_alignment_C),
            dim=3)
        intratemporal_attention = nn.functional.softmax( # [batch_sz, story_len, n_blocks] -> [batch_sz, story_len, n_blocks, 1]
            intratemporal_alignment, dim=2).unsqueeze(3)

        # Temporal memory
        temporal_memory = torch.sum( # [batch_sz, story_len, embeddings_sz]
            intratemporal_attention * memories, dim=2)

        # Temporal attention
        temporal_alignment_D = torch.matmul( # [batch_sz, story_len, inner_sz]
            temporal_memory, self.D)
        temporal_alignment_E = torch.matmul(  # [batch_sz, story_len, inner_sz]
            stories, self.E) if self.temporal_attention_to_sentence else 0
        temporal_alignment_F = torch.matmul( # [batch_sz, inner_sz] -> [batch_sz, 1, inner_sz]
            queries, self.F).unsqueeze(1)
        temporal_alignment_unmasked = torch.sum( # [batch_sz, story_len]
            self.w * torch.tanh(temporal_alignment_D + temporal_alignment_E + temporal_alignment_F),
            dim=2)
        temporal_alignment = torch.mul(temporal_alignment_unmasked, stories_mask)

        if self.temporal_activation == "sigmoid":
            temporal_attention_unmasked = torch.sigmoid(temporal_alignment)
            temporal_attention = torch.mul(temporal_attention_unmasked, stories_mask)
        elif self.temporal_activation == "softmax":
            flipped_mask = ((stories_mask * -1) + 1) # [1, 1, 0] -> [0, 0, 1]
            inf_mask = flipped_mask * self.minus_inf # [nan, nan, -inf]
            inf_mask[inf_mask != inf_mask] = 0 # [0, 0, -inf]
            temporal_inf_masked = temporal_alignment + inf_mask
            temporal_attention = torch.softmax(temporal_inf_masked, dim=1)

        return temporal_attention, temporal_alignment, temporal_memory


class _ParallelOutputModule(_OutputModule):
    def forward(self, keys, memories, queries, stories_mask, stories=None, **kwargs):
        """
        Args:
            - keys: a tensor with shape [n_blocks].
            - memories: a tensor with shape [batch_sz, story_len, n_blocks, block_sz].
            - queries: a tensor with shape [batch_sz, queries_len].
            - stories_mask: a tensor with shape [batch_sz, stories_len].
            - stories: a tensor with shape [batch_sz, story_len, embeddings_sz].
        Returns:
            - answers: a tensor with shape [batch_sz, answers_vocab_sz].
            - temporal_alignment: a tensor with shape [batch_sz, story_len].
            - temporal_attention: a tensor with shape [batch_sz, story_len].
        """
        # Embed query
        queries_reduced = self._reduce_query(queries) # [batch_sz, embeddings_sz]
        queries_unsqueezed = torch.unsqueeze( # [batch_sz, 1, embeddings_sz]
            queries_reduced, dim=1)

        # Temporal attention
        temporal_attention, temporal_alignment, _ = self._get_temporal_attention(
            memories,
            keys,
            queries_reduced,
            stories_mask,
            stories)

        memories = memories[:, -1, :, :]
        alignment = torch.sum( # [batch_sz, n_blocks]
            queries_unsqueezed * memories, dim=2)
        p = nn.functional.softmax( # [batch_sz, n_blocks]
            alignment, dim=1)
        p_unsqueezed = torch.unsqueeze( # [batch_sz, n_blocks, 1]
            p, dim=2)
        u = torch.sum( # [batch_sz, embeddings_sz]
            p_unsqueezed * memories, dim=1)

        inner_y = self.prelu( # [batch_sz, embeddings_sz]
            queries_reduced + torch.matmul(u, self.H))
        answers = torch.matmul(inner_y, self.R)

        return answers, temporal_alignment, temporal_attention


class _JointOutputModule(_OutputModule):
    def forward(self, keys, memories, queries, stories_mask, stories=None, supporting_facts=None):
        """
        Args:
            - keys: a tensor with shape [n_blocks].
            - memories: a tensor with shape [batch_sz, story_len, n_blocks, block_sz].
            - queries: a tensor with shape [batch_sz, queries_len].
            - stories: a tensor with shape [batch_sz, story_len, embeddings_sz].
            - supporting_facts: a tensor with shape [batch_sz, story_len].
        Returns:
            - answers: a tensor with shape [batch_sz, answers_vocab_sz].
            - temporal_alignment: a tensor with shape [batch_sz, story_len].
            - temporal_attention: a tensor with shape [batch_sz, story_len].
        """
        # Embed query
        queries_reduced = self._reduce_query(queries) # [batch_sz, embeddings_sz]

        # Temporal attention
        temporal_attention, temporal_alignment, temporal_memory = self._get_temporal_attention(
            memories,
            keys,
            queries_reduced,
            stories_mask,
            stories)

        if supporting_facts is not None:
            # Teacher forcing
            supporting_facts_unsqueezed = torch.unsqueeze( # [batch_sz, story_len, 1]
                supporting_facts, dim=2)
            u = torch.sum( # [batch_sz, embeddings_sz]
                supporting_facts_unsqueezed * temporal_memory, dim=1)
        else:
            temporal_attention_unsqueezed = torch.unsqueeze( # [batch_sz, story_len, 1]
                temporal_attention, dim=2)
            u = torch.sum( # [batch_sz, embeddings_sz]
                temporal_attention_unsqueezed * temporal_memory, dim=1)

        inner_y = self.prelu( # [batch_sz, embeddings_sz]
            queries_reduced + torch.matmul(u, self.H))
        answers = torch.matmul(inner_y, self.R)
        return answers, temporal_alignment, temporal_attention
