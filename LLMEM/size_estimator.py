import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class SizeEstimator(object):

    def __init__(self, model, batch, real_bs, bytes=2, bytes_input=4, 
                 gpu_n=1, tp=0, lm_fp32=True, m_total=0, method='none', peft='none', gradient_checkpointing=True, token_ratio=0.1):
        '''
        Estimates the size of PyTorch models in memory
        for a given input size
        '''
        self.model = model
        self.batch = batch
        self.bytes = bytes
        self.bytes_input = bytes_input
        self.gpu_n = gpu_n
        self.tp = tp
        self.base_size = 512 #1024*1024*2
        self.real_bs = real_bs
        self.lm_fp32 = lm_fp32
        self.m_total = m_total
        self.method = method
        self.peft = peft
        self.gradient_checkpointing = gradient_checkpointing
        self.token_ratio = token_ratio

        self.optimizer = None

        self.cudnn_workspace = 0
        self.activation_bytes = 0


    def get_output_sizes(self):

        self.inout_sizes = []
        self.backward_all_gather_sizes = []

        hooks = []
        def hook_fn(module, inp, out):
            ts = [out] if isinstance(out, torch.Tensor) else list(out)
            for t in ts:
                self.inout_sizes.append(tuple(t.size()))

        for name, m in self.model.named_modules():
            if self.gradient_checkpointing:
                if name.endswith("input_layernorm") or name.endswith("post_attention_layernorm"):
                    hooks.append(m.register_forward_hook(hook_fn))
                elif name.endswith("lm_head"):
                    hooks.append(m.register_forward_hook(hook_fn))

            else:
                if self.peft in ('lora', 'dora'):
                    if hasattr(m, 'base_layer') and isinstance(m.base_layer, nn.Linear):
                        hooks.append(m.base_layer.register_forward_hook(hook_fn))
                    elif isinstance(m, nn.Embedding):
                        hooks.append(m.register_forward_hook(hook_fn))
                else:
                    if isinstance(m, (nn.Embedding, nn.Linear)):
                        hooks.append(m.register_forward_hook(hook_fn))

        self.model.eval()
        torch.cuda.empty_cache(); torch.cuda.synchronize()
        with torch.no_grad():
            if isinstance(self.batch, dict):
                _ = self.model(**self.batch)
            elif isinstance(self.batch, (list, tuple)):
                _ = self.model(*self.batch)
            else:
                _ = self.model(self.batch)

        for h in hooks:
            h.remove()

        self.inout_sizes = [np.array(sz) for sz in self.inout_sizes]


    # def collect_activation_bytes(self):
    #     hooks = []
    #     self.activation_bytes = 0
    #     def hook_fn(module, inp, out):
    #         def count(t):
    #             return t.numel() * self.bytes if isinstance(t, torch.Tensor) else 0
    #         if isinstance(out, torch.Tensor):
    #             self.activation_bytes += count(out)
    #         else:
    #             for x in out:
    #                 self.activation_bytes += count(x)

    #     for m in self.model.modules():
    #         hooks.append(m.register_forward_hook(hook_fn))

    #     self.model.eval()
    #     torch.cuda.empty_cache(); torch.cuda.synchronize()
    #     with torch.no_grad():
    #         _ = self.model(self.batch)

    #     for h in hooks: h.remove()
    #     rem = self.activation_bytes % self.base_size
    #     if rem: self.activation_bytes += (self.base_size - rem)


    def param_bytes(self):
        params = []
        for m in self.model.modules():
            if not isinstance(m, (nn.Embedding, nn.Linear)):
                continue
            for p in m.parameters():
                params.append((np.array(p.size()), p.requires_grad))

        total_bytes = 0
        for i, (shape, requires) in enumerate(params):
            elems = np.prod(shape)
            b_param = elems * self.bytes
            # alignment
            if b_param % self.base_size != 0:
                b_param = (b_param // self.base_size + 1) * self.base_size
            total_bytes += b_param

            if requires:
                total_bytes += b_param

                b_opt = elems * self.bytes * 2  # fp32
                b_opt = b_opt / max(1, self.gpu_n)
                if b_opt % self.base_size != 0:
                    b_opt = (b_opt // self.base_size + 1) * self.base_size
                total_bytes += 2 * b_opt


        self.param_bytes_mem = total_bytes

    # def param_bytes(self):
    #     mods = list(self.model.modules())
    #     param_sizes =[]
    #     for i in range(1, len(mods)):
    #         if not 'Embedding' in mods[i]._get_name():
    #            if not mods[i]._get_name() in ['Linear']: #, 'ReLU', 'LayerNorm']:
    #                 continue
    #         m = mods[i]
    #         p = list(m.parameters())
    #         for j in range(len(p)):
    #             param_sizes.append(np.array(p[j].size()))

    #     '''Calculate total number of bytes to store `model` parameters'''
    #     total_bytes = 0
    #     for i in range(len(param_sizes)):
    #         s = param_sizes[i]
    #         bytes = np.prod(np.array(s))*self.bytes
    #         if bytes % self.base_size != 0:
    #             bytes = int(bytes / self.base_size) * self.base_size + self.base_size
    #         if i == len(param_sizes) - 1:
    #             # break
    #             total_bytes += bytes# / self.gpu_n
    #         else:
    #             # Chunk-based mixed-precision model parameter memory
    #             # 1. Parameters/gradients (fp16)
    #             if not self.tp:
    #                 if self.gpu_n > 1:  # ZeRO-3 Optimizer
    #                     # total_bytes += bytes
    #                     bytes = bytes * (self.gpu_n - 1) / self.gpu_n
    #                     if bytes % self.base_size != 0:
    #                         bytes = int(bytes / self.base_size) * self.base_size + self.base_size
    #                     total_bytes += bytes
    #             elif self.tp and self.gpu_n > 1 and self.tp != self.gpu_n: # DP + TP (Hybrid parallelism)
    #                 bytes = bytes * ((self.gpu_n - self.tp) / self.gpu_n - 1 / self.gpu_n)
    #                 if bytes % self.base_size != 0:
    #                     bytes = int(bytes / self.base_size) * self.base_size + self.base_size
    #                 total_bytes += bytes
    #             # 2. optimizer parameters (fp32)
    #             bytes = np.prod(np.array(s))*self.bytes*2
    #             bytes = bytes / self.gpu_n
    #             if bytes % self.base_size != 0:
    #                 bytes = int(bytes / self.base_size) * self.base_size + self.base_size
    #             ##### Real-size-based optimizer states memory
    #             # 3. gradient momentums (fp32), gradient variances (fp32)
    #             total_bytes += 2*bytes
    #             ##########################
    #     self.param_bytes_mem = total_bytes


    def calc_input_bytes(self):
        '''Calculate bytes to store input'''
        # self.input_bytes = np.prod(np.array(self.input_size))*self.bytes
        self.input_bytes = np.prod(self.inout_sizes[0])*self.bytes_input
        if self.input_bytes % self.base_size != 0:
            self.input_bytes = int(self.input_bytes / self.base_size) * self.base_size + self.base_size


    def calc_output_bytes(self):
        '''Calculate bytes to store forward and backward pass'''
        total_bytes = 0
        total_backward_all_gather_bytes = 0
        for i in range(0, len(self.inout_sizes)):
            self.inout_sizes[i][0] = self.real_bs
        for i in range(1, len(self.inout_sizes)-1):
            s = self.inout_sizes[i]
            bytes = np.prod(np.array(s))*self.bytes
            if bytes % self.base_size != 0:
                bytes = int(bytes / self.base_size) * self.base_size + self.base_size
            if self.method == "tokentune":
                total_bytes += bytes * self.token_ratio
            else:
                total_bytes += bytes

        # lm_head and loss function
        last_part = 0
        s = self.inout_sizes[-1]
        if self.lm_fp32:
            bytes = np.prod(np.array(s))*self.bytes*2
        else:
            bytes = np.prod(np.array(s))*self.bytes
        if bytes % self.base_size != 0:
            bytes = int(bytes / self.base_size) * self.base_size + self.base_size
        last_part += bytes
        s[1] -= 1
        # temporary_mem = 0
        if self.lm_fp32:
            bytes = np.prod(np.array(s))*self.bytes*2
            # temporary_mem = bytes / 2
        else:
            bytes = np.prod(np.array(s))*self.bytes
        if bytes % self.base_size != 0:
            bytes = int(bytes / self.base_size) * self.base_size + self.base_size
        # if temporary_mem % self.base_size != 0:
        #     temporary_mem = int(temporary_mem / self.base_size) * self.base_size + self.base_size
        last_part += bytes*2 # + temporary_mem

        if total_backward_all_gather_bytes > 0:
            last_part += total_backward_all_gather_bytes

        self.inout_bytes = total_bytes + last_part


    def bs_search(self):
        '''Find the maximum batch size'''
        cur_bs = 1
        last_total_mem = 0
        while True:
            total_bytes = 0
            total_backward_all_gather_bytes = 0
            total_mem = self.m_init
            for i in range(0, len(self.inout_sizes)):
                self.inout_sizes[i][0] = cur_bs
            for i in range(1, len(self.inout_sizes)-1):
                s = self.inout_sizes[i]
                bytes = np.prod(np.array(s))*self.bytes
                if bytes % self.base_size != 0:
                    bytes = int(bytes / self.base_size) * self.base_size + self.base_size
                total_bytes += bytes

            if self.tp:
                for i in range(0, len(self.backward_all_gather_sizes)):
                    self.backward_all_gather_sizes[i][0] = cur_bs
                for i in range(0, len(self.backward_all_gather_sizes)-1):
                    s = self.backward_all_gather_sizes[i]
                    bytes = np.prod(np.array(s))*self.bytes * (self.tp - 1) / self.tp
                    if bytes % self.base_size != 0:
                        bytes = int(bytes / self.base_size) * self.base_size + self.base_size
                    total_backward_all_gather_bytes += bytes

            # lm_head and loss function
            last_part = 0
            s = self.inout_sizes[-1]
            if self.lm_fp32:
                bytes = np.prod(np.array(s))*self.bytes*2
            else:
                bytes = np.prod(np.array(s))*self.bytes
            if bytes % self.base_size != 0:
                bytes = int(bytes / self.base_size) * self.base_size + self.base_size
            last_part += bytes
            s[1] -= 1
            # temporary_mem = 0
            if self.lm_fp32:
                bytes = np.prod(np.array(s))*self.bytes*2
                # temporary_mem = bytes / 2
            else:
                bytes = np.prod(np.array(s))*self.bytes
            if bytes % self.base_size != 0:
                bytes = int(bytes / self.base_size) * self.base_size + self.base_size
            # if temporary_mem % self.base_size != 0:
            #     temporary_mem = int(temporary_mem / self.base_size) * self.base_size + self.base_size
            last_part += bytes*2 #+ temporary_mem

            # if total_backward_all_gather_bytes > bytes:
            #     last_part += total_backward_all_gather_bytes - bytes
            if total_backward_all_gather_bytes > 0:
                last_part += total_backward_all_gather_bytes

            total_mem += (self.param_bytes_mem + self.input_bytes + total_bytes + last_part) / (1024**2)

            # ####################################################
            # import torch.distributed as dist
            # if dist.get_rank() == 0:
            #     with open('temp_size.txt', 'a') as f:
            #         # f.write('{}: {}\n'.format(cur_bs, total_mem))
            #         f.write('{}\n'.format(total_mem))
            # ####################################################
                    
            if total_mem > self.m_total:
                self.real_bs = cur_bs - 1
                if last_total_mem == 0:
                    return total_mem
                return last_total_mem
            # if cur_bs > 80:
            #     return last_total_mem
            
            cur_bs += 1
            last_total_mem = total_mem
            s[1] += 1

    def estimate_size(self, m_init=0):
        '''Estimate model size in memory in megabytes and bytes'''
        self.m_init = m_init
        self.param_bytes()
        self.get_output_sizes()

        self.calc_input_bytes()
        self.calc_output_bytes()

        if self.real_bs > 0:
            print(self.param_bytes_mem/(1024**2), self.inout_bytes/(1024**2), self.input_bytes/(1024**2), self.m_init)
            total = (self.param_bytes_mem 
                   + self.inout_bytes 
                   + self.input_bytes )            
            total_mb = total/(1024**2) + self.m_init
            return total_mb, self.real_bs
        else:
            total_mb = self.bs_search()
            return total_mb, self.real_bs