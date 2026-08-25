import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import quant_cuda

class QuantBase():
    def _quantization(x, quant_grid, normal_size=-1):
        shape = x.shape
        quant_array = x.reshape(-1)
        quant_grid = quant_grid.type_as(quant_array)
        quant_array, _ = quant_cuda.quant(quant_array, quant_grid, normal_size)
        quant_array = quant_array.view(shape)
        return quant_array
    @staticmethod
    def forward(real_val, quant_grid, normal_size=-1):
        with torch.no_grad():
            dequantized_val = QuantBase._quantization(real_val, quant_grid, normal_size)
            return dequantized_val

def pytorch_quantize(data, quant_grid):
    """
    純 PyTorch 版本的 quantization。
    比 CUDA kernel 慢，但支援任意大小的 codebook。
    """
    original_shape = data.shape
    flat = data.reshape(-1)                                    # (N,)
    quant_grid = quant_grid.type_as(flat)                      # 統一 dtype
    
    # 為了節省記憶體，分批處理
    batch_size = 10000  # 每次處理 10000 個 elements
    result = torch.empty_like(flat)
    
    for i in range(0, flat.shape[0], batch_size):
        chunk = flat[i:i+batch_size]                           # (batch,)
        diff = (chunk.unsqueeze(-1) - quant_grid).abs()        # (batch, K)
        idx = diff.argmin(dim=-1)                              # (batch,)
        result[i:i+batch_size] = quant_grid[idx]
    
    return result.view(original_shape)


class Quantizer(nn.Module):
    def __init__(self, mode="base", bit=8, is_signed=True, is_enable=False, is_input=False, args=None, operator=None, group_size=2):
        super(Quantizer, self).__init__()
        self.mode = mode
        self.is_input = is_input
        self.is_signed = is_signed
        self.is_enable = is_enable
        self.is_enable_activation = is_enable
        self.is_enable_weight = is_enable
        self.args = args
        self.operator = operator
        self.group_size = group_size
        
        self.w_up = self.args.w_up
        self.a_up = self.args.a_up
        self.w_low = self.args.w_low
        self.a_low = self.args.a_low

        self.alpha = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.register_buffer('bit', torch.tensor(bit))
        self.register_buffer('has_inited_quant_para', torch.tensor(0.0))
        self.register_buffer('quant_grid', torch.ones(2**bit))
        self.register_buffer('outliers', torch.ones(2**bit))
        self.percent = self.args.percent / 100
        self.is_perchannel = True
        if is_input:
            # Input shouldn't be per-channel quantizaton！
            self.is_perchannel = False
        self.search = args.search
        self.mse = torch.tensor(0.0)

        ## debug
        self.name = None

    def disable_input_quantization(self):
        self.is_enable_activation = False
        
    def enable_quantization(self, name):
        self.name = name
        self.is_enable = True

    def disable_quantization(self, name):
        self.name = name
        self.is_enable = False

    def update_signed(self, tensor):
        if tensor.min() < 0:
            self.is_signed = True

    @torch.no_grad()
    def int_value(self):
        bit_width = self.bit.item()
        B = bit_width
        if self.is_signed:
            B = bit_width - 1

        values = []
        values.append(0.)
        for i in range(1, 2 ** B):
            values.append(i)
            if self.is_signed:
                values.append(-i)
                
        values = torch.tensor(values, device=self.quant_grid.device) 
        values, _ = torch.sort(values)
        # add a bias to normalize the codebook (the threshold between outliers and normal values is 32)
        values *= 32 / (2 ** B)

        return values

    @torch.no_grad()
    def flint_value(self,  exp_base = 0):
        B = self.bit.item()
        if self.is_signed:
            B = B - 1
        value_bit = B
        if value_bit < 2:
            # 2-bit signed 做不了 flint，fallback 用 int
            return self.int_value()

        exp_num =     value_bit * 2 - 1
        neg_exp_num = value_bit - 1
        pos_exp_num = value_bit - 1
        
        exp_max = pos_exp_num + exp_base
        exp_min = -neg_exp_num

        ## zero
        values = [0.]

        ## exponent negtive
        for i in range(0, neg_exp_num + 1):
            exp_bit = i + 2
            exp_value = -(exp_bit - 1)
            mant_bit = value_bit - exp_bit
            for j in range(int(2 ** mant_bit)):
                v = 2 ** exp_value * (1 + 2 ** (-mant_bit) * j)
                values.append(v)
                if self.is_signed:
                    values.append(-v)

        ## exponent zero
        exp_bit = 2
        exp_value = 0
        mant_bit = value_bit - exp_bit
        for j in range(int(2 ** mant_bit)):
            v = 2 ** (exp_value + exp_base) * (1 + 2 ** (-mant_bit) * j)
            values.append(v)
            if self.is_signed:
                values.append(-v)
                
        ## exponent positive     
        for i in range(1, pos_exp_num):
            exp_bit = i + 2
            exp_value = i
            mant_bit = value_bit - exp_bit
            for j in range(int(2 ** mant_bit)):
                v = 2 ** (exp_value + exp_base) * (1 + 2 ** (-mant_bit) * j)
                values.append(v)
                if self.is_signed:
                    values.append(-v)
                    
        ## max value
        values.append(2 ** exp_max)
        if self.is_signed:
            values.append(-2 ** exp_max)
            
        values = torch.tensor(values, device=self.quant_grid.device)
        values, _ = torch.sort(values)
        # add a bias to normalize the codebook (the threshold between outliers and normal values is 32)
        values *= 32 / (2 ** exp_max)

        return values
    
    # abfloat 
    @torch.no_grad()
    def outlier_value(self, exp_bit = 2, exp_base = 5):
        # 根據 group_size 決定 outlier bit 數
        # group=2 → 4-bit (E2M1，原論文)
        # group=4 → 12-bit (E2M8)
        # group=8 → 28-bit (E2M24)
        # Group=4 和 Group=8 都用 12-bit outlier（E2M9）
        # Group_size 只影響 victim 犧牲率，不影響 outlier 精度
        # outlier_bit 設計：
        # W4A4: g=2→4bit(E2M1), g=4→12bit(E2M9), g=8→12bit(E2M9)
        # 2W4A: g=4→6bit(E2M3), g=8→14bit(E2M11)
        if self.bit.item() <= 2:
            outlier_bit = (self.group_size - 1) * self.bit.item()
            exp_bit = 2
        else:
            outlier_bit = 12 if self.group_size >= 4 else self.bit.item()
            if self.group_size >= 4:
                exp_bit = 2
    
        B = outlier_bit
        if self.is_signed:
            B = B - 1
        
        value_bit = B
        mant_bit = value_bit - exp_bit
        values = []
        
        for i in range(exp_base, exp_base + 2 ** exp_bit):
            for j in range(int(2 ** mant_bit)):
                if i == exp_base and j == 0:
                    continue

                v = 2 ** i * (1 + 2 ** (-mant_bit) * j)
                values.append(v)
                if self.is_signed:
                    values.append(-v)
                    
        values = torch.tensor(values, device=self.quant_grid.device)
        values, _ = torch.sort(values)
                    
        return values

    @torch.no_grad()
    def mse_loss(self, quant_tensor, source_tensor, p=2.0, is_perchannel=True):
        if is_perchannel:
            mean_tensor =  (quant_tensor-source_tensor).abs().pow(p).view(quant_tensor.shape[0], -1).mean(-1).unsqueeze(1)
            return mean_tensor
        else:
            return (quant_tensor-source_tensor).abs().pow(p).mean()
        
    @torch.no_grad()
    def search_mse(self, tensor):
        if self.is_perchannel and (not self.is_input):
            if not self.args.no_outlier:
                mean = tensor.view(tensor.shape[0], -1).mean(dim=-1)
                std = tensor.view(tensor.shape[0], -1).std(dim=-1)
                x_max = torch.maximum((mean + 3 * std).abs(), (mean - 3 * std).abs())
            else:
                x_max, _ = tensor.view(tensor.shape[0], -1).abs().max(1)
            x_max = x_max.unsqueeze(1)            
            best_score = torch.ones_like(x_max) * 1e10
            alpha = x_max.clone()
            base_alpha = x_max.clone()
            lb = int(self.w_low)
            ub = int(self.w_up)
            for i in range(lb, ub, 2):
                new_alpha = base_alpha * (i * 0.01)
                self.alpha.data = new_alpha
                quant_tensor = self._forward(tensor)

                score = self.mse_loss(quant_tensor, tensor)
                alpha[score < best_score] = new_alpha[score < best_score]
                best_score[score < best_score] = score[score < best_score]
        else:        
            if not self.args.no_outlier:
                mean = tensor.mean()
                std = tensor.std()
                x_max = torch.maximum((mean + 3 * std).abs(), (mean - 3 * std).abs())
            else:
                x_max = tensor.abs().max()
            best_score = 1e10
            alpha = x_max.clone()
            base_alpha = alpha.clone()            
            lb = int(self.a_low)
            ub = int(self.a_up)
            for i in range(lb, ub, 2):
                new_alpha = base_alpha * (i * 0.01)
                self.alpha.data = new_alpha
                quant_tensor = self._forward(tensor)
                score = self.mse_loss(quant_tensor, tensor, p = 2, is_perchannel=False)
                if score < best_score:
                    best_score = score
                    alpha = new_alpha

        return best_score.sum(), alpha, (alpha / x_max).mean().item()

    @torch.no_grad()
    def search_adaptive_numeric_type(self, data):
        modes = []
        mse_list = []
        mode = self.mode
        if "-int" in mode:
            self.mode = 'int'
            self.quant_grid.data = self.int_value()
            best_score_int, _, _ = self.search_mse(data)
            modes.append('int')
            mse_list.append(best_score_int.item())

        if "-flint" in mode:
            self.mode = 'flint'
            self.quant_grid.data = self.flint_value()
            best_score_flint, _, _ = self.search_mse(data)
            modes.append('flint')
            mse_list.append(best_score_flint.item())

        mse_list = np.array(mse_list)
        mse_idx = np.argsort(mse_list)
        self.mode = modes[mse_idx[0]]

    @torch.no_grad()
    def _init_quant_para(self, data, data_b):
        with torch.no_grad():                    
            if self.has_inited_quant_para == 0:
                self.update_signed(data)                
                self.outliers.data = self.outlier_value()

                if self.is_perchannel:
                    x_max = data.view(data.shape[0], -1).abs().max(1).values
                    self.alpha.data = x_max.unsqueeze(1)
                else:
                    self.alpha.data = data.abs().max()

                if self.bit > 6:
                    self.mode = 'int'
                    self.quant_grid.data = self.int_value()
                else:
                    if "ant-" in self.mode:
                        self.search_adaptive_numeric_type(data)

                if self.mode == "flint":
                    self.quant_grid.data = self.flint_value()
                elif self.mode == "int":
                    self.quant_grid.data = self.int_value()
                else:
                    raise RuntimeError("Unsupported mode: " + self.mode)
                
                _, self.alpha.data, alpha_ratio = self.search_mse(data)

                quant_data = self._forward(data)
                self.mse = self.mse_loss(quant_data, data, 2, is_perchannel=self.is_perchannel).mean()
                print(self.mode, end="\t")
                print("%d-bit \t %s," %(self.bit.item(), self.name))
                
                self.has_inited_quant_para.data = torch.ones_like(self.has_inited_quant_para)
         
    @torch.no_grad()   
    def _forward(self, data, display=False):
        scale = self.alpha / torch.max(self.quant_grid)
        
        if self.is_perchannel: 
            data = (data.view(data.shape[0], -1) / scale).view(data.shape)
        else:
            data = data / scale
            
        if not self.args.no_outlier:
            # normal codebook (前面)
            normal_grid = self.quant_grid                            # 已經 sort 好
            normal_grid, _ = torch.sort(normal_grid)
            # outlier codebook (後面)
            outlier_grid = self.outliers
            outlier_grid, _ = torch.sort(outlier_grid)
            # 拼接：前 N 個 normal，後面 outlier（binary search 用）
            quant_grid = torch.cat((normal_grid, outlier_grid), dim=0)
        else:
            quant_grid = self.quant_grid
            
        # 全部走 CUDA kernel（不管 group_size）
        normal_size = self.quant_grid.shape[0] if not self.args.no_outlier else -1
        quant_data = QuantBase.forward(data, quant_grid, normal_size)
        shape = data.shape
        
        # Outlier Victim Pair Encoding / NoVictim Block Coding
        if not self.args.no_outlier:
            quant_data = quant_data.view(-1)
            N = quant_data.shape[0]

            if getattr(self.args, 'novictim', False):
                # ---- NoVictim Block Coding (向量化 v3) ----
                import sys, os
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                from novictim_codec import encode_blocks_batch, decode_blocks_batch, NO_OUTLIER, POS_STATES
                import numpy as np
                BSIZE = 16
                pad = (BSIZE - N % BSIZE) % BSIZE
                if pad > 0:
                    quant_data = torch.cat([quant_data, torch.zeros(pad, device=quant_data.device, dtype=quant_data.dtype)])
                flat = quant_data.cpu().numpy()
                # 動態 threshold：codebook 最大值 = (2^B-1) × 32/(2^B)
                B = self.bit.item() - 1
                dynamic_threshold = (2**B - 1) * 32 / (2**B)
                codes = encode_blocks_batch(flat, outlier_threshold=dynamic_threshold)
                out, scale_indices = decode_blocks_batch(codes)
                # 只對 Case A 乘 per-block scale
                scale_table = np.array([0.25, 0.50, 0.75, 1.00]) * dynamic_threshold
                num_blocks_actual = len(codes)
                is_caseA = ((codes % POS_STATES) == NO_OUTLIER)
                out_2d = out.reshape(num_blocks_actual, BSIZE)
                block_scales = np.where(is_caseA, scale_table[scale_indices], 1.0)
                out_2d = out_2d * block_scales[:, None]
                out = out_2d.reshape(-1)
                quant_data = torch.tensor(out, dtype=quant_data.dtype, device=quant_data.device)
                quant_data = quant_data[:N]
            else:
                # ---- 原本 OVP ----
                mask = quant_data.abs() > 32
                G = self.group_size
                pad = (G - N % G) % G
                if pad > 0:
                    quant_data = torch.cat([quant_data, torch.zeros(pad, device=quant_data.device, dtype=quant_data.dtype)])
                    mask = torch.cat([mask, torch.zeros(pad, dtype=torch.bool, device=mask.device)])
                groups = quant_data.view(-1, G)
                mask_g = mask.view(-1, G)
                has_outlier = mask_g.any(dim=1)
                outlier_idx = groups.abs().argmax(dim=1)
                victim_mask = torch.ones_like(groups, dtype=torch.bool)
                victim_mask.scatter_(1, outlier_idx.unsqueeze(1), False)
                victim_mask = victim_mask & has_outlier.unsqueeze(1)
                # victim 位置設為 0
                groups = groups * (~victim_mask)
                # flatten 回去，並去掉 padding
                quant_data = groups.view(-1)[:N]

        quant_data = quant_data.view(shape)
        tensor = (quant_data - data).detach() + data
        
        if self.is_perchannel:
            tensor = (tensor.view(tensor.shape[0], -1) * scale).view(data.shape)
        else:
            tensor = tensor * scale

        return tensor

    @torch.no_grad()
    def tensor_forward(self, tensor, input_tensor = None):
        if self.mode == "base":
            return tensor
        if not self.is_enable:
            return tensor
        if self.is_input:
            if not self.is_enable_activation:
                return tensor
        else:
            if not self.is_enable_weight:
                return tensor

        self._init_quant_para(tensor, input_tensor)

        q_tensor = self._forward(tensor)

        return q_tensor    

class TensorQuantizer(Quantizer):
    def __init__(self, **kwargs):
        super(TensorQuantizer, self).__init__(**kwargs)

    def forward(self, tensor, input_tensor = None):
        return self.tensor_forward(tensor, input_tensor)

class Conv1dQuantizer(nn.Module):
    """
    Class to quantize given convolutional layer
    """
    def __init__(self, mode=None, wbit=None, abit=None, args=None):
        super(Conv1dQuantizer, self).__init__()
        assert mode is not None,'Quantizer is not initilized!'
        gs = getattr(args, 'group_size', 2)
        self.quant_weight = TensorQuantizer(mode=mode, bit=wbit, is_signed=True, is_enable=True, args=args, operator=self._conv_forward, group_size=gs)
        self.quant_input  = TensorQuantizer(mode=mode, bit=abit, is_signed=False, is_enable=True, args=args, operator=self._conv_forward, is_input=True, group_size=gs)

    def set_param(self, conv):

        self.nf = conv.nf
        self.weight = nn.Parameter(conv.weight.data.clone())
        try:
            self.bias = nn.Parameter(conv.bias.data.clone())
        except AttributeError:
            self.bias = None
            
    def _conv_forward(self, x, weight):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), weight)
        x = x.view(size_out)
        return x

    def forward(self, input):
        weight = self.quant_weight(self.weight, input)
        input = self.quant_input(input, self.weight)
        return self._conv_forward(input, weight)


class Conv2dQuantizer(nn.Module):
    """
    Class to quantize given convolutional layer
    """
    def __init__(self, mode=None, wbit=None, abit=None, args=None):
        super(Conv2dQuantizer, self).__init__()
        assert mode is not None,'Quantizer is not initilized!'
        gs = getattr(args, 'group_size', 2)
        self.quant_weight = TensorQuantizer(mode=mode, bit=wbit, is_signed=True, is_enable=True, args=args, operator=self._conv_forward, group_size=gs)
        self.quant_input  = TensorQuantizer(mode=mode, bit=abit, is_signed=False, is_enable=True, args=args, operator=self._conv_forward, is_input=True, group_size=gs)

    def set_param(self, conv):
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        
        self.quant_weight.alpha.data = torch.ones([self.out_channels,1])

        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.weight = nn.Parameter(conv.weight.data.clone())
        try:
            self.bias = nn.Parameter(conv.bias.data.clone())
        except AttributeError:
            self.bias = None

    def _conv_forward(self, input, weight):
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        weight = self.quant_weight(self.weight, input)
        input = self.quant_input(input, self.weight)
        return self._conv_forward(input, weight)


class LinearQuantizer(nn.Module):
    """
    Class to quantize given linear layer
    """
    def __init__(self, mode=None, wbit=None, abit=None, args=None):
        super(LinearQuantizer, self).__init__()
        assert mode is not None,'Quantizer is not initilized!'
        gs = getattr(args, 'group_size', 2)
        self.quant_weight = TensorQuantizer(mode=mode, bit=wbit, is_signed=True, is_enable=True, args=args, operator=F.linear, group_size=gs)
        self.quant_input  = TensorQuantizer(mode=mode, bit=abit, is_signed=False, is_enable=True, args=args, operator=F.linear, is_input=True, group_size=gs)

    def set_param(self, linear):
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.quant_weight.alpha.data = torch.ones([self.out_features, 1])

        self.weight = nn.Parameter(linear.weight.data.clone())
        try:
            self.bias = nn.Parameter(linear.bias.data.clone())
        except AttributeError:
            self.bias = None

    def forward(self, input): 
        weight = self.quant_weight(self.weight, input) 
        input = self.quant_input(input, self.weight)
        return F.linear(input, weight, self.bias)