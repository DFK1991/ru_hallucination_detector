import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class CustomFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = torch.full_like(targets, self.alpha, dtype=torch.float) if isinstance(self.alpha, (int, float)) else self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum() if self.reduction == 'sum' else focal_loss

class HallucDetector(nn.Module):
    def __init__(self, lossft="CEL", use_numfeatslin=False, pools_list=['mean'], cl_layers=3, dor=0.4, model_name='cointegrated/rubert-tiny2', num_labels=4):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        h = self.bert.config.hidden_size  # 312
        self.pools_list = pools_list 
        self.attn_proj = nn.Linear(h, 1, bias=False)  # for attention pool
        self.numfeats_size = 32 if use_numfeatslin else 2  # for expanding num_feats (if needed)
        self.numfeats_lin = nn.Linear(2, self.numfeats_size, bias=True)
        self.use_numfeatslin = use_numfeatslin
        
        self.input_dim = h * (3 * len(pools_list)) + 2 + self.numfeats_size

        # classifier (head)
        if cl_layers == 3:
            self.classifier = nn.Sequential(
                nn.Linear(self.input_dim, h * 2),  # h - hidden_size (312)
                nn.LayerNorm(h *2 ),
                nn.GELU(),
                nn.Dropout(p=dor),
                nn.Linear(h * 2, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(p=dor),
                nn.Linear(h, num_labels)
            )
        elif cl_layers == 2:
            self.classifier = nn.Sequential(
                nn.Linear(self.input_dim, h * 2),
                nn.LayerNorm(h * 2),
                nn.GELU(),
                nn.Dropout(p=dor),
                nn.Linear(h * 2, num_labels)
            )
        else:
            raise ValueError("cl_layers must be 2 or 3")

        # Loss
        if lossft == "CEL":
            self.lossf = nn.CrossEntropyLoss()
        elif lossft == "CELwLS":
            self.lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
        elif lossft == "FL":
            self.lossf = CustomFocalLoss(gamma=2.0)
        else:
            raise ValueError("lossft must be: CEL, CELwLS, FL")
            
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, num_feats=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # retrieving [CLS] context [SEP] and evidence_aug [SEP]
        cntx_mask = (token_type_ids == 0) & (attention_mask == 1)
        evid_mask = (token_type_ids == 1) & (attention_mask == 1)
        
        # Attention Pooling
        def attn_pool(hidden_state, mask):
            scores = self.attn_proj(hidden_state).squeeze(-1)
            scores = scores.masked_fill(mask == 0, -1e9)
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            return (weights * hidden_state).sum(dim=1)

        # Max Pooling
        def max_pool(hidden_state, mask):
            mask_exp = mask.unsqueeze(-1).expand(hidden_state.size()).float()
            hidden_masked = hidden_state.masked_fill(mask_exp == 0, -1e9)
            return torch.max(hidden_masked, dim=1)[0]

        # Mean Pooling
        def mean_pool(hidden_state, mask):
            mask_exp = mask.unsqueeze(-1).float()
            sum_emb = (hidden_state * mask_exp).sum(dim=1)
            count = mask_exp.sum(dim=1).clamp(min=1e-9)
            return sum_emb / count

        ctx_attn_pool = attn_pool(last_hidden_state, cntx_mask)
        ctx_mean_pool = mean_pool(last_hidden_state, cntx_mask)
        ctx_max_pool = max_pool(last_hidden_state, cntx_mask)
        ctx_pool_dict = {'attn': ctx_attn_pool, 'mean': ctx_mean_pool, 'max': ctx_max_pool}
        ctx_pool_list = [ctx_pool_dict[x] for x in self.pools_list]

        evid_attn_pool = attn_pool(last_hidden_state, evid_mask)
        evid_mean_pool = mean_pool(last_hidden_state, evid_mask)
        evid_max_pool = max_pool(last_hidden_state, evid_mask)
        evid_pool_dict = {'attn': evid_attn_pool, 'mean': evid_mean_pool, 'max': evid_max_pool}
        evid_pool_list = [evid_pool_dict[x] for x in self.pools_list]
        
        v_ctx = torch.cat(ctx_pool_list, dim=1)
        v_evid = torch.cat(evid_pool_list, dim=1)
        diff_vec = v_ctx - v_evid

        # constant features
        cos_sim = F.cosine_similarity(v_ctx, v_evid, dim=1, eps=1e-8).unsqueeze(1)
        euc_norm = torch.norm(v_ctx - v_evid, p=1, dim=1, keepdim=True)

        all_feats_list = []
        if num_feats is None:
            num_feats = torch.zeros(input_ids.size(0), 2, device=input_ids.device)
        if self.use_numfeatslin:
            num_feats = self.numfeats_lin(num_feats)
        all_feats_list.extend([v_ctx, v_evid, diff_vec, num_feats, cos_sim, euc_norm])
        all_feats = torch.cat(all_feats_list, dim=1)
        
        logits = self.classifier(all_feats)

        loss = None
        if labels is not None:
            loss = self.lossf(logits, labels)
            
        return {"loss": loss, "logits": logits} if loss is not None else logits