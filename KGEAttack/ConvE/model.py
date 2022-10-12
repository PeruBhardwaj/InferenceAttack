import torch
from torch.nn import functional as F, Parameter
from torch.autograd import Variable


from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



class Complex(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations):
        super(Complex, self).__init__()
        self.num_entities = num_entities
        if args.max_norm:
            self.emb_e_real = torch.nn.Embedding(num_entities, args.embedding_dim, max_norm=1.0)
            self.emb_e_img = torch.nn.Embedding(num_entities, args.embedding_dim, max_norm=1.0)
            self.emb_rel_real = torch.nn.Embedding(num_relations, args.embedding_dim)
            self.emb_rel_img = torch.nn.Embedding(num_relations, args.embedding_dim)
        else:
            self.emb_e_real = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=None)
            self.emb_e_img = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=None)
            self.emb_rel_real = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=None)
            self.emb_rel_img = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=None)
        self.inp_drop = torch.nn.Dropout(args.input_drop)
        #self.loss = torch.nn.BCELoss()
        self.loss = torch.nn.BCEWithLogitsLoss()

    def init(self):
        xavier_normal_(self.emb_e_real.weight)
        xavier_normal_(self.emb_e_img.weight)
        xavier_normal_(self.emb_rel_real.weight)
        xavier_normal_(self.emb_rel_img.weight)

    def forward(self, sub, rel, mode='rhs', sigmoid=False):

        '''
        When mode is 'rhs' we expect (s,r); for 'lhs', we expect (o,r)
        For distmult, computations for both modes are equivalent, so we do not need if-else block
        '''
        if mode == 'lhs':
            e2_embedded_real = self.emb_e_real(sub).squeeze(dim=1)
            rel_embedded_real = self.emb_rel_real(rel).squeeze(dim=1)
            e2_embedded_img =  self.emb_e_img(sub).squeeze(dim=1)
            rel_embedded_img = self.emb_rel_img(rel).squeeze(dim=1)

            e2_embedded_real = self.inp_drop(e2_embedded_real)
            rel_embedded_real = self.inp_drop(rel_embedded_real)
            e2_embedded_img = self.inp_drop(e2_embedded_img)
            rel_embedded_img = self.inp_drop(rel_embedded_img)
        
            # complex space bilinear product (equivalent to HolE)
            realrealreal = torch.mm(rel_embedded_real*e2_embedded_real, self.emb_e_real.weight.transpose(1,0))
            realimgimg = torch.mm(rel_embedded_img*e2_embedded_img, self.emb_e_real.weight.transpose(1,0))
            imgrealimg = torch.mm(rel_embedded_real*e2_embedded_img, self.emb_e_img.weight.transpose(1,0))
            imgimgreal = torch.mm(rel_embedded_img*e2_embedded_real, self.emb_e_img.weight.transpose(1,0))
            pred = realrealreal + realimgimg + imgrealimg - imgimgreal
            
        else:
            e1_embedded_real = self.emb_e_real(sub).squeeze(dim=1)
            rel_embedded_real = self.emb_rel_real(rel).squeeze(dim=1)
            e1_embedded_img =  self.emb_e_img(sub).squeeze(dim=1)
            rel_embedded_img = self.emb_rel_img(rel).squeeze(dim=1)

            e1_embedded_real = self.inp_drop(e1_embedded_real)
            rel_embedded_real = self.inp_drop(rel_embedded_real)
            e1_embedded_img = self.inp_drop(e1_embedded_img)
            rel_embedded_img = self.inp_drop(rel_embedded_img)
        
            # complex space bilinear product (equivalent to HolE)
            realrealreal = torch.mm(e1_embedded_real*rel_embedded_real, self.emb_e_real.weight.transpose(1,0))
            realimgimg = torch.mm(e1_embedded_real*rel_embedded_img, self.emb_e_img.weight.transpose(1,0))
            imgrealimg = torch.mm(e1_embedded_img*rel_embedded_real, self.emb_e_img.weight.transpose(1,0))
            imgimgreal = torch.mm(e1_embedded_img*rel_embedded_img, self.emb_e_real.weight.transpose(1,0))
            pred = realrealreal + realimgimg + imgrealimg - imgimgreal
        
        
        if sigmoid:
            pred = torch.sigmoid(pred)

        return pred
    
    def score_triples(self, sub, rel, obj, sigmoid=False):
        #e1_embedded_real = self.emb_e_real(sub).squeeze(dim=1)
        #rel_embedded_real = self.emb_rel_real(rel).squeeze(dim=1)
        #e2_embedded_real = self.emb_e_real(obj).squeeze(dim=1)
        
        #e1_embedded_img =  self.emb_e_img(sub).squeeze(dim=1)
        #rel_embedded_img = self.emb_rel_img(rel).squeeze(dim=1)
        #e2_embedded_img = self.emb_e_img(obj).squeeze(dim=1)
        
        # complex space bilinear product (equivalent to HolE)
        #realrealreal = torch.mm(e1_embedded_real*rel_embedded_real, e2_embedded_real.transpose(1,0))
        #realimgimg = torch.mm(e1_embedded_real*rel_embedded_img, e2_embedded_img.transpose(1,0))
        #imgrealimg = torch.mm(e1_embedded_img*rel_embedded_real, e2_embedded_img.transpose(1,0))
        #imgimgreal = torch.mm(e1_embedded_img*rel_embedded_img, e2_embedded_real.transpose(1,0))
        #pred = realrealreal + realimgimg + imgrealimg - imgimgreal
        
        e_s_real = self.emb_e_real(sub).squeeze(dim=1)
        e_p_real = self.emb_rel_real(rel).squeeze(dim=1)
        e_o_real = self.emb_e_real(obj).squeeze(dim=1)
        
        e_s_img =  self.emb_e_img(sub).squeeze(dim=1)
        e_p_img = self.emb_rel_img(rel).squeeze(dim=1)
        e_o_img = self.emb_e_img(obj).squeeze(dim=1)
        
        realrealreal = torch.sum(e_s_real*e_p_real*e_o_real, dim=-1)
        realimgimg = torch.sum(e_s_real*e_p_img*e_o_img, axis=-1)
        imgrealimg = torch.sum(e_s_img*e_p_real*e_o_img, axis=-1)
        imgimgreal = torch.sum(e_s_img*e_p_img*e_o_real, axis=-1)
        pred = realrealreal + realimgimg + imgrealimg - imgimgreal
        
        if sigmoid:
            pred = torch.sigmoid(pred)

        return pred
    
    def score_emb(self, emb_s, emb_r, emb_o, sigmoid=False):
        # above interface works better for IJCAI attack
        emb_s_real, emb_s_img = torch.chunk(emb_s, 2, dim=-1)
        emb_r_real, emb_r_img = torch.chunk(emb_r, 2, dim=-1)
        emb_o_real, emb_o_img = torch.chunk(emb_o, 2, dim=-1)
        
        #realrealreal = torch.mm(emb_s_real*emb_r_real, emb_o_real.transpose(1,0))
        #realimgimg = torch.mm(emb_s_real*emb_r_img, emb_o_img.transpose(1,0))
        #imgrealimg = torch.mm(emb_s_img*emb_r_real, emb_o_img.transpose(1,0))
        #imgimgreal = torch.mm(emb_s_img*emb_r_img, emb_o_real.transpose(1,0))
        #pred = realrealreal + realimgimg + imgrealimg - imgimgreal
        
        realrealreal = torch.sum(emb_s_real*emb_r_real*emb_o_real, dim=-1)
        realimgimg = torch.sum(emb_s_real*emb_r_img*emb_o_img, axis=-1)
        imgrealimg = torch.sum(emb_s_img*emb_r_real*emb_o_img, axis=-1)
        imgimgreal = torch.sum(emb_s_img*emb_r_img*emb_o_real, axis=-1)
        pred = realrealreal + realimgimg + imgrealimg - imgimgreal
        
        if sigmoid:
            pred = torch.sigmoid(pred)

        return pred
    
    def score_triples_vec(self, sub, rel, obj, sigmoid=False):
        '''
        Inputs - subject, relation, object
        Return - a vector score for the triple instead of reducing over the embedding dimension
        '''
        s_real = self.emb_e_real(sub).squeeze(dim=1)
        rel_real = self.emb_rel_real(rel).squeeze(dim=1)
        o_real = self.emb_e_real(obj).squeeze(dim=1)
        
        s_img =  self.emb_e_img(sub).squeeze(dim=1)
        rel_img = self.emb_rel_img(rel).squeeze(dim=1)
        o_img = self.emb_e_img(obj).squeeze(dim=1)
        
#         sub_emb = self.emb_e(sub).squeeze(dim=1)
#         rel_emb = self.emb_rel(rel).squeeze(dim=1)
#         obj_emb = self.emb_e(obj).squeeze(dim=1)
        
#         s_real, s_img = torch.chunk(sub_emb, 2, dim=-1)
#         rel_real, rel_img = torch.chunk(rel_emb, 2, dim=-1)
#         o_real, o_img = torch.chunk(obj_emb, 2, dim=-1)
        
        realrealreal = s_real*rel_real*o_real
        realimgimg = s_real*rel_img*o_img
        imgrealimg = s_img*rel_real*o_img
        imgimgreal = s_img*rel_img*o_real
        
        pred = realrealreal + realimgimg + imgrealimg - imgimgreal
        
        if sigmoid:
            pred = torch.sigmoid(pred)

        return pred
    

class Transe(torch.nn.Module):
    '''
    Remember to check label_smoothing in main file while making changes to this
    '''
    def __init__(self, args, num_entities, num_relations):
        super(Transe, self).__init__()
        self.margin = args.transe_margin
        self.norm = args.transe_norm
        if args.max_norm:
            self.emb_e = torch.nn.Embedding(num_entities, args.embedding_dim, max_norm=1.0)
            self.emb_rel = torch.nn.Embedding(num_relations, args.embedding_dim)
        else:
            self.emb_e = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=None)
            self.emb_rel = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=None)
        
        self.inp_drop = torch.nn.Dropout(args.input_drop)
        
        #self.loss = torch.nn.BCELoss()
        self.loss = torch.nn.BCEWithLogitsLoss()

    def init(self):
        xavier_normal_(self.emb_e.weight)
        xavier_normal_(self.emb_rel.weight)

    def forward(self, sub, rel, mode='rhs', sigmoid=False):

        '''
        When mode is 'rhs' we expect (s,r); for 'lhs', we expect (o,r)
        For distmult, computations for both modes are equivalent, so we do not need if-else block
        '''
        batch_size, num_entities = sub.shape[0], self.emb_e.weight.shape[0]
        if mode == 'lhs':
            # obj_emb and rel_emb are of shape (num_batches, 1, emb_dim)
            obj_emb = self.emb_e(sub).squeeze(dim=1)
            rel_emb = self.emb_rel(rel).squeeze(dim=1)

            obj_emb = self.inp_drop(obj_emb)
            rel_emb = self.inp_drop(rel_emb)
            
            # below will be of shape (1, num_entities, emb_dim) to enable broadcast
            #sub_emb = self.emb_e.weight[None,:,:]
            
            #pred = self.emb_e.weight[None,:,:] + (rel_emb - obj_emb)[:,None, :]
            obj_emb = obj_emb.unsqueeze(dim=1)
            rel_emb = rel_emb.unsqueeze(dim=1)
            sub_emb = self.emb_e.weight.unsqueeze(dim=0)
            pred = sub_emb + (rel_emb - obj_emb)
            pred = self.margin - torch.norm(pred, p=self.norm, dim=-1)
            #pred = torch.mm(obj_emb*rel_emb, self.emb_e.weight.transpose(1,0))
            
        else:
            # sub_emb and rel_emb are of shape (num_batches, 1, emb_dim)
            sub_emb = self.emb_e(sub).squeeze(dim=1)
            rel_emb = self.emb_rel(rel).squeeze(dim=1)

            sub_emb = self.inp_drop(sub_emb)
            rel_emb = self.inp_drop(rel_emb)
            
            # below will be of shape (1, num_entities, emb_dim) to enable broadcast
            #obj_emb = self.emb_e.weight[None,:,:]
            
            #pred = (sub_emb + rel_emb)[:,None, :] - self.emb_e.weight[None,:,:]
            sub_emb = sub_emb.unsqueeze(dim=1)
            rel_emb = rel_emb.unsqueeze(dim=1)
            obj_emb = self.emb_e.weight.unsqueeze(dim=0)
            pred = (sub_emb + rel_emb) - obj_emb
            pred = self.margin - torch.norm(pred, p=self.norm, dim=-1)
            #pred = torch.mm(sub_emb*rel_emb, self.emb_e.weight.transpose(1,0))
        
        if sigmoid:
            pred = torch.sigmoid(pred)

        return pred
    
    def score_triples(self, sub, rel, obj, sigmoid=False):
        # below will be shape (num_batches, emb_dim)
        sub_emb = self.emb_e(sub).squeeze(dim=1)
        rel_emb = self.emb_rel(rel).squeeze(dim=1)
        obj_emb = self.emb_e(obj).squeeze(dim=1)
            
        pred = sub_emb + rel_emb - obj_emb
        pred = self.margin - torch.norm(pred, p=self.norm, dim=-1)
        
        if sigmoid:
            pred = torch.sigmoid(pred)

        return pred
    
    def score_emb(self, emb_s, emb_r, emb_o, sigmoid=False):
        pred = emb_s + emb_r - emb_o
        pred = self.margin - torch.norm(pred, p=self.norm, dim=-1)
        
        if sigmoid:
            pred = torch.sigmoid(pred)

        return pred
    
    def score_triples_vec(self, sub, rel, obj, sigmoid=False):
        '''
        Inputs - subject, relation, object
        Return - vector score for the triple instead of reducing over the embedding dimension
        '''
        sub_emb = self.emb_e(sub).squeeze(dim=1)
        rel_emb = self.emb_rel(rel).squeeze(dim=1)
        obj_emb = self.emb_e(obj).squeeze(dim=1)
        
        pred = -(sub_emb + rel_emb - obj_emb)
        pred += torch.tensor(self.margin).to(self.args.device).expand_as(pred)
        #pred = self.margin - torch.norm(pred, p=self.norm, dim=-1)
        
        if sigmoid:
            pred = torch.sigmoid(pred)

        return pred



class Distmult(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations):
        super(Distmult, self).__init__()
        
        if args.max_norm:
            self.emb_e = torch.nn.Embedding(num_entities, args.embedding_dim, max_norm=1.0)
            self.emb_rel = torch.nn.Embedding(num_relations, args.embedding_dim)
        else:
            self.emb_e = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=None)
            self.emb_rel = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=None)
        
        self.inp_drop = torch.nn.Dropout(args.input_drop)
        
        #self.loss = torch.nn.BCELoss()
        self.loss = torch.nn.BCEWithLogitsLoss()

    def init(self):
        xavier_normal_(self.emb_e.weight)
        xavier_normal_(self.emb_rel.weight)

    def forward(self, sub, rel, mode='rhs', sigmoid=False):
        '''
        When mode is 'rhs' we expect (s,r); for 'lhs', we expect (o,r)
        For distmult, computations for both modes are equivalent, so we do not need if-else block
        '''
        sub_emb = self.emb_e(sub)
        rel_emb = self.emb_rel(rel)
        sub_emb = sub_emb.squeeze(dim=1)
        rel_emb = rel_emb.squeeze(dim=1)

        sub_emb = self.inp_drop(sub_emb)
        rel_emb = self.inp_drop(rel_emb)

        pred = torch.mm(sub_emb*rel_emb, self.emb_e.weight.transpose(1,0))
        
        if sigmoid:
            pred = torch.sigmoid(pred)

        return pred
    
    def score_triples(self, sub, rel, obj, sigmoid=False):
        sub_emb = self.emb_e(sub).squeeze(dim=1)
        rel_emb = self.emb_rel(rel).squeeze(dim=1)
        obj_emb = self.emb_e(obj).squeeze(dim=1)
        
        pred = torch.sum(sub_emb*rel_emb*obj_emb, dim=-1)
        #pred = torch.mm(sub_emb*rel_emb, obj_emb.transpose(1,0))
        
        if sigmoid:
            pred = torch.sigmoid(pred)

        return pred
    
    def score_emb(self, emb_s, emb_r, emb_o, sigmoid=False):
        pred = torch.sum(emb_s*emb_r*emb_o, dim=-1)
        #pred = torch.mm(emb_s*emb_r, emb_o.transpose(1,0))
        
        if sigmoid:
            pred = torch.sigmoid(pred)

        return pred
    
    def score_triples_vec(self, sub, rel, obj, sigmoid=False):
        '''
        Inputs - subject, relation, object
        Return - a vector score for the triple instead of reducing over the embedding dimension
        '''
        sub_emb = self.emb_e(sub).squeeze(dim=1)
        rel_emb = self.emb_rel(rel).squeeze(dim=1)
        obj_emb = self.emb_e(obj).squeeze(dim=1)
        
        pred = sub_emb*rel_emb*obj_emb
        
        if sigmoid:
            pred = torch.sigmoid(pred)

        return pred



class Conve(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations):
        super(Conve, self).__init__()
        
        if args.max_norm:
            self.emb_e = torch.nn.Embedding(num_entities, args.embedding_dim, max_norm=1.0)
            self.emb_rel = torch.nn.Embedding(num_relations, args.embedding_dim)
        else:
            self.emb_e = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=None)
            self.emb_rel = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=None)
        
        self.inp_drop = torch.nn.Dropout(args.input_drop)
        self.hidden_drop = torch.nn.Dropout(args.hidden_drop)
        self.feature_drop = torch.nn.Dropout2d(args.feat_drop)
        
        self.embedding_dim = args.embedding_dim #default is 200
        self.num_filters = args.num_filters # default is 32
        self.kernel_size = args.kernel_size # default is 3
        self.stack_width = args.stack_width # default is 20
        self.stack_height = args.embedding_dim // self.stack_width
        
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.num_filters)
        self.bn2 = torch.nn.BatchNorm1d(args.embedding_dim)

        self.conv1 = torch.nn.Conv2d(1, out_channels=self.num_filters, 
                                     kernel_size=(self.kernel_size, self.kernel_size), 
                                     stride=1, padding=0, bias=args.use_bias)
        #self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=args.use_bias) # <-- default
        
        flat_sz_h = int(2*self.stack_width) - self.kernel_size + 1
        flat_sz_w = self.stack_height - self.kernel_size + 1
        self.flat_sz  = flat_sz_h*flat_sz_w*self.num_filters
        self.fc = torch.nn.Linear(self.flat_sz, args.embedding_dim)
        
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        
        #self.loss = torch.nn.BCELoss()
        self.loss = torch.nn.BCEWithLogitsLoss()

    def init(self):
        xavier_normal_(self.emb_e.weight)
        xavier_normal_(self.emb_rel.weight)

    def forward(self, sub, rel, mode='rhs', sigmoid=False):
        '''
        When mode is 'rhs' we expect (s,r); for 'lhs', we expect (o,r)
        For conve, both modes are equivalent, so we do not need if-else block
        '''
        
        sub_emb = self.emb_e(sub)
        rel_emb = self.emb_rel(rel)
        stacked_inputs = self.concat(sub_emb, rel_emb)
        stacked_inputs = self.bn0(stacked_inputs)
        x  = self.inp_drop(stacked_inputs)
        x  = self.conv1(x)
        x  = self.bn1(x)
        x  = F.relu(x)
        x  = self.feature_drop(x)
        #x  = x.view(x.shape[0], -1)
        x  = x.view(-1, self.flat_sz)
        x  = self.fc(x)
        x  = self.hidden_drop(x)
        x  = self.bn2(x)
        x  = F.relu(x)
        
        x  = torch.mm(x, self.emb_e.weight.transpose(1,0))
        x += self.b.expand_as(x)
        
        if sigmoid:
            pred = torch.sigmoid(x)
        else:
            pred = x

        return pred
    
    def score_triples(self, sub, rel, obj, sigmoid=False):
        sub_emb = self.emb_e(sub)
        rel_emb = self.emb_rel(rel)
        obj_emb = self.emb_e(obj)
        stacked_inputs = self.concat(sub_emb, rel_emb)
        stacked_inputs = self.bn0(stacked_inputs)
        x  = self.inp_drop(stacked_inputs)
        x  = self.conv1(x)
        x  = self.bn1(x)
        x  = F.relu(x)
        x  = self.feature_drop(x)
        #x  = x.view(x.shape[0], -1)
        x  = x.view(-1, self.flat_sz)
        x  = self.fc(x)
        x  = self.hidden_drop(x)
        x  = self.bn2(x)
        x  = F.relu(x)
        
        x = torch.mm(x, obj_emb.transpose(1,0)) 
        #x += self.b.expand_as(x) # leaving this out because can't use this for score_emb (score_trip is used by proposed attacks; but score_emb is used by IJCAI baseline)
        # above works fine for single input triples; 
        # but if input is batch of triples, then this is a matrix where diagonal is scores
        # so use torch.diagonal() after calling this function
        
        if sigmoid:
            pred = torch.sigmoid(x)
        else: #using BCE with logits
            pred = x
            
        return pred
        
        
    def score_emb(self, emb_s, emb_r, emb_o, sigmoid=False):
        stacked_inputs = self.concat(emb_s, emb_r)
        stacked_inputs = self.bn0(stacked_inputs)
        x  = self.inp_drop(stacked_inputs)
        x  = self.conv1(x)
        x  = self.bn1(x)
        x  = F.relu(x)
        x  = self.feature_drop(x)
        #x  = x.view(x.shape[0], -1)
        x  = x.view(-1, self.flat_sz)
        x  = self.fc(x)
        x  = self.hidden_drop(x)
        x  = self.bn2(x)
        x  = F.relu(x)
        
        x = torch.mm(x, emb_o.transpose(1,0))
        #x += self.b.expand_as(x) # can't use this because don't know which object (because IJCAI perturbs the embedding)
        
        if sigmoid:
            pred = torch.sigmoid(x)
        else: #using BCE with logits
            pred = x
            
        return pred
    
    def score_triples_vec(self, sub, rel, obj, sigmoid=False):
        '''
        Inputs - subject, relation, object
        Return - a vector score for the triple instead of reducing over the embedding dimension
        '''
        sub_emb = self.emb_e(sub)
        rel_emb = self.emb_rel(rel)
        obj_emb = self.emb_e(obj)
        
        x = self.conve_architecture(sub_emb, rel_emb)
        
        #pred = torch.mm(x, obj_emb.transpose(1,0))
        pred = x*obj_emb
        #print(pred.shape, self.b[obj].shape) #shapes are [7,200] and [7]
        #pred += self.b[obj].expand_as(pred) #taking the bias value for object embedding - can't add scalar to vector
        
        #pred = sub_emb*rel_emb*obj_emb
        
        if sigmoid:
            pred = torch.sigmoid(pred)

        return pred
    
    def concat(self, e1_embed, rel_embed, form='plain'):
        if form == 'plain':
            e1_embed = e1_embed. view(-1, 1, self.stack_width, self.stack_height)
            rel_embed = rel_embed.view(-1, 1, self.stack_width, self.stack_height)
            stack_inp = torch.cat([e1_embed, rel_embed], 2)

        elif form == 'alternate':
            e1_embed = e1_embed. view(-1, 1, self.embedding_dim)
            rel_embed = rel_embed.view(-1, 1, self.embedding_dim)
            stack_inp = torch.cat([e1_embed, rel_embed], 1)
            stack_inp = torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2*self.stack_width, self.stack_height))

        else: raise NotImplementedError
        return stack_inp
    
    def conve_architecture(self, sub_emb, rel_emb):
        stacked_inputs = self.concat(sub_emb, rel_emb)
        stacked_inputs = self.bn0(stacked_inputs)
        x  = self.inp_drop(stacked_inputs)
        x  = self.conv1(x)
        x  = self.bn1(x)
        x  = F.relu(x)
        x  = self.feature_drop(x)
        #x  = x.view(x.shape[0], -1)
        x  = x.view(-1, self.flat_sz)
        x  = self.fc(x)
        x  = self.hidden_drop(x)
        x  = self.bn2(x)
        x  = F.relu(x)
        
        return x


# Add your own model here

class MyModel(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(DistMult, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(args.input_drop)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_embedded= self.emb_e(e1)
        rel_embedded= self.emb_rel(rel)

        # Add your model function here
        # The model function should operate on the embeddings e1 and rel
        # and output scores for all entities (you will need a projection layer
        # with output size num_relations (from constructor above)

        # generate output scores here
        prediction = torch.sigmoid(output)

        return prediction

