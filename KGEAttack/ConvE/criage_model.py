import torch
from torch.nn import functional as F, Parameter
from torch.autograd import Variable


from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



class Distmult(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations):
        super(Distmult, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=None)
        self.emb_rel = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=None)
        self.emb_e.weight.requires_grad=False 
        self.emb_rel.weight.requires_grad=False 
        
        self.linear_t = torch.nn.Linear(args.embedding_dim, args.embedding_dim)
        self.linear_rel = torch.nn.Linear(args.embedding_dim, num_relations)
        self.linear_e1 = torch.nn.Linear(args.embedding_dim, num_entities)
        self.linear_t.weight.requires_grad=True
        self.linear_e1.weight.requires_grad=True
        self.linear_rel.weight.requires_grad=True
        self.hidden_drop = torch.nn.Dropout(args.hidden_drop)
        
        self.inp_drop = torch.nn.Dropout(args.input_drop)
        self.loss = torch.nn.CrossEntropyLoss()
        
        self.args = args

    def init(self):
        xavier_normal(self.emb_e.weight.data)
        xavier_normal(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        #e1_embedded= self.emb_e(e1)
        #rel_embedded= self.emb_rel(rel)
        #e1_embedded = e1_embedded.view(-1, args.embedding_dim)
        #rel_embedded = rel_embedded.view(-1, args.embedding_dim)

        #pred = e1_embedded*rel_embedded
        pred = self.encoder(e1, rel)
        return self.decoder(pred)

        
    def encoder(self, e1, rel):
        e1_embedded= self.emb_e(e1)
        rel_embedded= self.emb_rel(rel)
        e1_embedded = e1_embedded.squeeze(dim=1)
        rel_embedded = rel_embedded.squeeze(dim=1)

        pred = e1_embedded*rel_embedded

        return pred

    def encoder_2(self, e1):
        e1_embedded= self.emb_e(e1)
        return e1_embedded

    def decoder(self, pred):
        pred = self.linear_t(pred)
        pred= F.relu(pred)
        E1 = self.linear_e1(pred)
        R = self.linear_rel(pred)
        return E1, R


class Conve(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations):
        super(Conve, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=None)
        self.emb_rel = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=None)
        self.emb_e.weight.requires_grad = False
        self.emb_rel.weight.requires_grad = False
        
        self.embedding_dim = args.embedding_dim #default is 200
        self.num_filters = args.num_filters # default is 32
        self.kernel_size = args.kernel_size # default is 3
        self.stack_width = args.stack_width # default is 20
        self.stack_height = args.embedding_dim // self.stack_width
        
        
        flat_sz_h = int(2*self.stack_width) - self.kernel_size + 1
        flat_sz_w = self.stack_height - self.kernel_size + 1
        self.flat_sz  = flat_sz_h*flat_sz_w*self.num_filters
        
        self.linear_t  = torch.nn.Linear(args.embedding_dim, self.flat_sz)
        self.linear_rel = torch.nn.Linear(2*args.embedding_dim, num_relations) # 2* is needed because encoder stacks the embeddings
        self.linear_e1 = torch.nn.Linear(2*args.embedding_dim, num_entities)
        
        self.deconv1= torch.nn.ConvTranspose2d(in_channels =32, out_channels=1, kernel_size =3)
        
        self.linear_t.weight.requires_grad = True
        self.linear_rel.weight.requires_grad = True
        self.linear_e1.weight.requires_grad = True
        self.deconv1.weight.requires_grad = True
        
        self.inp_drop = torch.nn.Dropout(args.input_drop)
        self.hidden_drop = torch.nn.Dropout(args.hidden_drop)
        self.feature_map_drop = torch.nn.Dropout2d(args.feat_drop)
        self.loss = torch.nn.CrossEntropyLoss()
        #self.loss = torch.nn.BCELoss()
        #self.emb_dim1 = args.embedding_shape1
        #self.emb_dim2 = args.embedding_dim // self.emb_dim1

        self.conv1 = torch.nn.Conv2d(1, out_channels=self.num_filters, 
                                     kernel_size=(self.kernel_size, self.kernel_size), 
                                     stride=1, padding=0, bias=args.use_bias)
        #self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=args.use_bias)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.num_filters)
        self.bn2 = torch.nn.BatchNorm1d(args.embedding_dim)
                                     
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(self.flat_sz,args.embedding_dim)
        self.conv1.weight.requires_grad = False
        self.fc.weight.requires_grad = False
        self.args = args

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        x = self.encoder(e1, rel)

        return self.decoder(x)
    
    def encoder(self, e1, rel):
        #e1_embedded= self.emb_e(e1).view(-1, 1, self.emb_dim1, self.emb_dim2)
        e1_embedded = self.emb_e(e1).view(-1, 1, self.stack_width, self.stack_height)
        #rel_embedded = self.emb_rel(rel).view(-1, 1, self.emb_dim1, self.emb_dim2)
        rel_embedded = self.emb_rel(rel).view(-1, 1, self.stack_width, self.stack_height)

        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)

        stacked_inputs = self.bn0(stacked_inputs)
        #x= self.inp_drop(stacked_inputs)
        x = stacked_inputs
        #print(x.shape)
        x= self.conv1(x)
        #print(x.shape)
        x= self.bn1(x)
        x= F.relu(x)
        #x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        #x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        #x = torch.mm(x, self.emb_e.weight.transpose(1,0))
        #x += self.b.expand_as(x)
        #pred = torch.sigmoid(x)
        
        return x
    
    def encoder_2(self, e1):
        e1_embedded = self.emb_e(e1)
        return e1_embedded
    
    def decoder(self, pred):
        if self.args.embedding_dim == 1000:
            pred = self.linear_t(pred).view(-1, 32, 38, 48) #I got these reshape values by printing shape after conv in encoder
        else:
            pred = self.linear_t(pred).view(-1, 32, 38, 8) #I got these reshape values by printing shape after conv in encoder
        #print(pred.shape)
        pred = self.deconv1(pred)
        #print(pred.shape)
        
        pred = F.relu(pred.view(-1, 2*self.args.embedding_dim))
        E1 = self.linear_e1(pred)
        R = self.linear_rel(pred)
        return E1, R




