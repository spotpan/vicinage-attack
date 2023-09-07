import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import math

torch.cuda.empty_cache()
parser = argparse.ArgumentParser(description='')
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=695, help='Random seed.')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--device', type=str, default='cuda', choices=['cuda','cpu'], help='device')
parser.add_argument('--oitr', type=int, default=5, help='Outer loop') 
parser.add_argument('--iitr1', type=int, default=1, help='Inner loop') 
parser.add_argument('--iitr2', type=int, default=12, help='Inner loop') 
parser.add_argument('--K_edges', type=int, default=200, help='Choose K edges') 

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
root_dir = os.getcwd().replace('\\', '/')


G = nx.DiGraph()
f = open("bitcoin_alpha/bitcoin_alpha_tri.txt","r")
for l in f:
    ls = l.strip().split(",")
    G.add_edge(ls[0], ls[1], weight = float(ls[2])) ## the weight should already be in the range of -1 to 1
f.close()
print('G:',G)


triple = []
f = open("bitcoin_alpha/bitcoin_alpha_tri.txt","r")
for l in f:
    ls = l.strip().split(",")
    #source, target, weight, stat, grdt
    triple.append( (int(ls[0]), int(ls[1]), float(ls[2]), float(1.0), float(0.0)) ) ## the weight should already be in the range of -1 to 1
f.close()


triple_c=[]
f = open("bitcoin_alpha/bitcoin_alpha_tri_search_pool_st.txt","r")
for l in f:
    ls = l.strip().split(",")
    #source, target, weight
    triple_c.append( (int(ls[0]), int(ls[1]), float(ls[2])) ) ## the weight should already be in the range of -1 to 1
f.close()


triple = np.array(triple)
triple_c = np.array(triple_c)
sample_nodes = np.random.choice(G.nodes(), 100, replace=False)
fixed_target_nodes = sample_nodes[:50]
#fixed_attack_nodes = sample_nodes[50:100]
fixed_target_nodes = np.array(fixed_target_nodes)
#fixed_attack_nodes = np.array(fixed_attack_nodes)
n_nos=len(G.nodes())
print('n_nos',n_nos)


beta=6
class my_sign_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (torch.div(torch.exp(beta*input)-1,torch.exp(beta*input)+1)+1)/2

    @staticmethod
    def backward(ctx, grad_output):
        input,=ctx.saved_tensors
        return beta*torch.div(beta*torch.exp(beta*input),torch.square(torch.exp(beta*input)+1)) * grad_output
mysign = my_sign_function.apply


class continuousB(nn.Module):
    def __init__(self, target_nos, n, tri, tri_c):
        super().__init__()
        self.tar = target_nos
        self.n=n
        self.weight = torch.Tensor(tri[:,2]).to(args.device)
        self.stat = torch.Tensor(tri[:,3]).to(args.device)
        self.grdt = nn.Parameter(torch.Tensor(tri_c[:,2]).to(args.device))
        self.tri = tri
        self.tri_c= tri_c

    def Weighted_W(self):
        W = torch.sparse_coo_tensor(self.tri[:,:2].T, self.weight, size=[self.n,self.n]).to_dense()
        return W

    def adjac_A(self):
        A = torch.sparse_coo_tensor(self.tri[:,:2].T, self.stat, size=[self.n,self.n]).to_dense()
        return A

    def maniplt_B(self):
        B = torch.sparse_coo_tensor(self.tri_c[:,:2].T, self.grdt, size=[self.n,self.n]).to_dense()
        return B
    
    def in_edges(self,Z,node):
        col=Z[:,node]
        return col.nonzero()

    def out_edges(self,Z,node):
        row=Z[node,:]
        return row.nonzero()

    def in_degree(self,Z,node):
        Z=torch.abs(Z[:,node])
        N_in=torch.sum(mysign(Z))
        return N_in

    def out_degree(self,Z,node):
        Z=torch.abs(Z[node,:])
        N_out=torch.sum(mysign(Z))
        return N_out

    def prior_scores(self):
        fairness = {}
        goodness = {}
        for node in range(self.n):
            fairness[node] = torch.as_tensor(1.0)
            goodness[node] = torch.as_tensor(1.0)
        return fairness, goodness

    def posterior_scores(self,B,Fair,Good):
        W=self.Weighted_W()
        A=self.adjac_A()
        Z=torch.mul(B,torch.sub(A,1))
        M=torch.mul(A,W)
        V=torch.add(Z,M)

        for node in range(self.n):
            inedges=self.in_edges(V,node)
            N_in=self.in_degree(V,node)
            g=0.0
            tmp=Good[node]
            for edge in inedges:
                g+=float(Fair[int(edge)])*V[edge,node]
            if(N_in.item()!=0):
                Good[node] = g/N_in
            else:
                Good[node] = tmp
        
        for node in range(self.n):
            outedges=self.out_edges(V,node)
            N_out=self.out_degree(V,node)
            f=0.0
            tmp=Fair[node]
            for edge in outedges:
                f+=1-abs( V[node,edge] - Good[int(edge)] )/2.0
            if(N_out.item()!=0):
                Fair[node] = f/N_out
            else:
                Fair[node]=tmp

        return Fair, Good
    
    def train(self,B,Fair,Good):
        W=self.Weighted_W()
        A=self.adjac_A()
        Z=torch.mul(B,torch.sub(A,1))
        M=torch.mul(A,W)
        V=torch.add(Z,M)
 
        for node in range(self.n):
            inedges=self.in_edges(V,node)
            N_in=self.in_degree(V,node)
            g=torch.mul(Fair[int(0)],0.0)
            tmp=Good[node]
            for edge in inedges:
                g=torch.add(g,torch.mul(Fair[int(edge)],V[edge,node]))
            if(N_in.item()!=0):
                Good[node] = torch.clamp(torch.div(g,N_in),min=-1,max=1)
            else:
                Good[node]=tmp
        
        for node in range(self.n):
            outedges=self.out_edges(V,node)
            N_out=self.out_degree(V,node)
            f=torch.mul(Good[int(0)],0.0)
            tmp=Fair[node]
            for edge in outedges:
                f=torch.add(f,1-torch.abs( V[node,edge] - Good[int(edge)] )/2.0)
            if(N_out.item()!=0):
                Fair[node] = torch.clamp(torch.div(f,N_out),min=0,max=1)
            else:
                Fair[node]=tmp

        G_SUM=0.0
        for node in self.tar:
            if(torch.isnan(Good[int(node)])):
                print('Yes!')
                continue
            G_SUM += Good[int(node)]
        print('Objective_Function:',G_SUM)

        return G_SUM

    def GOOD_SUM(self,Good):
        G_SUM=0.0
        for node in self.tar:
            if(torch.isnan(Good[int(node)])):
                continue
            G_SUM += Good[int(node)]
        return G_SUM


def initialize_scores(G):
    fairness = {}
    goodness = {}
    
    nodes = G.nodes()
    for node in nodes:
        fairness[node] = 1
        try:
            goodness[node] = G.in_degree(node)*1.0/G.in_degree(node)
        except:
            goodness[node] = 1
    return fairness, goodness

def compute_fairness_goodness(G):
    fairness, goodness = initialize_scores(G)
    
    nodes = G.nodes()
    iter = 0
    while iter < 100:
        df = 0
        dg = 0

        #print('-----------------')
        print("Iteration number", iter)
        
        #print('Updating goodness')
        for node in nodes:
            inedges = G.in_edges(node, data='weight')
            g = 0
            for edge in inedges:
                g += fairness[edge[0]]*edge[2]

            try:
                dg += abs(g/len(inedges) - goodness[node])
                goodness[node] = g/len(inedges)
            except:
                pass

        #print('Updating fairness')
        for node in nodes:
            outedges = G.out_edges(node, data='weight')
            f = 0
            for edge in outedges:
                f += 1.0 - abs(edge[2] - goodness[edge[1]])/2.0
            try:
                df += abs(f/len(outedges) - fairness[node])
                fairness[node] = f/len(outedges)
            except:
                pass
        
        #print('Differences in fairness score and goodness score = %.2f, %.2f' % (df, dg))
        if df < math.pow(10, -6) and dg < math.pow(10, -6):
            break
        iter+=1
    
    return fairness, goodness
#initialization: model, prior_score


model = continuousB(target_nos = fixed_target_nodes, n=n_nos, tri = triple, tri_c=triple_c)
fairness,goodness=model.prior_scores()

RECORD={}
RECORD[0]=model.GOOD_SUM(goodness).item()

add_pool=[]
del_pool=[]
for t in range(args.oitr):

    for i in range(args.iitr1):
        B=model.maniplt_B()
        res=model.train(B,fairness,goodness)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr) 
        optimizer.zero_grad()
        model.grdt.retain_grad()
        res.requires_grad_(True)
        res.backward(retain_graph=True)
        optimizer.step()
        model.grdt.data = torch.clamp(model.grdt.data, min=0, max=1)
        B=model.maniplt_B()
        fairness,goodness=model.posterior_scores(B,fairness,goodness)

        index=B.nonzero()
        index1=torch.LongTensor(index[:,0].tolist())
        index2=torch.LongTensor(index[:,1].tolist())
        indices=(index1,index2)
        current_budgets=torch.abs(B[indices]-0.5).mean().item()*len(index)
        if current_budgets >= args.K_edges:
            continue
    
    B=model.maniplt_B()
    for j in range(args.iitr2):
        fairness,goodness=model.posterior_scores(B,fairness,goodness)
    RECORD[t+1]=model.GOOD_SUM(goodness).item()
            
    if abs(RECORD[t+1]-RECORD[t]) < 1e-11 :
        break


B=model.maniplt_B()
A=model.adjac_A()
index=((A-B)<0).nonzero()
index1=torch.LongTensor(index[:,0].tolist())
index2=torch.LongTensor(index[:,1].tolist())
indices=(index1,index2)

index_list=index.tolist()
value_list=B[indices].tolist()
index_value=[]
for index in range(len(index_list)):
    u,v=index_list[index]
    w=value_list[index]
    index_value.append([w,u,v])
index_value.sort(reverse=True)

to_add=[]
for i in range(args.K_edges):
    to_add.append(index_value[i])
to_add=np.array(to_add)
to_add=to_add[:,1:]
to_add=to_add.tolist()
add_pool+=to_add

DEFENDER={}
Const=args.K_edges/10
Graph_clean=nx.DiGraph()
triple=model.tri.tolist()
for tup in triple:
    (u,v,w,stat,grdt)=tup
    Graph_clean.add_edge( str(int(u)), str(int(v)), weight=float(w) )
fairness, goodness = compute_fairness_goodness(Graph_clean)

goodness_evaluation=0
for node in fixed_target_nodes:
    goodness_evaluation += goodness[node]
DEFENDER[0]=goodness_evaluation

for t in range(10):
    if Const > args.K_edges:
        break
    Graph_aid=nx.DiGraph()

    triple=model.tri.tolist()
    for tup in triple:
        (u,v,w,stat,grdt)=tup
        Graph_aid.add_edge( str(int(u)), str(int(v)), weight=float(w) )
    print(Graph_aid)
    fairness, goodness = compute_fairness_goodness(Graph_aid)

    goodness_evaluation=0
    for node in fixed_target_nodes:
        goodness_evaluation += goodness[node]

    to_add=[]
    for i in range(Const):
        to_add.append(add_pool[i])
    for edge in to_add:
        (u,v)=edge
        Graph_aid.add_edge(str(int(u)), str(int(v)), weight=float(-1) )
    print(Graph_aid)

    #update F&G
    fairness, goodness = compute_fairness_goodness(Graph_aid)

    goodness_evaluation=0
    for node in fixed_target_nodes:
        goodness_evaluation += goodness[node]
    print("For Budget %d, the sum of goodness is %f" % (Const, goodness_evaluation))
    print("--------------------")
    DEFENDER[t+1]=goodness_evaluation
    Const+=args.K_edges/10

print('TRAIN:',RECORD)
print('TEST:',DEFENDER)
