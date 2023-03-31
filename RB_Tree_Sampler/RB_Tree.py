   import time
#import torch.nn.Functional as F
import torch

from ctypes import *
import ctypes

import copy##
import random##
adder = CDLL("./adder_5.so")
#adder.restype = pointer(dtype= ctypes.c_int)#, shape=(10,) )# shape是你数组的形状，要贴合你C++的数组shape


data_list_ = []
key_list = []
data_num_list = []##For key of each data not of each nodes
#定义红黑树
class RBTree(object):
    def __init__(self):
        self.nil = RBTreeNode(torch.zeros(128), None, None, None)
        self.root = self.nil
        self.HitRate = 0
        #self.level = 0##the num of the first level is 0
        self.n_hard = []
class RBTreeNode(object):
    def __init__(self, key, data_num, data_y, last_loss, last_time = None, T = None):
        ##T for not None only when we initialize a RBTreeNode
        self.key = key ##key for feature deduced by the model
        self.Last_Time = last_time
        self.data = [data_num]
        self.y = data_y## data_y for data.y
        ##use data.x[0][0] instead of num self.num = num## num for data_num in the dataset
        self.last_loss = last_loss
        ##
        self.if_h = 0##mean if it's a hard node
        self.if_e = 0
        self.in_hard = False
        self.in_Topic = False
        self.Topic_Root = False
        self.Seed_Begin = False##abandoned
        self.Colse_to_bridge = 0##mean distance to the edge of an another topic cluster it ranges from 1 to 5 the bigger the closer
        self.level = 0##the num of the first level is 0
        ##
        self.HitRate = 0
        self.shadowconnect = []
        #self.seedconnect = None
        self.if_seed = False
        ##
        if T != None:
            self.left = T.nil
            self.right = T.nil
        else:
            self.left = None
            self.right = None
        self.parent = None
        self.color = 'black'
        self.size=None

##Hard nodes topic#x for hard ones
def H_topic(T, x):
    if Sim(x, x.parent) < 0.2 and x.in_hard == True:
        if Sim(x, x,right) < 0.2 or Sim(x, x.left):
            x.parent.in_Topic = True
            x.parent.Topic_Root = True
        elif x.left.in_hard == True:
            CExchange(T,x)
            x.parent.in_Topic = True
            x.parent.Topic_Root = True
        elif x.right.in_hard == True:
            x.parent.in_Topic = True
            x.parent.Topic_Root = True
    elif Sim(x.key, x.right.key) < 0.2 or Sim(x.key, x.left.key) < 0.2:
        x.in_Topic = True
        x.Topic_Root = True
##assume for F2Sim ones
def Sim(x_1: torch.Tensor, x_2: torch.Tensor):
    Sim_dist = torch.norm(x_1 - x_2, p=2)
    return torch.tanh(1/4*Sim_dist.clamp(min=1e-12).sqrt() ) # for numerical stability
    #-torch.mean(x_1 - x_norm) + torch.log(torch.mean(torch.exp(x_2 - x_norm)))

##assume for CosSim ones
def Sim_special(x_1: torch.Tensor, x_2: torch.Tensor, x_norm: torch.Tensor):
    a = x_1 - x_norm
    b = x_2 - x_norm
    return torch.cosine_similarity(a, b, dim=0)
##左右孩子节点交换
def CExchange( T, x):
    y = x.right
    x.right = x.left
    x.left = y
##将较难叶子节点移动至右端
def Move_hard( T, x):
    x.level = G_x_level(T , x)
    #T.level = 10# ProxHigh(T, T.root)
    #for i in range(from_, T.level):
    while x.level > max(T.level) +1:
        #if x.parent != T.nil:
        y = x.parent
        if x == y.left:
            CExchange( T, y)
    T.n_hard.append(x.level)
##Get x level(x should not be root!)
def G_x_level( T, x):
    num = 0
    while x.parent != T.nil:
        x = x.parent
        num += 1
    return num

#左旋转
def LeftRotate( T, x):
    y = x.right
    '''##
    if x.Topic_Root == True:
        x.Topic_Root = False
        y.Topic_Root = True
    if x.parent != T.nil and x.parent.Seed_Begin == True:
        ##JUst to mention that I've thought on that situation and found I needn't to handle it 

        pass
    
    if y.Topic_Root == True:
        if y.left is not None:
            y.left.Topic_Root = True
            y.shadowconnect = y.left
            y.left.shadowconnect = y
    if x.Seed_Begin == True:
        if x.right.if_seed == True:
            x.right.if_seed = False
            x.right.Seed_Begin = True
    ##'''
    
    x.right = y.left
    if y.left != T.nil:
        y.left.parent = x
    y.parent = x.parent
    if x == T.root:
        T.root = y
        y.parent = T.nil
    ##
    elif x == x.parent.left:
        x.parent.left = y
    else:
        x.parent.right = y
    y.left = x
    x.parent = y
#右旋转
def RightRotate( T, x):
    y = x.left
    '''##
    if x.Topic_Root == True:
        x.Topic_Root = False
        y.Topic_Root = True
    if y.Topic_Root == True:
        if y.right is not None:
            y.right.Topic_Root = True
            y.shadowconnect = y.right
            y.right.shadowconnect = y
    if x.Seed_Begin == True:
        if x.right.if_seed == True:
            x.right.if_seed = False
            x.right.Seed_Begin = True
    ##'''
    
    x.left = y.right
    if y.right != T.nil:
        y.right.parent = x
    y.parent = x.parent
    if x.parent == T.nil:
        T.root = y
    
    elif x == x.parent.right:
        x.parent.right = y
    else:
        x.parent.left = y
    y.right = x
    x.parent = y
    
BeginPoint = [None]*5 ## for those node which have more degree_num and/or higher hitrate 
#红黑树的插入
def RBInsert(T, T_num, z, Last_node = None, if_hard = False, if_insert = True, Return_node = False, if_respace = False):
    ##In fact this fuction supposed to be insert or find the position. But in name of RBInsert would make readers know where it origins from the normal RBTree functions  
    
    ####
    global BeginPoint
    if Last_node is None:
        y = T.root
        x = T.root
        max_sim = 0##
        stop = True##
        
        if BeginPoint[T_num] != None:
            for node in BeginPoint[T_num]:
                sim_ = Sim(z.key, node.key)
                if sim_ >= 0.8:
                    x = node
                    max_sim = sim_
        while x != T.nil:
            
            stop = True
            y = x
            sim_candidate = [];
            if x.left != None and x.left != T.nil:
                sim_l = Sim(z.key, x.left.key)
                
                if sim_l > max_sim:
                    x = x.left
                    max_sim = sim_l
                    stop = False       
            
            if x.right != None and x.right != T.nil:
                sim_r = Sim(z.key, x.right.key)
                if sim_r > max_sim:
                    x = x.right
                    max_sim = sim_r
                    stop = False
           
            if stop:
                break
        y = x
        y.HitRate += 1
        if y.HitRate + len(y.shadowconnect) > 10 and y not in BeginPoint:
            BeginPoint.append(y)
        if if_insert == False:
            ##False for not in exact the original place while True for still there
            if z.data[0] in y.data:
                if y != None and y.parent == None:
                    print("no_1")
                return y, True
            else:
                if y == T.root and y.right == T.nil and y.left == T.nil and if_insert == True:
                    print("no_3")
                return y, False
            ####
        sim_l = 0
        sim_r = 0
        if y.left != None and y.left != T.nil:
            sim_l = Sim(z.key, y.left.key)         
        if y.right != None and y.right != T.nil:
            sim_r = Sim(z.key, y.right.key)
            ####
        if sim_r >= sim_l:
            if y.right == T.nil:
                done = True
                z.color = 'red'
                z.parent = y
                z.left = T.nil
                z.right = T.nil
                y.right = z
                RBInsertFixup(T, z)
                return y, True
        else:
            if y.left == T.nil:
                done = True
                z.color = 'red'
                print("no_4")
                z.parent = y
                z.left = T.nil
                z.right = T.nil
                y.left = z
                RBInsertFixup(T, z)
                return y, True
        Last_node = y
            
    if Last_node is not None :
        if Last_node.last_loss < 0.1 and z.last_loss < 1:
            if Sim_special(Last_node.key, z.key, Last_node.left.key) > 0.5:
                for i in range(0, len(z.data)):
                    Last_node.data.append(z.data[i])
                return y, False
            #if Last_node.right is T.nil or Last_node.right.last_loss < 0.1:  
        
        z.parent = Last_node
        left_sim = 0
        right_sim = 0
        if Last_node.left != T.nil and Last_node.left != None :
            left_sim = Sim(z.key, Last_node.left.key)
        else:
            left_sim = 1
        if Last_node.right != T.nil and Last_node.right != None :
            right_sim = Sim(z.key, Last_node.right.key)
        else:
            right_sim = 1
        if left_sim >= right_sim:
            z.left = Last_node.left
            z.right = T.nil
            Last_node.left = z##优先向左插入##the model is set to insert to left first
            z.parent = Last_node 
            z.color = 'red'
            if z.left.color == 'black':
                RBInsertFixup(T, z)
            elif Last_node.right.color == 'balck':
                RightRotate(T, z)
                z.right = Last_node.right
                Last_node.right.parent = z
                Last_node.right = z
                Last_node.left.right = T.nil
            else:
                RightRotate(T, z)
                Last_node.color = 'red'
                Last_node.left.color = 'black'
                if Last_node.right.left != T.nil:
                    Last_node.right.left.color = 'red'
                if Last_node.right.right != T.nil:
                    Last_node.right.right.color = 'red'
                
        else:
            z.right = Last_node.right
            z.left = T.nil
            Last_node.right = z
            z.parent = Last_node      
            z.color = 'red'
            if z.right == T.nil or z.right == None or z.right.color == 'black':
                #print("yes t")
                RBInsertFixup(T, z)
            elif Last_node.right.color == 'balck':
                LeftRotate(T, z)
                z.left = Last_node.left
                Last_node.left.parent = z
                Last_node.left = z
                Last_node.right.left = T.nil
            else:
                LeftRotate(T, z)
                Last_node.color = 'red'
                Last_node.right.color = 'black'
                if Last_node.left.right != T.nil:
                    Last_node.left.right.color = 'red'
                if Last_node.left.left != T.nil:
                    Last_node.left.left.color = 'red'
    
    if if_hard == True:
        Move_hard( T, z)
    return z, True##z.key, '颜色为', z.color
    
#红黑树的上色
def RBInsertFixup( T, z):        
    while z.parent.color == 'red':
        if z.parent == z.parent.parent.left:
            y = z.parent.parent.right##on
            if y != T.nil and y.color == 'red':
                z.parent.color = 'black'
                y.color = 'black'
                z.parent.parent.color = 'red'
                z = z.parent.parent
            else:
                if z == z.parent.right:
                    z = z.parent
                    LeftRotate(T, z)
                z.parent.color = 'black'
                z.parent.parent.color = 'red'
                RightRotate(T,z.parent.parent)
        else:
            y = z.parent.parent.left
            if y.color == 'red':
                z.parent.color = 'black'
                y.color = 'black'
                z.parent.parent.color = 'red'
                z = z.parent.parent
            else:
                if z == z.parent.left:
                    z = z.parent
                    RightRotate(T, z)
                z.parent.color = 'black'
                z.parent.parent.color = 'red'
                LeftRotate(T, z.parent.parent)
    T.root.color = 'black'
def RBTransplant( T, u, v):
    if u.Topic_Root == True:
        v.Topic_Root = True    
    if u.parent == T.nil:
        T.root = v
    elif u == u.parent.left:
        u.parent.left = v
    else:
        u.parent.right = v
    v.parent = u.parent
##For Hard topic
def HTDeletedFixup(T, z):
    out = False
    start = time.perf_counter()
    x = z
    while x != T.root and x.color == 'black':
        if x == x.parent.left:
            w = x.parent.right
            if w.color == 'red':
                w.color = 'black'
                x.parent.color = 'red'
                LeftRotate(T, x.parent)
                #if x.parent.Topic_Root == True:
                    
                w = x.parent.right
            if w.left.color == 'black' and w.right.color == 'black':
                w.color = 'red'
                x = x.parent
                if x.in_Topic == False:
                    out = True
            else:
                if w.right.color == 'black':
                    w.left.color = 'black'
                    w.color = 'red'
                    RightRotate(T, w)
                    w = x.parent.right
                w.color = x.parent.color
                x.parent.color = 'black'
                w.right.color = 'black'
                LeftRotate(T, x.parent)
                if out == False:
                    if x.Topic_Root != 0:
                        x = x.parent
                else:
                    x = T.root
        else:
            w = x.parent.left
            if w.color == 'red':
                w.color = 'black'
                x.parent.color = 'red'
                RightRotate(T, x.parent)
                w = x.parent.left
            if w.right.color == 'black' and w.left.color == 'black':
                w.color = 'red'
                x = x.parent
                if x.in_Topic == False:
                    out = True
            else:
                if w.left.color == 'black':
                    w.right.color = 'black'
                    w.color = 'red'
                    LeftRotate(T, w)
                    w = x.parent.left
                w.color = x.parent.color
                x.parent.color = 'black'
                w.left.color = 'black'
                RightRotate(T, x.parent)
                if out == False:
                    if x.Topic_Root != 0:
                        x = x.parent
                else:
                    x = T.root
    x.color = 'black'
    end = time.perf_counter()
    print('r:', end - start)
def RBDelete(T, z):
    print('z_c:', z.color)
    print('z_k:',z.key)
    s = time.perf_counter()
    y = z
    y_original_color = y.color
    if z.left == T.nil:
        x = z.right
        RBTransplant(T, z, z.right)
    elif z.right == T.nil:
        x = z.left
        RBTransplant(T, z, z.left)
    else:
        y = TreeMinimum(z.right)
        y_original_color = y.color
        x = y.right
        if y.parent == z:
            x.parent = y
        else:
            RBTransplant(T, y, y.right)
            y.right = z.right
            y.right.parent = y
        RBTransplant(T, z, y)
        y.left = z.left
        y.left.parent = y
        y.color = z.color
    if y_original_color == 'black':
        e = time.perf_counter()
        RBDeleteFixup(T, x)
    else:
        e = time.perf_counter()
    print('r_0:', e-s)
#红黑树的删除
def RBDeleteFixup(T, x):
    start = time.perf_counter()
    while x != T.root and x.color == 'black':
        if x == x.parent.left:
            w = x.parent.right
            if w.color == 'red':
                w.color = 'black'
                x.parent.color = 'red'
                LeftRotate(T, x.parent)
                w = x.parent.right
            if w.left.color == 'black' and w.right.color == 'black':
                w.color = 'red'
                x = x.parent
            else:
                if w.right.color == 'black':
                    w.left.color = 'black'
                    w.color = 'red'
                    RightRotate(T, w)
                    w = x.parent.right
                w.color = x.parent.color
                x.parent.color = 'black'
                w.right.color = 'black'
                LeftRotate(T, x.parent)
                x = T.root
        else:
            w = x.parent.left
            if w.color == 'red':
                w.color = 'black'
                x.parent.color = 'red'
                RightRotate(T, x.parent)
                w = x.parent.left
            if w.right.color == 'black' and w.left.color == 'black':
                w.color = 'red'
                x = x.parent
            else:
                if w.left.color == 'black':
                    w.right.color = 'black'
                    w.color = 'red'
                    LeftRotate(T, w)
                    w = x.parent.left
                w.color = x.parent.color
                x.parent.color = 'black'
                w.left.color = 'black'
                RightRotate(T, x.parent)
                x = T.root
    x.color = 'black'
    end = time.perf_counter()
    print('r:', end - start)
'''
def ProxHigh(T, x):
    h = 0
    
    if T.root.left != T.nil:
        x.left = T.root.left
    if T.root.right != T.nil:
        x.right = T.root.left
    print('x_k:', type(x))
    if True:        
        while x.left != T.nil or x.right != 0:
            if x.left == T.nil:
                x = x.right
            else:
                x = x.left
            h = h + 1
        return h 
    else: 
        print('w')
        return 0
       '''
#def LearningConnection(x, y):

'''def Topicsort(T, x, sample, sample_location, sample_num, max, min):
    ##Screen often refer to TV so I use 'sort' to discribe these functions, just think it like rank 
    ##probabilities of nodes to be sampled
    ###
    x_ = x.key
    sim_ = Sim(x.key, x.parent.key)
    #sim_m = []
    
    while sim_ < 0.2:
        ##0.2 is a randomly set num
        x = x.parent
        sim_ = Sim(x.key, x.parent.key)
        #sim_m.append(torch.Tensor(sim_))
    #sim_m_ = torch.cat(sim_m,dim=0)
    #sim_ = torch.mean(sim_m_)
    TopicSample(T = T, x = x, sample_num = 64, max = 64, min = 0, sample_location = sample_location, Begin = True)
    '''

    


####
sample_ = []
sample_num = 0
for_topic_sample_edge_e = []
for_topic_sample_node_list = []
for_topic_sample_node_list_n = []##废弃
for_topic_sample_edge_f = []####
for_topic_sample_edge_t = []####
for_topic_sample_edge_w = []
##for_topic_sample_edge_p = []####

def FrontSample(T, x, x_last, x_standard, Sim_Min, x_last_p = None, shadow_num = None, target_node = None):
    ##on
    ##x_last_p is in the list way
    global for_topic_sample_edge_f
    global for_topic_sample_edge_t
    global for_topic_sample_edge_w 
    ##global for_topic_sample_edge_p 
    global for_topic_sample_edge_e 
    global for_topic_sample_node_list
    global for_topic_sample_node_list_n
    
    x_p = None
    if x != T.nil and x != None :     
        if len(for_topic_sample_edge_e) >= 0 and len(for_topic_sample_edge_e) < 2000:
            print("1")
        else:
            print("2:", len(for_topic_sample_edge_e), type(for_topic_sample_edge_e))
            
        ####if shadow_num != None:
        sim_ = Sim(x.key, x_last.key)
        term_1 = False
        term_1 = (sim_ >= Sim_Min)
        #print("term_1:", x_last_p)
        x_in_shadow = x != x_last.right and x != x_last.left
        #print("x_in_shadow:", x_in_shadow)
        if term_1 or x_in_shadow:
            
            edge_f = int(x_last.data[0])
            edge_t = int(random.choice(x.data))
            
            edge_e = {'from':edge_f, 'to':edge_t}
            if edge_e not in for_topic_sample_edge_e:
                
                for_topic_sample_edge_e.append(edge_e)                    
                
                for_topic_sample_edge_f.append(edge_f)
                for_topic_sample_edge_t.append(edge_t)
                if x_in_shadow:
                    #Sim_special(x.key, x_last.key, )
                    for_topic_sample_edge_w.append(sim_)##test for now
                else:
                    for_topic_sample_edge_w.append(sim_)
                if x == x.parent.left:
                    ##x_p = copy.deepcopy(x_last_p)
                    ##x_p = x_p.append(0)
                    
                    if edge_t not in for_topic_sample_node_list:
                        for_topic_sample_node_list.append(edge_t)
                        ##for_topic_sample_node_list_n.append(x_p)
                        
                    FrontSample(T, x.right, x, x_standard, Sim_Min, x_p)
                    FrontSample(T, x.left, x, x_standard, Sim_Min, x_p)
                    for i in range(0,len(x.shadowconnect) ):
                        FrontSample(T, x.shadowconnect[i], x, x_standard, Sim_Min, x_p, i)
                elif x == x.parent.right:
                    #x_p = copy.deepcopy(x_last_p)
                    #x_p = x_p.append(1)
                    
                    if edge_t not in for_topic_sample_node_list:
                        for_topic_sample_node_list.append(edge_t)
                        ##for_topic_sample_node_list_n.append(x_p)
                        
                    FrontSample(T, x.right, x, x_standard, Sim_Min, x_p)
                    FrontSample(T, x.left, x, x_standard, Sim_Min, x_p)
                    for i in range(0,len(x.shadowconnect) ):
                        FrontSample(T, x.shadowconnect[i], x, x_standard, Sim_Min, x_p, i)
                else:
                    #x_p = copy.deepcopy(x_last_p)
                    #x_p = x_p.append(1 + shadow_num)
                    
                    if edge_t not in for_topic_sample_node_list:
                        for_topic_sample_node_list.append(edge_t)
                        ##for_topic_sample_node_list_n.append(x_p)
                        
                    FrontSample(T, x.right, x, x, Sim_Min, x_p)
                    FrontSample(T, x.left, x, x, Sim_Min, x_p)
        elif sim_ < Sim_Min:
            if target_node == None:
                target_node = []
            target_node.append(x_last.key)
    
##########
def Giant_Component_Sample(T, x, sample_num, max, min, sample_location = None, Up_or_Down = True):
    #->list
    ##Up_or_Down:True for Up while False for Down(for saving memory)
    ##only be used when some nodes be found in place close to two cleared(means have been changed to easy after some training) topic
    ##boundaries or an important node(have high hitrate and/or high )
    num = 0
    while num <= max:
        pass
        
##########
def TopicSample(T, x, max = None, min = None, sample_location = None, Begin = False): 
    ##For those x with high F2Sim to the query data_num
    ##rank by x.key['RD_n'] to the max-flow sample probability arrangment, Rd_n for num of
    # the representative_data in a node and I use a dict to map this num to the location for 
    #further 
    ##calculation  
    ####
    i = 0
    Sim_Min_ = 0.2##If the next node's key has similarity with the query node's key lower than this then break
    Sim_Min = 1##To get the smallest Sim so that we can restrict the follow requested nodes to
    ##have higher Sim to narrow down the scale
    
    x_ = sample_location
    
    
    while x_ != T.root and x_.parent != T.root:
        ##item Sim_Min_ set for now
        ##for it's quite complex using the global location I use the begins of the topic(maybe in
        # fact should be called topics for it may contains other topics connect by shadowconnect)
        ##to be the origin of the topics locations 
        Sim_ = Sim(x_.parent.key, x_.key)
        if Sim_ < Sim_Min_:
            break
        if Sim_ < Sim_Min:
            Sim_Min = Sim_
        x_ = x_.parent
        i += 1###on
        #Sim_Min = 0.5
    #if i == 0:
        #print("Sim_min:", Sim_Min)
     #   pass
        #return None
    #if True:
    x_standard = x
    x = x_
    x_last_p_ = [0]
    global for_topic_sample_edge_f 
    global for_topic_sample_edge_t 
    global for_topic_sample_edge_w 
    ##global for_topic_sample_edge_p
    global for_topic_sample_edge_e 
    global for_topic_sample_node_list 
    global for_topic_sample_node_list_n         
    ##Store num of each data instead of store places
    for_topic_sample_edge_f = []
    for_topic_sample_edge_t = []
    for_topic_sample_edge_w = []
    ##for_topic_sample_edge_p = []
    for_topic_sample_edge_e = []
    for_topic_sample_node_list = [x.data[0]]
    ####for_topic_sample_node_list_n = [0]
    print("test_0:", len(for_topic_sample_edge_e))
    FrontSample(T, x.right, x, x_standard, Sim_Min, x_last_p_)
    FrontSample(T, x.left, x, x_standard, Sim_Min, x_last_p_)
    for i in range(0,len(x.shadowconnect) ):
        if x.shadowconnect[i] != None:
            FrontSample(T, x.shadowconnect[i], x, x_standard, Sim_Min, x_last_p_, i)
    '''def KN_to_P(kn:int):
        ##Should be abandoned 
        ##Map key to place on the RB-tree
        try:
            kn_ = for_topic_sample_edge_t.index(kn)
            P = for_topic_sample_edge_p[kn_]['to_p']
        except:
            print("1")
            P = [0]
        return P'''
    
    for_topic_sample_edge_t.append(x_.data[0])
    ##for_topic_sample_edge_t_set = list(set(for_topic_sample_edge_t))
    def KN_to_Num(kn):
        return for_topic_sample_node_list.index(kn)
    def Num_to_KN(N):
        return for_topic_sample_edge_t[N]

    ##Input_to_MF = []
    print("test:", len(for_topic_sample_edge_e))
    #test = KN_to_Num(for_topic_sample_edge_f[0])####
    Input_to_MF_f = list(map(KN_to_Num, for_topic_sample_edge_f) )
    Input_to_MF_t = list(map(KN_to_Num, for_topic_sample_edge_t) )
    Input_to_MF_w = for_topic_sample_edge_w
    if len(Input_to_MF_w) == 1:
        Input_to_MF_w = torch.Tensor(Input_to_MF_w)
    print("test__:", Input_to_MF_w, "test___:", len(for_topic_sample_edge_f))
    ##print("s_of_Input_to_MF_w:", len(Input_to_MF_w) )
    
    Input_to_MF = (c_int*(3*len(for_topic_sample_edge_e) )  )()
    for i in range(0, len(for_topic_sample_edge_e) ):
        Input_to_MF[3*i] = (Input_to_MF_f[i])
        Input_to_MF[3*i + 1] = (Input_to_MF_t[i])
        Input_to_MF[3*i + 2] = (Input_to_MF_w)
        '''
        Input_to_MF.append(KN_to_Num(for_topic_sample_f[i]) )
        Input_to_MF.append(KN_to_Num(for_topic_sample_t[i]) )
        Input_to_MF.append(KN_to_Num(for_topic_sample_w[i]) )'''
        
        

    Num_of_nodes = len(for_topic_sample_node_list)
    Num_of_t = Num_of_nodes + 1
    
    ####用于向MF数据添加到度较大的节点的跳接
    '''for i in range(0, len(target_node) ):
        Input_to_MF.append(target_node[i])
        Input_to_MF.append(Num_of_t)
        Input_to_MF.append(10)
        Num_of_nodes += 1'''
    
    c_arr = (ctypes.c_int*128)()
    Num_of_edge = len(for_topic_sample_edge_e)
    adder.HLPP(Num_of_nodes, Num_of_edge, 0, Num_of_t, Input_to_MF, False, c_arr)
    print("testRS:", c_arr[0])
    #if len(Return_Sample) < min:
     #   pass
        ##here should be Giant_Component_Sample()
    #Return_Sample = map(K_to_P, map(Num_to_K(N), Return_Sample) )
    Return_ = []
    #for item in Return_Sample.new_list:
     #   Return_.append(data_list_[data_num_list[item]][1])
    return Return_
    

            


def TreeMinimum(x):
    s = time.perf_counter()
    while x.left != T.nil:
        x = x.left
    e = time.perf_counter()
    print('i:',e-s)
    return x
#中序遍历
def Midsort(x):
    if x != None:
        Midsort(x.left)
        if x.key!=0:
            if x == x.parent.left:
                print('key:', x.key,'x.parent',x.parent.key,'x.color:',x.color,'x_p:l_c')
            elif x == x.parent.right:
                print('key:', x.key,'x.parent',x.parent.key,'x.color:',x.color,'x_p:r_c')
            else:
                print('key:', x.key,'x.parent',x.parent.key,'x.color:',x.color,'x_p:root')
        Midsort(x.right)
##前序遍历
def Frontsort(x):
    if x!= None:
        
        if x.key!=0:
            if x == x.parent.left:
                print('key:', x.key,'x.parent',x.parent.key,'x.color:',x.color,'x_p:l_c')
            elif x == x.parent.right:
                print('key:', x.key,'x.parent',x.parent.key,'x.color:',x.color,'x_p:r_c')
            else:
                print('key:', x.key,'x.parent',x.parent.key,'x.color:',x.color,'x_p:root')
        Frontsort(x.left)
        Frontsort(x.right)
##后序遍历
def Lastsort(x):
    if x!= None:
        Lastsort(x.left)
        Lastsort(x.right)
        if x.key!=0:
            if x == x.parent.left:
                print('key:', x.key,'x.parent',x.parent.key,'x.color:',x.color,'x_p:l_c')
            elif x == x.parent.right:
                print('key:', x.key,'x.parent',x.parent.key,'x.color:',x.color,'x_p:r_c')
            else:
                print('key:', x.key,'x.parent',x.parent.key,'x.color:',x.color,'x_p:root')
        
'''nodes = [3,17,14,27,9,31,12,16,1,21,7,15,5,8,4,-1]
for node in nodes:
    if node == 8 or node == 17 or node == 15:
        print('插入数据',RBInsert(T,RBTreeNode(T, node, T.nil)))
    else:
        print('插入数据',RBInsert(T,RBTreeNode(node)))
'''

    

T_1 = RBTree()
T_2 = RBTree()
T_3 = RBTree()
T_4 = RBTree()
T_5 = RBTree()
Forest = [T_1, T_2]##, T_3, T_4, T_5]
def Free_func():
    for T in Forest:
        Free_PreOrder(T, T.root)
def Free_PreOrder(T, node):
    if (node != T.nil):
        lchild = node.left
        rchild = node.right
        for i in range(0, len(node.shadowconnect) ):
            schild = node.shadowconnect[i]
            Free_PreOrder(schild)
        del tree
        Free_PreOrder(lchild)
        Free_PreOrder(rchild)
import networkx as nx
G = nx.Graph()
'''def Frontsort_test(T, x, last_node = None, if_s = False):
    global G
    
    print("num_0:", len(list(G.nodes))
    if (x != T.nil):
        print(x.key)
        G.add_node(x.data_num[0])
        if last_node is not None:
            if if_s == False:
                if x.last_loss > 1:
                    G.add_edge(last_node.data_num[0], weight = 10*x.last_loss*Sim(last_node.key, x.key))
                else:
                    G.add_edge(last_node.data_num[0], weight = Sim(last_node.key, x.key))
            else:
                G.add_edge(last_node.data_num[0], weight = 10)
        
        Frontsort_test(T, x.left, x)
        Frontsort_test(T, x.right, x)
        for i in x.shadowconnect:
            Frontsort_test(T, x.shadowconnect[i], x, True)
def Frontsort_test_(T, x, last_node = None, if_s = False):
    global G_
    print("num:", len(list(G.nodes))
    if x != T.nil:
        
        G_.add_node(x.data_num[0])
        if last_node is not None:
            if if_s == False:
                if x.last_loss > 1:
                    if las_node.last_loss < 1:
                        G_.add_edge(last_node.data_num[0], weight = 10*x.last_loss*Sim(last_node.key, x.key))
                else:
                    G_.add_edge(last_node.data_num[0], weight = Sim(last_node.key, x.key))
            else:
                G_.add_edge(last_node.data_num[0], weight = 10)
        
        Frontsort_test_(T, x.left, x)
        Frontsort_test_(T, x.right, x)
        for i in x.shadowconnect:
            Frontsort_test_(x.shadowconnect[i], x, True)'''


def Insert_One_By_One(Node, Re_space, Re_space_last_node, In_tree, i_test = 0):
    ##Now Re_space mean a list of node_num
    data_list_output_ = []
    data_list_output = []
    for node_ in Node:
        i_test += 1####
        node = RBTreeNode(node_[0], node_[1]['x'][0][0].item(), node_[2], node_[3], node_[4])
        sim_s = []##for one node's sim to each places we've found suitable to it 
        
        sim_t = []##for tree num of each inserted nodes
        sim_ = []
        done_ = False##mean whether we've got place to insert our node or not
        if Forest[0].root.y == None:
            done_ = True
            #print("node_0:", node_[0])
            Forest[0].root.Key = node_[0]
            #print("Forest[0].root.Key:", node_[0])
            Forest[0].root.data_num = [node_[1].x[0][0]]
            Forest[0].root.y = node_[2]
            Forest[0].root.last_lost = node_[3]
            Forest[0].root.right = Forest[0].nil
            Forest[0].root.left = Forest[0].nil
            query_num = node_[1].x[0][0]
            if query_num in Re_space:
                last_node = Re_space_last_node[Re_space.index(query)]
                last_node.shadowconnect.append(T.root)
                T.root.shadowconnect.append(last_node)
        for i in range(0, len(Forest) ):
            if done_:
                break
            ####if node.Last_Time < 10 and i > 0:
             ####   break
            T = Forest[i]
            
            if T.root.y == None and len(sim_) == 0:
                done_ = True
                T.root = node
                T.root.right = T.nil
                T.root.left = T.nil
                #T.root.Key = node_[0]
                #T.root.data_num = [node_[1].x[0][0]]
                #T.root.y = node_[2]
                #T.root.last_lost = node_[3]
                
                query_num = node_[1].x[0][0]
                if query_num in Re_space:
                    last_node = Re_space_last_node[Re_space.index(query)]
                    last_node.shadowconnect.append(T.root)
                    T.root.shadowconnect.append(last_node)
                break
            
            elif T.root.y == None or i == 1:
                if len(sim_s) != 0 or True:
                    #Sim(T.root.key, node.key) >= 0.5:
                    down = True
                    sim_1 = max(sim_s)
                    sim_s.remove(sim_1)
                    In_tree.append(sim_t[0])
                    #node = RBTreeNode(node_[0], node_[1], node_[2], node_[3], node_[4])
                    possible_branch, if_in_new_p = RBInsert(T, Forest.index(T), node, Last_node = None, if_insert = False)
                    ####test
                    if possible_branch.parent == None and possible_branch.right == None and possible_branch.left == None:
                        if possible_branch != T.root:
                            print("pb_p:")
                            print(possible_branch.key)
                    
                    if i_test > 5: 
                        data_list_output_ = TopicSample(T, node, sample_location = possible_branch, Begin = True)
                    ####
                        
                    if data_list_output_ != None:
                        data_list_output.extend(data_list_output_)
                    #if possible_branch.right == T.nil and possible_branch.left == T.nil:
                     #   possible_branch = None
                    if len(sim_s):
                        sim_2 = max(sim_)
                        In_tree.append(sim_t[1])
                        sim_try, if_in_new_p = RBInsert(T, Forest.index(T), node, if_insert = True)##Insert the node Here
                        query_num = node_[1].x[0][0]
                        if query_num in Re_space:
                            last_node = Re_space_last_node[Re_space.index(query)]
                            last_node.shadowconnect.append(sim_try)
                            sim_try.shadowconnect.append(last_node)

                        if possible_branch != None:
                            if Sim(sim_try.key, possible_branch.key) <= 0.1:
                                sim_try.shadowconnect.append(possible_branch)
                                possible_branch.shadowconnect.append(sim_try)
                    else:
                        sim_try, if_in_new_p = RBInsert(T, Forest.index(T), node, if_insert = True)##Insert the node Here
                elif T.root.y == None:
                    ##the same as insert the first node
                    done_ = True
                    print("node_0:", node_[0])
                    Forest[0].root.Key = node_[0]
                    print("Forest[0].root.Key:", node_[0])
                    Forest[0].root.data_num = [node_[1].x[0][0]]
                    Forest[0].root.y = node_[2]
                    Forest[0].root.last_lost = node_[3]
                    query_num = node_[1].x[0][0]
                    if query_num in Re_space:
                        last_node = Re_space_last_node[Re_space.index(query)]
                        last_node.shadowconnect.append(T.root)
                        T.root.shadowconnect.append(last_node)
            else:
                sim__ = 1#Sim(T.root.key, node_[0])
                if sim__ >= 0.2:
                    sim_s.append(1)
                    sim_t.append(T)
                    
        if done_ == False:
            i = 0
            #node = RBTreeNode(node_[0], node_[1], node_[2], node_[3], node_[4])
            while done_ == False and i <= 4: 
                if BeginPoint[i] != None: 
                    for point in BeginPoint[i]:
                        if Sim(node_.key, point.key) < 0.1:

                            query_node_p = RBInsert(Forest[0], 0, node, if_insert = True)
                            query_num = node_[1].x[0][0]
                            if query_num in Re_space:
                                last_n = Re_space_last_node[Re_space.index(query_num)] 
                                last_n.shadowconnect.append(query_node)
                                query_node_p.shadowconnect.append(last_n)

                            data_list_.append(node_[i])
                            data_num_list.append(node[1].x[0][0])
                            done_ = True
                            data_list_output_ = TopicSample(T, node, sample_location = query_node_p, Begin = True)
                            data_list_output.extend(data_list_output_)
                            break
                i += 1
            key_list.append(node.key)
            data_num_list.append(node.data[0])       
    return data_list_output       
def test_main(Node, Node_num = None, Re_space = None):
    ####data_list_review, AnchorList, VentureList, BaseLoss = test_main(Node, Node_num)
    ##Re_space mean the num of those nodes inside it already in the tree and we are not just gonna remove it or replace it but first recalculate the topic sample of it then move their place if needed
    done_ = True
    In_tree = []
    #n = Node[0][0]
    #print(n)
    #Node = 
    data_list_output_ = []
    
    Re_space_ = []
    Re_space_last_node = []
    
    if Re_space != None: 
        for node_ in Re_space:
            node_before = data_list_[data_num_list.index(node_[1].x[0][0]) ]
            node_before.last_loss = node_[3]
            node = RBTreeNode(node_[0], node_[1].x[0][0], node_[2], node_[3], node_[4])
            if sim(node_before.x[0][0], node_[0]) < 0.5:
                if node_before.parent != None:
                    if node_[4] - node_before.Last_Time > 5:
                        ##only after 5 batches change the place and key of a node so as to enhance the stability of the model training process
                        #RBDelete(forest[data_T[data_num_list[node_] ] ], node_before)
                        if node_before.loss_ > 1:
                            Re_space_.append(node_[1].x[0][0])
                            Re_space_last_node.append(node_.parent)
    data_list_output = Insert_One_By_One(Node, Re_space_, Re_space_last_node, In_tree)##on
    print("data_list_output", len(data_list_output) )
    return data_list_output
    if T_1.root.key != None:
        print(T_1.nil.parent.right == T_1.nil)
    #Frontsort_test(Forest[0], Forest[0].root)
    #print("result:", nx.average_clustering(G))
    #print("result_1:", nx.average_shortest_path_length(G))
    G_ = nx.Graph()
    #Frontsort_test_(Forest[0], Forest[0].root)
    #print("result_2:", nx.number_connected_components(G))
        #TopicSample(T, x, max, min, sample_location = None, Begin = False): ->list
