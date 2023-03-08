import time
#import torch.nn.Functional as F
import torch

from ctypes import *
adder = CDLL("./adder_5.so")
data_list_ = []
data_num_list = []
#定义红黑树
class RBTree(object):
    def __init__(self):
        self.nil = RBTreeNode(torch.zeros(1), None, None, None)
        self.root = self.nil
        self.HitRate = 0
        #self.level = 0##the num of the first level is 0
        self.n_hard = []
class RBTreeNode(object):
    def __init__(self, key, data_num, data_y, last_loss):
        self.key = key ##key for feature deduced by the model
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
    elif Sim(x, x.right) < 0.2 or Sim(x, x.left) < 0.2:
        x.in_Topic = True
        x.Topic_Root = True
##assume for F2Sim ones
def Sim(x_1: torch.Tensor, x_2: torch.Tensor):
    Sim_dist = torch.norm(x_1 - x_2, p=2)
    return Sim_dist.clamp(min=1e-12).sqrt()  # for numerical stability
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
    ##
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
    ##
    x.right = y.left
    if y.left != T.nil:
        y.left.parent = x
    y.parent = x.parent
    if x.parent == T.nil:
        T.root = y
    ##
    elif x.parent.Seed_Root == True:
        y.Seed_Root = True
        x.parent.Seed_Root = False
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
    ##
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
    ##
    x.left = y.right
    if y.right != T.nil:
        y.right.parent = x
    y.parent = x.parent
    if x.parent == T.nil:
        T.root = y
    ##
    elif x.parent.Seed_Root == True:
        y.Seed_Root = True
        x.parent.Seed_Root = False
    ##
    elif x == x.parent.right:
        x.parent.right = y
    else:
        x.parent.left = y
    y.right = x
    x.parent = y
    
BeginPoint = [None]*5 ## for those node which have more degree_num and/or higher hitrate 
#红黑树的插入
def RBInsert(T, T_num, z, Last_node = None, if_hard = False, if_insert = True, Return_node = False):
    ##In fact this fuction supposed to be insert or find the position. But in name of RBInsert would make readers know where it origins from the normal RBTree functions  
    
    ####
    if Last_node is None:
        y = T.nil
        x = T.root
        max_sim = 0##
        stop = True##
        if BeginPoint[T_num] != None:
            for node in BeginPoint[T_num]:
                sim_ = Sim(z.key, node.key)
                if sim_ <= 0.1:
                    x = node
                    max_sim = sim_
        while x != T.nil:
            stop = True
            y = x
            sim_candidate = [];
            if x.left != T.nil:
                sim_l = Sim(z.key, x.left.key)
                if sim_l > max_sim:
                    x = x.left
                    max_sim = sim_1
                    stop = False               
            if x.right != T.nil:
                sim_r = Sim(z.key, x.right.key)
                if sim_r > max_sim:
                    x = x.right
                    max_sim = sim_r
                    stop = False
            if stop:
                break
        y = x
        z.parent = y
        y.HitRate += 1
        if y.HitRate + len(y.shadowconnect) > 10 and y not in BeginPoint:
            BeginPoint.append(y)
        if if_insert == False:
            return y
        
        if y == T.nil:
            T.root = z
            '''
            elif max_sim > 0.8:
            if z.y == z.parent.y:
                y.data.append(z.data_num[0])'''
        
        else:
            Last_node = y
    if Last_node is not None and Last_node.left is not None:
        if Last_node.last_loss < 0.1 and z.last_loss < 1:
            if Sim_special(Last_node.key, z.key, Last_node.left.key) < 0.5:
                for i in range(0, len(z.data_num)):
                    Last_node.data_num.append(z.data_num[i])
                return y
            #if Last_node.right is T.nil or Last_node.right.last_loss < 0.1:
                
        if Last_node.color == 'black':
            if Last_node.left.color == 'black':
                z.left = Last_node.left
                Last_node.left.parent = z
                Last_node.left = z
                z.parent = Last_node
                z.Seed_Begin = True
                z.color = 'red'
            else:
                z.left = Last_node.left
                Last_node.left.parent = z
                Last_node.left.color = 'black'
                Last_node.left = z
                z.parent = Last_node
                z.Seed_Begin = True
                z.color = 'red'
                if z.left.left is not T.nil:
                    z.left.left.color = 'red'
                if z.left.right is not T.nil:
                    z.left.right.color = 'red'
        else:
            if Last_node.left.left.color == 'black' and Last_node.left.right.color == 'black':
                z.left = Last_node.left
                Last_node.left.parent = z
                Last_node.left = z
                z.parent = Last_node
                z.color = 'black'
                z.left.color = 'red'
            elif Last_node.left.left.color == 'red' and Last_node.left.right.color == 'black':
                Last_node.left.color = 'red'
                LeftRotate(T, Last_node.left)                
                z.left = Last_node.left
                Last_node.left.parent = z
                Last_node.left = z
                z.parent = Last_node
                z.color = 'black'
                z.right = z.left.right.right
                z.left.right.right = T.nil
            elif Last_node.left.right.color == 'red' and Last_node.left.left.color == 'black':
                Last_node.left.color = 'red'
                RightRotate(T, Last_node.left)                
                z.left = Last_node.left
                Last_node.left.parent = z
                Last_node.left = z
                z.color = 'black'
                z.right = z.left.left.left
                z.left.left.left = T.nil
            elif Last_node.left.right.color == 'red' and Last_node.left.left.color == 'red':
                if Last_node.parent is not T.nil:
                    if Last_node.parent.left == Last_node:
                        Last_node.parent.left = z
                    else:
                        Last_node.parent.right = z
                else:
                    T.root = z
                z.left = Last_node.left
                z.right = Last_node
                Last_node.parent = z
                Last_node.color = 'black'
                if Last_node.right != T.nil:
                    Last_node.right.color = 'red'
                z.color = 'red'
                ###
    elif Last_node is not None:
        if Last_node.last_loss < 0.1 and z.last_loss > 1:
            if Last_node.right is not None:
                if Sim_special(Last_node.key, z.key, Last_node.right.key) < 0.5:
                    for i in range(0, len(z.data_num)):
                        Last_node.data_num.append(z.data_num[i])
                        data_num.append(z.data_num)
                        data_list_.append(z_data) 
                    return y
        z.parent = Last_node
        Last_node.left = z
        z.left = T.nil
        z.right = T.nil
        z.color = 'red'
        RBInsertFixup(T, z)
    
    if if_hard == True:
        Move_hard( T, z)

    return y##z.key, '颜色为', z.color
    
#红黑树的上色
def RBInsertFixup( T, z):        
    while z.parent.color == 'red':
        if z.parent == z.parent.parent.left:
            y = z.parent.parent.right
            if y.color == 'red':
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
def FrontSample(T, x, x_last, x_standard, Sim_Min, x_last_p, shadow_num = None):
    ##on
    ##x_last_p is in the list way
    if x != T.nil:        
        sim_ = Sim(x, x_last)
        if sim_ >= Sim_Min or (x != x_last.right and x != x_last.left):
            edge_f = x_last.key
            edge_t = x.key
            if edge_f not in for_topic_sample_edge_f or edge_t not in for_topic_sample_edge_t:
                for_topic_sample_edge_f.append(edge_f)
                for_topic_sample_edge_t.append(edge_t)
                for_topic_sample_edge_w.append(10*sim_)
                if x == x.parent.left:
                    x_p = x_last_p.copy()
                    x_p = x_p.append(0)
                    edge__ = {'from_p':x_last_p, 'to_p':x_p}
                    for_topic_sample.append(edge__)
                    FrontSample(T, x.right, x, x_standard, Sim_Min, x_p)
                    FrontSample(T, x.left, x, x_standard, Sim_Min, x_p)
                    for i in range(0,len(x.shadowconnect) ):
                        FrontSample(T, x.shadowconnect[i], x, x_standard, Sim_Min, x_p, i)
                elif x == x.parent.right:
                    x_p = x_last_p.copy()
                    x_p = x_p.append(1)
                    edge__ = {'from_p':x_last_p, 'to_p':x_p}
                    for_topic_sample.append(edge__)
                    FrontSample(T, x.right, x, x_standard, Sim_Min, x_p, target_node)
                    FrontSample(T, x.left, x, x_standard, Sim_Min, x_p, target_node)
                    for i in range(0,len(x.shadowconnect) ):
                        FrontSample(T, x.shadowconnect[i], x, x_standard, Sim_Min, x_p, target_node)
                else:
                    x_p = x_last_p.copy()
                    x_p = x_p.append(1 + shadow_num)
                    edge__ = {'from_p':x_last_p, 'to_p':x_p}
                    for_topic_sample.append(edge__)
                    FrontSample(T, x.right, x, x, Sim_Min, x_p, target_node)
                    FrontSample(T, x.left, x, x, Sim_Min, x_p, target_node)
        elif sim_ < Sim_Min:
            target_node.append(x_last.key)
##########
def Giant_Component_Sample(T, x, sample_num, max, min, sample_location = None, Up_or_Down = True):
    #->list
    ##Up_or_Down:True for Up while False for Down(for saving memory)
    ##only be used when some nodes be found in place close to two cleared(means have been changed to easy after some training) topic
    ##boundaries or an important node(have high hitrate and/or high )
    num = 0
    while num <= max:
        
##########
def TopicSample(T, x, max, min, sample_location = None, Begin = False): 
    ##For those x with high F2Sim to the query data_num
    ##rank by x.key['RD_n'] to the max-flow sample probability arrangment, Rd_n for num of
    # the representative_data in a node and I use a dict to map this num to the location for further 
    ##calculation  
    ####
    i = 0
    Sim_Min_ = 0.2
    Sim_Min = 1
    while x_.parent != T.root:
        ##item Sim_Min_ set for now
        ##for it's quite complex using the global location I use the begins of the topic(maybe in
        # fact should be called topics for it may contains other topics connect by shadowconnect)
        ##to be the origin of the topics locations 
        Sim_ = Sim(x_.parent, x_)
        if Sim_ < Sim_Min_:
            break
        if Sim_ < Sim_Min:
            Sim_Min = Sim_
        x_ = x_.parent
        i += 1
    if i == 0:
        return None
    else:
        x = x_
        x_last_p_ = [0]

        FrontSample(T, x.right, x, x_standard, Sim_Min, x_last_p_, target_node)
        FrontSample(T, x.left, x, x_standard, Sim_Min, x_last_p_, target_node)
        for i in range(0,len(x.shadowconnect) ):
            FrontSample(T, x.shadowconnect[i], x, x_standard, Sim_Min, x_last_p_, target_node)
    def K_to_P(k):
        try:
            k_ = for_topic_sample_edge_t.index(k)
            P = for_topic_sample[k_]['to_p']
        except:
            P = [0]
        return P
    ##K_to_Num(x_now):
    for_topic_sample_edge_t.append(x_.key)
    
    def K_to_Num(k):
        return for_topic_sample_edge_t.index(k)
    def Num_to_K(N):
        return for_topic_sample_edge_t[N]

    Input_to_MF = []
    for i in range(0, len(for_topic_sample) ):
        Input_to_MF.append(K_to_Num(for_topic_sample_f[i]) )
        Input_to_MF.append(K_to_Num(for_topic_sample_t[i]) )
        Input_to_MF.append(K_to_Num(for_topic_sample_w[i]) )

    Num_of_nodes_ = list(set(map(K_to_Num, for_topic_sample_edge_t) ) )
    Num_of_nodes = len(Num_of_nodes_)
    Num_of_t = Num_of_nodes + 1
    for i in range(0, len(target_node) ):
        Input_to_MF.append(target_node[i])
        Input_to_MF.append(Num_of_t)
        Input_to_MF.append(10)
        Num_of_nodes += 1
    ####
    b_arr = (c_int*(len(for_topic_sample) ) )(*Input_to_MF)
    Return_Sample = adder.HLPP(Num_of_nodes, len(for_topic_sample), 0, Num_of_t,b_arr)
    if len(Return_Sample) < min:
        pass
        ##here should be Giant_Component_Sample()
    #Return_Sample = map(K_to_P, map(Num_to_K(N), Return_Sample) )
    Return_ = []
    for item in Return_Sample:
        Return_.append(data_list_[data_num_list[item]][1])
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
Forest = [T_1, T_2, T_3, T_4, T_5]
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

        
def test_main(Node, Node_num = None, Re_space):
    ##Re_space mean the nodes inside it already in the tree and we are not gonna remove it or replace it but just recalculate the topic sample of it 
    done_ = True
    In_tree = []
    n = Node[0][0]
    #print(n)
    #Node = 
    data_list_output_ = []
    data_shadowconnect_to_beforeParent = []
    for node_ in Re_space:
        node_before = data_list_[data_num_list[node_]]
        node = RBTreeNode(node_[0], node_[1].x[0][0], node_[2], node_[3])
        if sim(node_before.x[0][0]][0], node_[0]) < 0.5:
            if node_before.parent != None:
                
                RBDelete(forest[data_T[data_num_list[node_] ] ], node_before)
                if node_before.loss_
                node_before.parent.shadowconnect.append()
    for node_ in Node:
        sim_s = []
        sim_t = []
        for T in Forest:
            ####
            '''node = RBTreeNode(node_[0], node_[1], node_[2], node_[3])
            possible_branch = RBInsert(T, Forest.index(T), RBTreeNode(node_[0], node_[1], node_[2], node_[3]), if_insert = False )
            sim_try = RBInsert(T, Forest.index(T), RBTreeNode(node_[0], node_[1], node_[2], node_[3]), if_insert = False)
            sim_try.shadowconnect.append(possible_branch)
            test_ = RBInsert(T, Forest.index(T), RBTreeNode(node_[0], node_[1], node_[2], node_[3]), if_insert = False)
            if test_.shadowconnect[len(test_.shadowconnect) - 1] == possible_branch:
                print("yes_")
            else:
                print("no_")'''
            
            ####
            if Forest[0].root == None:
                T.root.Key = node_[0]
                #print("t?:", type(T.root.Key))
                T.root.data_num = [node_[1].x[0][0]]
                T.root.y = node_[2]
                T.root.last_lost = node_[3]
            elif T.root == None and len(sim_) == 0:
                T.root.Key = node_[0]
                T.root.data_num = [node_[1].x[0][0]]
                T.root.y = node_[2]
                T.root.last_lost = node_[3]
                done_ = False
                break
            
            elif T.root == None:
                sim_1 = max(sim_s)
                sim_s.remove(sim_1)
                In_tree.append(sim_t[index(max(sim_s) )])
                node = RBTreeNode(node_[0], node_[1].x[0][0], node_[2], node_[3])
                possible_branch = RBInsert(T, Forest.index(T), node)
                data_list_output_ = TopicSample(T, node, Begin = True)
                data_list_output.extend(data_list_output)
                if possible_branch.right == T.nil and possible_branch.left == T.nil:
                    possible_branch = None
                if len(sim_):
                    sim_2 = max(sim_)
                    In_tree.append(sim_t[index(max(sim_s) )])
                    sim_try = RBInsert(T, RBTreeNode(node), if_insert = False)
                    if possible_branch != None:
                        if Sim(sim_try.key, possible_branch.key) <= 0.1:
                            sim_try.shadowconnect.append(possible_branch)
                            possible_branch.shadowconnect.append(sim_try)
            else:
                sim__ = 1#Sim(T.root.key, node_[0])
                if sim__ <= 0.2:
                    sim_s.apppend(sim__)
                    sim_t.append(T)
                    
        if done_ == False:
            i = 0
            while done_ == False and i <= lenlen(5):                
                for point in BeginPoint[i]:
                    if Sim(node_.key, point.key) < 0.1:
                        RBInsert(Forest[0], 0, RBTreeNode(node) )
                        data_list_.append(node_[i])
                        data_num_list.append(node_[i].x[0][0])
                        done_ = True
                        data_list_output_ = TopicSample(T, node, Begin = True)
                        data_list_output.extend(data_list_output)
                        break
                i += 1
            data_list_.append(node_[i])
            data_num_list.append(node_[i].x[0][0])       
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
