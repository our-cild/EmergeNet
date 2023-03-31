// 编译命令 g++ -o libtryPython.so -shared -fPIC tryPython.cpp
#include<iostream>
#include <cstdio>
#include <cstring>
#include <queue>
#include <stack>

#include <random>

#include <list>
using namespace std;
extern "C"{
    
    bool E_or_C = 0;//if go on explore first way or check what not considered way 0 for the latter and vice verse
    const int N = 1200, M = 120000, INF = 0x3f3f3f3f;
    int n, m, s, t;
    //random_device rd;
    struct edge {
      int nex, t, v, t_h_e, if_f;
      //t_h_e for the hardship of each node, positive for the hardship while negative for how easy the node is, we can judge easy nodes as a 
      //net ladder for the model to learn the foundation knowledges and skills to get to those hard ones and became more generalizable.Here 
      //I decide to let the training progress fancy more to the easy ones at begining to check the fondations and add to the momentum for if       
      //we've got enough easy nodes as ladder we may have more confident to get to those hard ones, maybe some of them are close to the           //ones.
      //And if after one query progress the trained model get better to the queried specimen and over a thereshold, the followed query          
      //progress would try a venturesome way, which means it would give more momentum to the easy ones and ignore some easy ones passed with       
      //high accuracy and give rate to directly get to those hard ones. If those easy ones passed, the update progress would goes on, and it       //would stop vice verse 
    };
    
    struct ex_ {
      int ex = 0;
      int momentum = 0;
      int flow = 0;
    };

    edge e[M * 2 + 1];
    int h[N + 1], cnt = 1;
    stack<int> G;////G for those node found meet the requirement in the process

    void add_path(int f, int t, int v, int f_h_e, bool if_f) { 
        //printf("\nf:%d,h[f]:%d\n",f,h[f]);
        e[++cnt] = (edge){h[f], t, v, f_h_e, if_f};
        h[f] = cnt; 
        //printf("\nf:%d,h[f]:%d\n",f,h[f]);
    }

    void add_flow(int f, int t, int v, int f_h_e, int t_h_e) {
      add_path(f, t, v, t_h_e, 1);
      add_path(t, f, 0, f_h_e, 0);
    }

    int ht[N + 1];
    ex_ ex[N + 1];
    int gap[N];       // 高度; 超额流和动量以及流量; gap 优化 gap[i] 为高度为 i 的节点的数量
    stack<int> B[N];  // 桶 B[i] 中记录所有 ht[v]==i 的v
    int level = 0;    // 溢出节点的最高高度

    int Push(int u) {      // 尽可能通过能够推送的边推送超额流
      bool init = u == s;  // 是否在初始化
      //int mt = 0;
      for (int i = h[u]; i; i = e[i].nex) {
        const int &v = e[i].t, &w = e[i].v;
        if (!w || init == false && ht[u] != ht[v] + 1)  // 初始化时不考虑高度差为1
            continue;
        int k = init ? w : min(w, ex[u].ex);
        // 取到剩余容量和超额流的最小值，初始化时可以使源的溢出量为负数。
        if (v != s && v != t && !ex[v].ex) B[ht[v]].push(v), level = max(level, ht[v]);
        if (e[i].if_f){
            ex[u].momentum += e[i].t_h_e;
            ex[v].momentum += ex[u].momentum;
            ex[u].ex -= k, ex[v].ex += (1+ex[u].momentum) * k, e[i].v -= k, e[i ^ 1].v += (1+ex[u].momentum) * k;  // push 
            ex[v].flow += (1+ex[u].momentum) *k;
            if (ex[v].flow >= 50){
                int v_ = v;
                G.push(v_);
            }  
            else{
                time_t t;
                
                //srand((unsigned) time(&t));/* 初始化随机数发生器 */
                if ((5 % (10-1+1))+ 1 > 9){
                    //?
                    //To get a random return num in [1, 10]
                    int v_ = v;
                    G.push(v_);
                }
            }
        }
        else{
            if (E_or_C){
                int v_ = v;
                ex[v].flow -= k;
                if (ex[v].flow < 50 && ex[v_].flow > 50){                    
                    if (G.top() == v){
                        G.pop();
                    }
                }
            }
            ex[u].ex -= k, ex[v].ex += k , e[i].v -= k, e[i ^ 1].v += k;  // push 
        }
        if (ex[u].ex + ex[u].momentum <= 0) return 0;  // 如果已经推送完就返回
      }
      return 1;
    }

    void relabel(int u) {  // 重贴标签（高度）
      ht[u] = INF;
      for (int i = h[u]; i; i = e[i].nex)
        if (e[i].v) ht[u] = min(ht[u], ht[e[i].t]);
      if (++ht[u] < n) {  // 只处理高度小于 n 的节点
        B[ht[u]].push(u);
        level = max(level, ht[u]);
        ++gap[ht[u]];  // 新的高度，更新 gap
      }
    }

    bool bfs_init() {
      memset(ht, 0x3f, sizeof(ht));
      queue<int> q;
      q.push(t), ht[t] = 0;
      while (q.size()) {  // 反向 BFS, 遇到没有访问过的结点就入队
        int u = q.front();
        q.pop();
        for (int i = h[u]; i; i = e[i].nex) {
          const int &v = e[i].t;
          if (ex[u].momentum){
              ex[u].momentum += 1/2*e[u].t_h_e;
          }
          ex[v].momentum += 1/4*ex[u].momentum;
          if (e[i ^ 1].v && ht[v] > ht[u] + 1) ht[v] = ht[u] + 1, q.push(v);
        }
      }
      return ht[s] != INF;  // 如果图不连通，返回 0
    }

    // 选出当前高度最大的节点之一, 如果已经没有溢出节点返回 0
    int Select_() {
      while (B[level].size() == 0 && level > -1) level--;
      return level == -1 ? 0 : B[level].top();
    }
    
    typedef struct StructPointer
    {   //just for pass parameter to the python code for it seems can't convert an array to python code
        list<int> new_list = {};
        int num = 0;
        bool operator==(const StructPointer &p)
        {
            return (this->num == p.num);
        }
    } s_test ;
    
    int hlpp(bool conservatism_or_adventurism, int *return__ ) {                  // 返回最大流
      //int* return_ = new int [200];
      //StructPointer *return_;
      if (!bfs_init()) return 0;  // 图不连通
      memset(gap, 0, sizeof(gap));
      for (int i = 1; i <= n; i++)
        if (ht[i] != INF) gap[ht[i]]++;  // 初始化 gap
      ht[s] = n;
      Push(s);  // 初始化预流
      int u;
      while ((u = Select_())) {
        B[level].pop();
        if (Push(u)) {  // 仍然溢出
          if (!--gap[ht[u]]){
            for (int i = 1; i <= n; i++){
              if (i != s && i != t && ht[i] > ht[u] && ht[i] < n + 1){
                ht[i] = n + 1;  // 这里重贴成 n+1 的节点都不是溢出节点
              }
            }
          }
          relabel(u);
        }
      }
      //StructPointer p = (StructPointer)malloc(200*sizeof(StructPointerTest)); 
      int rand_ = 5;
      int i_term = 0;//Just for convert stack<int> to int
      int i_term_ = 0;
      time_t t;          
      //srand((unsigned) time(&t));/* 初始化随机数发生器 */
      for (int i = 0; i <= G.size() && i < 127;i++){
          
          //rand_ = rand() % (10 - 0 + 1) + 0;//to get a random return num in [0, 10]
          i_term = G.top();
          G.pop();
          i_term_ = ex[i_term].ex;
          if (rand_* i_term_ > 5){
              ////return__.new_list.append(G[i]);
              return__[i] = i_term;
          //return_[i]->new_list[i] = G[i];
          }
      }
      return 1;
      //return ex[t].ex;
      //return p
    }
    //Num_of_nodes, len(for_topic_sample), 0, Num_of_t,b_arr
    void HLPP(int n_, int m_, int s_, int t_, int *Input_to_MF, bool conservatism_or_adventurism, int *Return_) {
      n = n_;
      m = m_;
      s = s_;
      t = t_;
      int u, v, w;
      int min = 0;
      int max = 100;
      /*
      for (int i = 0;i < 10;i++){
          // 产生10个随机数
          for(int i = 0; i < 10; i++) {
              int q = rd()%(max - min) + min;
              //printf("\nq:%d\n",q);
          }
          
      }*/
      
      //scanf("%d%d%d%d", &n, &m, &s, &t);
      for (int i = 0; i < m_ - 1; i++) {
        //scanf("%d%d%d", &u, &v, &w);
        u = Input_to_MF[3*i];
        v = *(Input_to_MF + 3*i + 1);
        w = *(Input_to_MF + 3*i + 2);
        printf("\ntest_f:%d", u);
        printf("test_t:%d\n",v);
        add_flow(u, v, w, 0, 0);
      }
      
      hlpp(false, Return_);
      //list<int>new_list;
      //return__.append(1);
      //##
      for(int i=0;i<3;i++)
      {
          Return_[i]++;
      }

      //##
      //return return_;
      //printf("%d\n", hlpp(false, return_));
    }
    /*
    int main() {
      scanf("%d%d%d%d", &n, &m, &s, &t);
      for (int i = 1, u, v, w; i <= m; i++) {
        scanf("%d%d%d", &u, &v, &w);
        add_flow(u, v, w, 0, 0);
      }
      printf("%d", hlpp(false, return_));
      return 0;
    }
*/


}
