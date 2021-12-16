/*
Copyright (c) 2020 Software Platform Lab
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
3. Neither the name of the Software Platform Lab nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

#include <torch/csrc/jit/passes/utils/DAGDecomposition.h>

#include <utility>
#include <algorithm>
#include <iostream>
#include <stack>

namespace DAGDecomposition {

void print_dag(const Graph& dag) {
  int num_nodes = dag.size();
  for (int i = 0; i < num_nodes; i++) {
    std::cout << i << " -> ";
    const auto& child_nodes = dag.at(i);
    for (auto child_id : child_nodes) {
      std::cout << child_id << ", ";
    }
    std::cout << std::endl;
  }
}


void printVec(std::vector<int> v){
    for(int i=0;i<v.size();i++){
        std::cout<<v[i]<<" ";
    }
    std::cout<<std::endl;
}


void dfs(int start, const Graph& graph, std::vector<bool>& visited) {
  visited.at(start) = true;
  const std::vector<int>& adjacent_vertices = graph.at(start);
  for (auto vertex : adjacent_vertices) {
    if (!visited.at(vertex)) {
      dfs(vertex, graph, visited);
    }
  }
}

std::vector<std::vector<bool>> get_transitive_closure(const Graph& dag) {
  std::vector<std::vector<bool>> transitive_closure;
  int num_nodes = dag.size();
  for (int i = 0; i < num_nodes; i++) {
    std::vector<bool> reachable(num_nodes, false);
    DAGDecomposition::dfs(i, dag, reachable);
    reachable.at(i) = false;
    transitive_closure.push_back(std::move(reachable));
  }
  return transitive_closure;
}

Graph get_MEG(const Graph& dag) {
  const auto& transitive_closure = get_transitive_closure(dag);
  int num_nodes = dag.size();
  DAGDecomposition::Graph meg = dag;
  for (int i = 0; i < num_nodes; i++) {
    auto& meg_child_nodes = meg.at(i);
    const auto& child_nodes = dag.at(i);
    for (const auto child : child_nodes) {
      if (std::find(meg_child_nodes.begin(), meg_child_nodes.end(), child) == meg_child_nodes.end()) {
        continue;
      }
      for (const auto another_child : child_nodes) {
        if (transitive_closure.at(child).at(another_child)) {
          auto it = std::find(meg_child_nodes.begin(), meg_child_nodes.end(), another_child);
          if (it != meg_child_nodes.end()) {
            meg_child_nodes.erase(it);
          }
        }
      }
    }
  }
  return meg;
}

Bigraph meg_to_bigraph(const Graph& meg) {
  Bigraph bigraph;
  int num_vertices = meg.size();
  for (int i = 0; i < num_vertices; i++) {
    std::vector<bool> adjacency(num_vertices, false);
    for (auto child : meg.at(i)) {
      adjacency.at(child) = true;
    }
    bigraph.push_back(std::move(adjacency));
  }
  return bigraph;
}

Bigraph dag_to_bigraph(const Graph& dag) {
  Bigraph closure;
  int num_vertices = dag.size();
  for (int i = 0; i < num_vertices; i++) {
    std::vector<bool> reachable(num_vertices, false);
    dfs(i, dag, reachable);
    reachable.at(i) = false;
    closure.push_back(std::move(reachable));
  }
  return closure;
}

bool find_matching(int start, const Bigraph& graph, std::vector<bool>& seen, std::vector<int>& match_status) {
  int num_b = graph.at(0).size();
  for (int i = 0; i < num_b; i++) {
    if (graph.at(start).at(i) && !seen.at(i)) {
      seen.at(i) = true;
      int curr_match = match_status.at(i);
      if (match_status.at(i) == -1 || find_matching(curr_match, graph, seen, match_status)) {
        match_status.at(i) = start;
        return true;
      }
    }
  }
  return false;
}

std::vector<int> maximum_matching(const Bigraph& graph) {
  int num_b = graph.at(0).size();
  std::vector<int> match_result(num_b, -1);
  int num_a = graph.size();
  for (int i = 0; i < num_a; i++) {
    std::vector<bool> seen(num_b, false);
    find_matching(i, graph, seen, match_result);
  }
  return match_result;
}

std::tuple<std::vector<int>, std::vector<std::array<int, 2>>, int> get_mapping(const std::vector<int>& matching_BtoA) {
  int num_vertices = matching_BtoA.size();
  std::vector<std::array<int, 2>> chains;
  for(int i = 0; i < num_vertices; i++) {
    auto it = std::find(matching_BtoA.begin(), matching_BtoA.end(), i);
    if (it == matching_BtoA.end()) {
      chains.push_back({i, i});
    }
  }

  int group_num = 0;
  std::vector<int> mapping(num_vertices, -1);
  for (auto& chain : chains) {
    int group_id = group_num++;
    int curr = chain.at(1);
    while (true) {
      mapping.at(curr) = group_id;
      if (matching_BtoA.at(curr) == -1) {
        chain.at(0) = curr;
        break;
      } else {
        curr = matching_BtoA.at(curr);
      }
    }
  }
  return std::make_tuple(mapping, chains, group_num);
}



std::vector<int> longestCHelper(int u, const Graph& g, std::vector<int> vis, std::vector<int> l){
  if(vis[u]!=-1){
    return l;
  }
  if(g[u].size()==0){
    l.push_back(u);
    return l;
  }
  vis[u]=0;
  l.push_back(u);
  std::vector<int> l2;
  for(int i=0;i<g[u].size();i++){
    std::vector<int> l1 = longestCHelper(g[u][i], g, vis, l);
    if(l1.size()>l2.size()){
      l2=l1;
    }
  }
  return l2;
}

std::vector<int> longestChain(int u, const Graph &g) {
  std::vector<int> vis(g.size(), -1);
  std::vector<int> l;
  return longestCHelper(u, g, vis, l);
}

std::tuple<std::vector<int>, std::vector<int>, int> get_mapping_two_streams(const Graph& meg, const std::vector<int>& root_nodes) {
  int num_vertices = meg.size();
  std::vector<int> longestC;
  for(int i=0;i<root_nodes.size();i++){
    std::vector<int> tmp = longestChain(root_nodes[i], meg);
    if(tmp.size()>longestC.size()){
      longestC = tmp;
    }
  }
  /*std::cout<<"Completed Longest chain"<<std::endl;
  for(int i=0;i<longestC.size();i++){
    std::cout<<longestC[i]<<" ";
  }
  std::cout<<std::endl;*/
  std::vector<int> node_to_chain(meg.size(), 1);
  /*for(int i=0;i<chain_to_stream.size();i++){
    std::cout<<chain_to_stream[i]<<" ";
  }
  std::cout<<std::endl;*/
  int group_num=1;
  for(int i=0;i<longestC.size();i++){
    //std::cout<<"Hello "<<longestC[i]<<" "<<node_to_chain.size()<<std::endl;
    node_to_chain[longestC[i]]=0;
  }
  for(int i=0;i<node_to_chain.size();i++){
    if(node_to_chain[i]==0){
      continue;
    }
    node_to_chain[i]=group_num;
    group_num++;
  }
  std::vector<int> chain_to_stream(group_num, 1);
  chain_to_stream[0]=0;
  int stream_num = 2;
  if(chain_to_stream.size() < stream_num){
    stream_num = chain_to_stream.size();
  }
  return std::make_tuple(node_to_chain, chain_to_stream, stream_num);
}

///////////// Topological longest chain

void topo_helper(int u, const Graph& g, std::stack<int> &s, std::vector<int> &v) {
    if(v[u]!=-1){
        return;
    }
    v[u] = 0;
    for(int i=0;i<g[u].size();i++){
        topo_helper(g[u][i], g, s, v);
    }
    s.push(u);
}

std::vector<int> topological(const Graph& meg) {
    std::stack<int> s;
    std::vector<int> v(meg.size(), -1);
    for(int i=0;i<meg.size();i++){
        topo_helper(i, meg, s, v);
    }
    std::vector<int> res;
    while(!s.empty()){
        res.push_back(s.top());
        s.pop();
    }
    return res;
}

std::vector<int> getNumPreds(const Graph& g){
    std::vector<int> pred(g.size(), 0);
    for(int i=0;i<g.size();i++){
        for(int j=0;j<g[i].size();j++){
            pred[g[i][j]] = pred[g[i][j]]+1;
        }
    }
    return pred;
}


std::vector<std::vector<int> > getChains(const Graph &g){
    std::vector<int> topo = topological(g);
    std::cout<<"Got topology"<<std::endl;
    printVec(topo);
    std::vector<std::vector<int> > allChains;
    std::vector<int> vis(g.size(), -1);
    std::vector<int> pred = getNumPreds(g);
    std::cout<<"Got predecessors"<<std::endl;
    printVec(pred);

    int allAssigned = 0;
    while(allAssigned<g.size()){
        std::cout<<"visited "<<allAssigned<<" "<<g.size()<<std::endl;
        printVec(vis);
        printVec(pred);
        std::vector<int> l2;
        for(int i=0;i<topo.size();i++){
            if(vis[topo[i]]!=-1 || pred[topo[i]]>0){
                continue;
            }
	    std::cout<<"VIS: "<<topo[i]<<" "<<vis[topo[i]]<<" "<<pred[topo[i]]<<" Longest chain:"<<std::endl;
	    std::vector<int> l1 = longestCHelper(topo[i], g, vis, std::vector<int>());
	    printVec(l1);
            if(l1.size()>l2.size()){
                l2=l1;
            }
        }
	std::cout<<"longest chain so far"<<std::endl;
	printVec(l2);
        for(int i=0;i<l2.size();i++){
            vis[l2[i]]=0;
            allAssigned++;
            for(int j=0;j<g[l2[i]].size();j++){
                pred[g[l2[i]][j]] = pred[g[l2[i]][j]]-1;
            }
        }
        allChains.push_back(l2);

    for(int i=0;i<allChains.size();i++){
        std::cout<<"Chains in processing ";
	printVec(allChains[i]);
    }
    }
    for(int i=0;i<allChains.size();i++){
        std::cout<<"Chains "<<std::endl;
	printVec(allChains[i]);
    }
    return allChains;
}

std::tuple<std::vector<int>, std::vector<int>, int>  get_mapping_topo(const Graph& g){
    int c1=0, c2=0;
    std::vector<std::vector<int> > allChains = getChains(g);
    std::vector<int> node_to_chain(g.size(), 0);
    std::vector<int> chain_to_stream(allChains.size(), 0);
    for(int i=0;i<allChains.size();i++){
        for(int j=0;j<allChains[i].size();j++){
            node_to_chain[allChains[i][j]] = i;
        }
         if(c2<c1){
             chain_to_stream[i] = 1;
             c2+=allChains[i].size();
         }
         else{
            chain_to_stream[i] = 0;
            c1+=allChains[i].size();
         }

    }
    int stream_num = 2;
    if(allChains.size()<2){
        stream_num = 1;
    }
    /*
    for(int i=0;i<node_to_chain.size();i++){
        cout<<node_to_chain[i]<<" ";
    }
    cout<<endl;
    for(int i=0;i<chain_to_stream.size();i++){
        cout<<chain_to_stream[i]<<" ";
    }
    cout<<endl;
    */
    return std::make_tuple(node_to_chain, chain_to_stream, stream_num);
}



} // namespace DAGDecomposition

