#include <Rcpp.h>
#include <vector>
#include <random>
#include <cmath>
#include <omp.h>
#include <unistd.h>
#include <iostream>
#include <string>
using namespace Rcpp;

// sorting helper 
std::vector<size_t> rankSort(const std::vector<float>& v_temp) {
  std::vector<std::pair<float, size_t> > v_sort(v_temp.size());
  
  for (size_t i = 0U; i < v_sort.size(); ++i) {
    v_sort[i] = std::make_pair(v_temp[i], i);
  }
  
  std::sort(v_sort.begin(), v_sort.end());
  
  std::pair<double, size_t> rank;
  std::vector<size_t> result(v_temp.size());
  
  for (size_t i = 0U; i < v_sort.size(); ++i) {
    if (v_sort[i].first != rank.first) {
      rank = std::make_pair(v_sort[i].first, i);
    }
    result[v_sort[i].second] = rank.second;
  }
  return result;
}

// mean helper
double mean(const std::vector<size_t>& x, const std::vector<int>& id) {
  double out = 0.0;
  for(int i = 0; i < id.size(); ++i) {
    out += x[id[i]];
  }
  out = out / (double)id.size();
  return out;
}

// sd helper
double sd(const std::vector<double>& x) {
  double sum = 0.0, mean, sd = 0.0;
  int n = x.size();
  for(int i = 0; i < n; ++i) {
    sum += x[i];
  }
  mean = sum / n;
  for(int i = 0; i < n; ++i) {
    sd += pow(x[i] - mean, 2);
  }
  return sqrt(sd / n);
}

// find helper 
std::vector<int> findItems(std::vector<int> const &v, int target) {
  std::vector<int> indices;
  auto it = v.begin();
  while ((it = std::find_if(it, v.end(), [&] (int const &e) { return e == target; }))
           != v.end())
  {
    indices.push_back(std::distance(v.begin(), it));
    it++;
  }
  return indices;
}

// sampling with replacement helper 
std::vector<int> sample_replace(int max, int num_samples) {
  std::vector<int> samples(num_samples);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(0, max - 1);
  for(int i = 0; i < num_samples; ++i) {
    samples[i] = dist(gen);
  }
  return samples;
}

// rbc helper 
double get_rbc(const std::vector<float>& cont, const std::vector<int>& bin, int n) {
  std::vector<size_t> ranks = rankSort(cont);
  std::vector<int> id_0 = findItems(bin, 0);
  std::vector<int> id_1 = findItems(bin, 1);
  
  double y_1 = mean(ranks, id_1);
  double y_0 = mean(ranks, id_0);
  double rbc = 2 * (y_1 - y_0) / double(n);
  return rbc;
}

//progess 
void update_progress_bar(int progress, int total) {
  float percentage = (float)progress / total;
  int bar_width = 70;
  
  std::cout << "[";
  int pos = bar_width * percentage;
  for (int i = 0; i < bar_width; i++) {
    if (i < pos) std::cout << "=";
    else if (i == pos) std::cout << ">";
    else std::cout << " ";
  }
  std::cout << "] " << int(percentage * 100.0) << " %\r";
  std::cout.flush();
}

// [[Rcpp::export]]
std::vector<double> rbcorr(const std::vector<float>& cont, 
                           const std::vector<int>& bin,
                           bool boot_ci = false, 
                           int boot_n = 500,
                           int ncores = 2) {
  std::vector<double> out;
  int n = cont.size();
  double rbc = get_rbc(cont, bin, n);
  out.push_back(rbc);
  
  if(boot_ci) {
    std::vector<double> boot(boot_n);
    
    omp_set_num_threads(ncores);
    #pragma omp parallel for private(bin,cont)
    for(int i = 0; i < boot_n; ++i) {
      std::vector<int> id = sample_replace(n,n);
      std::vector<int> bin_i;
      std::vector<float> cont_i;
      bin_i.reserve(n);
      cont_i.reserve(n);
        
      for(int j = 0; j < n; ++j) {
        bin_i.push_back(bin[id[j]]);
        cont_i.push_back(cont[id[j]]);
      }
      
      boot[i] = get_rbc(cont_i, bin_i, n);
      #pragma omp single 
      {
        update_progress_bar(i + 1, boot_n);
      } 
    }
    double se = sd(boot);
    out.push_back(rbc - 1.96 * se);
    out.push_back(rbc + 1.96 * se);
  }
  return out;
}
