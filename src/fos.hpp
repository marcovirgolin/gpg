#ifndef LINKAGE_H
#define LINKAGE_H

#include <vector>
#include <unordered_map>
#include "globals.hpp"
#include "node.hpp"
#include "myeig.hpp"
#include "util.hpp"

using namespace std;
using namespace myeig;

struct FOSBuilder
{

  bool first_time = true;
  Mat B;

  vector<vector<int>> build_linkage_tree(vector<Node *> &population, bool ablate_to_random = false)
  {
    int num_random_variables = population[0]->subtree().size();

    Mat MI;

    if (ablate_to_random)
    {
      MI = Mat::Random(num_random_variables, num_random_variables);
    }
    else
    {
      // discretize population symbols for speed
      auto discrpop_n_numsymb = discretize_population_symbols(population, num_random_variables);
      auto discr_pop = discrpop_n_numsymb.first;
      int num_symbs = discrpop_n_numsymb.second;
      // estimate MI
      MI = compute_MI(discr_pop, num_symbs, num_random_variables);
    }

    vector<vector<int>> fos;
    return fos;
  }

  // utility for helping linkage learning to consider a contained number of constants
  float constant_binning(float c, vector<float> &binned_constants, int max_constants_binning = 25)
  {
    // check if this constant is already inserted
    auto it = find(binned_constants.begin(), binned_constants.end(), c);
    if (it != binned_constants.end())
    {
      return c;
    }
    // check if there is space to insert this
    if (binned_constants.size() < max_constants_binning)
    {
      binned_constants.push_back(c);
      return c;
    }
    // else, find nearest
    float closest_c;
    float min_dist = INF;
    for (float &other : binned_constants)
    {
      float dist = abs(other - c);
      if (dist < min_dist)
      {
        min_dist = dist;
        closest_c = other;
      }
    }
    return closest_c;
  }

  pair<vector<vector<int>>, int> discretize_population_symbols(vector<Node *> &population, int num_random_variables, int max_constants_binning = 25)
  {

    int pop_size = population.size();
    int num_symbs = 0;
    unordered_map<string, int> symb_to_discr_map;
    symb_to_discr_map.reserve(1000000);

    // maximum number of constants to consider for binning
    vector<float> binned_constants;
    binned_constants.reserve(100);

    // pre-process the population: make a symbolic representation of the population nodes
    vector<vector<int>> discr_pop(pop_size);

    for (int i = 0; i < pop_size; i++)
    {
      vector<int> discr_nodes;
      discr_nodes.reserve(num_random_variables);

      vector<Node *> nodes = population[i]->subtree();
      for (int j = 0; j < num_random_variables; j++)
      {
        Node *n = nodes[j];
        string v = n->op->sym();
        // if constant, actually use constant binning
        if (n->op->type() == OpType::otConst)
        {
          float c = ((Const *)n->op)->c;
          v = to_string(constant_binning(c, binned_constants, max_constants_binning));
        }
        // discretize
        auto it = symb_to_discr_map.find(v);
        if (it == symb_to_discr_map.end())
        {
          symb_to_discr_map[v] = num_symbs;
          discr_nodes.push_back(num_symbs);
          num_symbs++;
        }
        else
        {
          discr_nodes.push_back(it->second);
        }
      }
      discr_pop[i] = discr_nodes;
    }

    assert(num_symbs == symb_to_discr_map.size());

    return make_pair(discr_pop, num_symbs);
  }

  Mat compute_MI(vector<vector<int>> &discr_pop, int num_symbs, int num_random_variables)
  {
    int pop_size = discr_pop.size();

    // intiialize MI matrix at zero
    Mat MI = Mat::Zero(num_random_variables, num_random_variables);

    // build pairwise frequency matrix for symbol pairs
    Mat F = Mat::Zero(num_symbs, num_symbs);
    int val_i, val_j;

    // measure frequencies of pairs & compute single and joint entropy on the way
    for (int i = 0; i < num_random_variables; i++)
    {
      for (int j = i + 1; j < num_random_variables; j++)
      {
        for (int p = 0; p < pop_size; p++)
        {
          val_i = discr_pop[p][i];
          val_j = discr_pop[p][j];
          F(val_i, val_j) += 1.0;
        }

        double_t freq;
        for (int k = 0; k < num_symbs; k++)
        {
          for (int l = 0; l < num_symbs; l++)
          {
            freq = F(k, l);
            if (freq > 0.0)
            {
              freq = freq / pop_size;
              MI(i, j) += -freq * log(freq);
              F(k, l) = 0.0; // reset the freq;
            }
          }
        }
        MI(j, i) = MI(i, j);
      }

      for (int p = 0; p < pop_size; p++)
      {
        val_i = discr_pop[p][i];
        F(val_i, val_i) += 1.0;
      }

      double_t freq;
      for (int k = 0; k < num_symbs; k++)
      {
        for (int l = 0; l < num_symbs; l++)
        {
          freq = F(k, l);
          if (freq > 0)
          {
            freq = freq / pop_size;
            MI(i, i) += -freq * log(freq);
            F(k, l) = 0.0; // reset the freq;
          }
        }
      }
    }

    // register bias to account for non-uniform distribution of symbols in initialized GP population
    if (first_time)
    {
      for (int i = 0; i < num_random_variables; i++)
      {
        B(i, i) = 1.0 / MI(i, i);
        for (int j = i + 1; j < num_random_variables; j++)
        {
          B(i, j) = 2.0 / MI(i, j);
        }
      }
      first_time = false;
    }

    // apply bias
    for (int i = 0; i < num_random_variables; i++)
    {
      MI(i, i) = MI(i, i) * B(i, i);
      for (int j = i + 1; j < num_random_variables; j++)
      {
        MI(i, j) = MI(i, j) * B(i, j);
      }
    }

    // transform entropy into mutual info
    for (int i = 0; i < num_random_variables; i++)
    {
      for (int j = i + 1; j < num_random_variables; j++)
      {
        MI(i, j) = MI(i, i) + MI(j, j) - MI(i, j);
        MI(j, i) = MI(i, j);
      }
    }

    return MI;
  }

  vector<vector<int>> fast_upgma(Mat &S)
  {
    // fast UPGMA from a similarity matrix S
    // returns a hierarchical cluster
    vector<vector<int>> h_cluster;

    // S is an N*N matrix
    int num_entries = S.rows();

    // random order
    vector<int> random_order = rand_perm(num_entries);

    // initial marginal product model
    vector<vector<int>> mpm(num_entries, vector<int>(1));
    vector<int> mpm_number_of_indices(num_entries);
    int mpm_length = num_entries;

    for (int i = 0; i < num_entries; i++)
    {
      mpm[i][0] = random_order[i];
      mpm_number_of_indices[i] = 1;
    }

    // Initialize hierarchical cluster to the initial MPM
    h_cluster.resize(num_entries + num_entries - 1);
    int cl_index = 0;
    for (int i = 0; i < mpm_length; i++)
    {
      h_cluster[cl_index] = vector<int>(mpm[i].begin(), mpm[i].end());
      cl_index++;
    }

    // Rearrange similarity matrix based on random order of MPM
    Mat Sprime(num_entries, num_entries);
    for (int i = 0; i < mpm_length; i++)
      for (int j = 0; j < mpm_length; j++)
        Sprime(i, j) = S(mpm[i][0], mpm[j][0]);
    for (int i = 0; i < mpm_length; i++)
      Sprime(i, i) = 0; // no need to assess self-similarity

    vector<vector<int>> mpm_new;
    vector<int> NN_chain(num_entries + 2, 0);
    int NN_chain_length = 0;
    bool done = false;

    while (!done)
    {
      if (NN_chain_length == 0)
      {
        NN_chain[NN_chain_length] = (int)randu() * mpm_length;
        NN_chain_length++;
      }

      while (NN_chain_length < 3)
      {
        NN_chain[NN_chain_length] = nearest_neigh(NN_chain[NN_chain_length - 1], Sprime, mpm_number_of_indices, mpm_length);
        NN_chain_length++;
      }

      while (NN_chain[NN_chain_length - 3] != NN_chain[NN_chain_length - 1])
      {
        NN_chain[NN_chain_length] = nearest_neigh(NN_chain[NN_chain_length - 1], Sprime, mpm_number_of_indices, mpm_length);
        if (((Sprime(NN_chain[NN_chain_length - 1], NN_chain[NN_chain_length]) == Sprime(NN_chain[NN_chain_length - 1], NN_chain[NN_chain_length - 2]))) && (NN_chain[NN_chain_length] != NN_chain[NN_chain_length - 2]))
          NN_chain[NN_chain_length] = NN_chain[NN_chain_length - 2];
        NN_chain_length++;
        if (NN_chain_length > num_entries)
          break;
      }
      int r0 = NN_chain[NN_chain_length - 2];
      int r1 = NN_chain[NN_chain_length - 1];
      int rswap;
      if (r0 > r1)
      {
        rswap = r0;
        r0 = r1;
        r1 = rswap;
      }
      NN_chain_length -= 3;

      if (r1 < mpm_length)
      { /* This test is required for exceptional cases in which the nearest-neighbor ordering has changed within the chain while merging within that chain */
        vector<int> indices(mpm_number_of_indices[r0] + mpm_number_of_indices[r1]);

        int i = 0;
        for (int j = 0; j < mpm_number_of_indices[r0]; j++)
        {
          indices[i] = mpm[r0][j];
          i++;
        }
        for (int j = 0; j < mpm_number_of_indices[r1]; j++)
        {
          indices[i] = mpm[r1][j];
          i++;
        }

        h_cluster[cl_index] = indices;
        cl_index++;

        double_t mul0 = ((double_t)mpm_number_of_indices[r0]) / ((double_t)mpm_number_of_indices[r0] + mpm_number_of_indices[r1]);
        double_t mul1 = ((double_t)mpm_number_of_indices[r1]) / ((double_t)mpm_number_of_indices[r0] + mpm_number_of_indices[r1]);
        for (i = 0; i < mpm_length; i++)
        {
          if ((i != r0) && (i != r1))
          {
            Sprime(i, r0) = mul0 * Sprime(i, r0) + mul1 * Sprime(i, r1);
            Sprime(r0, i) = Sprime(i, r0);
          }
        }

        mpm_new = vector<vector<int>>(mpm_length - 1);
        vector<int> mpm_new_number_of_indices(mpm_length - 1);
        int mpm_new_length = mpm_length - 1;
        for (i = 0; i < mpm_new_length; i++)
        {
          mpm_new[i] = mpm[i];
          mpm_new_number_of_indices[i] = mpm_number_of_indices[i];
        }

        mpm_new[r0] = vector<int>(indices.begin(), indices.end());

        mpm_new_number_of_indices[r0] = mpm_number_of_indices[r0] + mpm_number_of_indices[r1];
        if (r1 < mpm_length - 1)
        {
          mpm_new[r1] = mpm[mpm_length - 1];
          mpm_new_number_of_indices[r1] = mpm_number_of_indices[mpm_length - 1];

          for (i = 0; i < r1; i++)
          {
            Sprime(i, r1) = Sprime(i, mpm_length - 1);
            Sprime(r1, i) = Sprime(i, r1);
          }

          for (int j = r1 + 1; j < mpm_new_length; j++)
          {
            Sprime(r1, j) = Sprime(j, mpm_length - 1);
            Sprime(j, r1) = Sprime(r1, j);
          }
        }

        for (i = 0; i < NN_chain_length; i++)
        {
          if (NN_chain[i] == mpm_length - 1)
          {
            NN_chain[i] = r1;
            break;
          }
        }

        mpm = mpm_new;
        mpm_number_of_indices = mpm_new_number_of_indices;
        mpm_length = mpm_new_length;

        if (mpm_length == 1)
          done = true;
      }
    }

    return h_cluster;
  }

  int nearest_neigh(int index, Mat &S, vector<int> &mpm_number_of_indices, int mpm_length)
  {
    int i, result;

    result = 0;
    if (result == index)
      result++;
    for (i = 1; i < mpm_length; i++)
    {
      if (((S(index, i) > S(index, result)) || ((S(index, i) == S(index, result)) && (mpm_number_of_indices[i] < mpm_number_of_indices[result]))) && (i != index))
        result = i;
    }

    return result;
  }
};

#endif