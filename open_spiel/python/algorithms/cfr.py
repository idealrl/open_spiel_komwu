# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Python implementation of the counterfactual regret minimization algorithm.

One iteration of CFR consists of:
1) Compute current strategy from regrets (e.g. using Regret Matching).
2) Compute values using the current strategy
3) Compute regrets from these values

The average policy is what converges to a Nash Equilibrium.
"""


import collections
import attr
import numpy as np

from open_spiel.python import policy
import pyspiel
from scipy.special import logsumexp

from numpy import random



LID = 0
INFOID = 0

@attr.s
class _InfoStateNode(object):
  """An object wrapping values associated to an information state."""
  # The list of the legal actions.

  legal_actions = attr.ib()
  index_in_tabular_policy = attr.ib()

  parent_infoset = attr.ib()
  #children_infoset = attr.ib()
  parent_seq = attr.ib()
  states = attr.ib()
  sequences = attr.ib()
  children = attr.ib()
  ID = attr.ib()
  children_other_players = attr.ib()
  player = attr.ib()
  info_state = attr.ib()
  depth = attr.ib()
  cfrp_sum = attr.ib()
  accum_sum = attr.ib()
  reach_probs_sum = attr.ib(default=[1.0, 1.0])
  reach_probs_store = attr.ib(default=[1.0, 1.0])
  actions_to_sequences = attr.ib(factory=dict)
  sequence_to_infoset = attr.ib(factory=dict)
  
  # Map from information states string representations and actions to the
  # counterfactual regrets, accumulated over the policy iterations
  cumulative_regret = attr.ib(factory=lambda: collections.defaultdict(float))
  # Same as above for the cumulative of the policy probabilities computed
  # during the policy iterations
  cumulative_policy = attr.ib(factory=lambda: collections.defaultdict(float))
  

def _apply_regret_matching_plus_reset(info_state_nodes):
  """Resets negative cumulative regrets to 0.

  Regret Matching+ corresponds to the following cumulative regrets update:
  cumulative_regrets = max(cumulative_regrets + regrets, 0)

  This must be done at the level of the information set, and thus cannot be
  done during the tree traversal (which is done on histories). It is thus
  performed as an additional step.

  This function is a module level function to be reused by both CFRSolver and
  CFRBRSolver.

  Args:
    info_state_nodes: A dictionary {`info_state_str` -> `_InfoStateNode`}.
  """
  for info_state_node in info_state_nodes.values():
    action_to_cum_regret = info_state_node.cumulative_regret
    for action, cumulative_regret in action_to_cum_regret.items():
      if cumulative_regret < 0:
        action_to_cum_regret[action] = 0


def _update_current_policy(current_policy, info_state_nodes):
  """Updates in place `current_policy` from the cumulative regrets.

  This function is a module level function to be reused by both CFRSolver and
  CFRBRSolver.

  Args:
    current_policy: A `policy.TabularPolicy` to be updated in-place.
    info_state_nodes: A dictionary {`info_state_str` -> `_InfoStateNode`}.
  """
  for info_state, info_state_node in info_state_nodes.items():
    state_policy = current_policy.policy_for_key(info_state)

    for action, value in _regret_matching(
        info_state_node.cumulative_regret,
        info_state_node.legal_actions).items():
      state_policy[action] = value


def _update_average_policy(average_policy, info_state_nodes):
  """Updates in place `average_policy` to the average of all policies iterated.

  This function is a module level function to be reused by both CFRSolver and
  CFRBRSolver.

  Args:
    average_policy: A `policy.TabularPolicy` to be updated in-place.
    info_state_nodes: A dictionary {`info_state_str` -> `_InfoStateNode`}.
  """
  for info_state, info_state_node in info_state_nodes.items():
    info_state_policies_sum = info_state_node.cumulative_policy
    state_policy = average_policy.policy_for_key(info_state)
    probabilities_sum = sum(info_state_policies_sum.values())
    if probabilities_sum == 0:
      num_actions = len(info_state_node.legal_actions)
      for action in info_state_node.legal_actions:
        state_policy[action] = 1 / num_actions
    else:
      for action, action_prob_sum in info_state_policies_sum.items():
        state_policy[action] = action_prob_sum / probabilities_sum


class _CFRSolverBase(object):
  r"""A base class for both CFR and CFR-BR.

  The main iteration loop is implemented in `evaluate_and_update_policy`:

  ```python
      game = pyspiel.load_game("game_name")
      initial_state = game.new_initial_state()

      solver = Solver(game)

      for i in range(num_iterations):
        solver.evaluate_and_update_policy()
        solver.current_policy()  # Access the current policy
        solver.average_policy()  # Access the average policy
  ```
  """

  def __init__(self, game, alternating_updates, linear_averaging,
               regret_matching_plus):
    # pyformat: disable
    """Initializer.

    Args:
      game: The `pyspiel.Game` to run on.
      alternating_updates: If `True`, alternating updates are performed: for
        each player, we compute and update the cumulative regrets and policies.
        In that case, and when the policy is frozen during tree traversal, the
        cache is reset after each update for one player.
        Otherwise, the update is simultaneous.
      linear_averaging: Whether to use linear averaging, i.e.
        cumulative_policy[info_state][action] += (
          iteration_number * reach_prob * action_prob)

        or not:

        cumulative_policy[info_state][action] += reach_prob * action_prob
      regret_matching_plus: Whether to use Regret Matching+:
        cumulative_regrets = max(cumulative_regrets + regrets, 0)
        or simply regret matching:
        cumulative_regrets = cumulative_regrets + regrets
    """
    # pyformat: enable
    assert game.get_type().dynamics == pyspiel.GameType.Dynamics.SEQUENTIAL, (
        "CFR requires sequential games. If you're trying to run it " +
        "on a simultaneous (or normal-form) game, please first transform it " +
        "using turn_based_simultaneous_game.")

    self._game = game
    self._num_players = game.num_players()
    self._root_node = self._game.new_initial_state()

    # This is for returning the current policy and average policy to a caller
    self._current_policy = policy.TabularPolicy(game)#, players=[0,1])
    # print(self._current_policy.action_probability_array)
    # print("AVOVE")
    
    self._average_policy = self._current_policy.__copy__()
    self._ref_policy = self._current_policy.__copy__()
    self.nodes_touched = 0
    # self._info_state_nodes = {}
    # self._initialize_info_state_nodes(self._root_node)

    

    self._info_state_nodes_komwu = [{} for _ in range(self._num_players)]
    self._reach_probs = [{} for _ in range(self._num_players)]
    
    # Set a 'block' which is how often to check the lazy update condition
  # So if block=3, only update update_infosets for 'block' iterations
  # After 'block' iterations, check conditions again
    # If sum(pi-i) > 1, add to update_blocks
    self._update_nodes_komwu = [[],[]]
    self.blocks = 20
    self.current_block = 1
    
    global LID
    global INFOID
    p = 0
    
    # self.seq_ID = 0
    LID = 0
    INFOID = 0
    # self.ID = 0 # infoset IDs
    # Arguments(state, last_action, player_id, parent_infoset, depth)
    self._initialize_info_state_nodes_komwu(self._root_node, -1, p,None, 0)
    self._initialize_children_komwu(self._root_node, -1, p, None)
    b0 = [0 for _ in range(LID)]

    p = 1
    # self.seq_id = 0 # Reset to make sequences
    self.ID = 0
    INFOID = 0
    LID = 0
    self._initialize_info_state_nodes_komwu(self._root_node, -1, p,None,0)
    self._initialize_children_komwu(self._root_node, -1, p, None)
    
    for player_id in range(self._num_players):
      for info_str, infoset in self._info_state_nodes_komwu[player_id].items():
        self._update_nodes_komwu[player_id].append(info_str)
    
    b1 = [0 for _ in range(LID)]

    self.K_j = [[None] * len(self._info_state_nodes_komwu[player_id]) for player_id in range(self._num_players)]


    self.b = [b0, b1]#[[0 for _ in range(self.seq_id)] for _ in range(self._num_players)]
    self.grad = [b0, b1]
    self.last_grad = [b0, b1]

    self.y = [[0 for _ in range(len(self.b[i]))] for i in range(self._num_players)]
    self.y_exp = [[0 for _ in range(len(self.b[i]))] for i in range(self._num_players)]  
    
    self.visited = []

    g0 = [0 for _ in range(len(self.b[0]))]
    g1 = [0 for _ in range(len(self.b[1]))]
    self.last_grad = [[],[]]
    self.last_grad[0] = [[0] for _ in range(len(self.b[0]))]
    for i in range(len(self.last_grad[0])):
      self.last_grad[0][i].append(0)
    self.last_grad[1] = [[0] for _ in range(len(self.b[1]))]
    for i in range(len(self.last_grad[1])):
      self.last_grad[1][i].append(0)

    
    # grad[player][seq][0] = grad
    # grad[player][seq][1] = depth/optimism to apply
    g0 = [0 for _ in range(len(self.b[0]))]
    g1 = [0 for _ in range(len(self.b[1]))]
    self.grad = [[],[]]
    self.grad[0] = [[0] for _ in range(len(self.b[0]))]
    for i in range(len(self.grad[0])):
      self.grad[0][i].append(0)
    self.grad[1] = [[0] for _ in range(len(self.b[1]))]
    for i in range(len(self.grad[1])):
      self.grad[1][i].append(0)
    # KEEP GRADS FOR UNTOUCHED INFOSETS?


    p0 = []
    p1 = []
    self._root_node = self._game.new_initial_state()


    # print(self._update_nodes_komwu)
    self._compute_x(t=0)

    self._iteration = 0  # For possible linear-averaging.
    self._linear_averaging = linear_averaging
    self._alternating_updates = alternating_updates
    self._regret_matching_plus = regret_matching_plus
      
        
  def _initialize_info_state_nodes_komwu(self, state, last_seq, player_id, parent_infoset, depth):

    if state.is_terminal():
      # Collect terminals?
      return

    if state.is_chance_node():
      # Check later, Leduc has multiple chance nodes
      for action, unused_action_prob in state.chance_outcomes():
        self._initialize_info_state_nodes_komwu(state.child(action), last_seq, player_id, parent_infoset, depth)
      return
     
    current_player = state.current_player()
    info_state = state.information_state_string(current_player)
    info_state_node = self._info_state_nodes_komwu[player_id].get(info_state)
  
    print(info_state)
    # Collect only infosets that are player_id
    # If not at the current player_id, keep going
    if current_player != player_id:
      for action in state.legal_actions(current_player):
        if state.child(action) != None:
          self._initialize_info_state_nodes_komwu(state.child(action), last_seq, player_id, parent_infoset, depth)
  
    # Are current player
    else:
      global INFOID
      if info_state_node is None:
        legal_actions = state.legal_actions(current_player)
        if len(legal_actions) <= 1:
          print("LESS: ", legal_actions)
        info_state_node = _InfoStateNode(
            legal_actions=legal_actions,
            index_in_tabular_policy=self._current_policy.state_lookup[info_state],
            parent_infoset=parent_infoset,
            parent_seq=last_seq,
            states=[state],
            sequences=[],
            children=[],
            children_other_players=[],
            ID=INFOID,
            player=current_player,
            info_state=info_state,
            depth=depth,
            cfrp_sum=0.0,
            accum_sum=1.0,
            reach_probs_sum=[0.0,0.0],
            reach_probs_store=[0.0,0.0]

            )
        global LID
        INFOID += 1
        infoid = INFOID - 1
        # print("Info_state: ", info_state, " par_seq: ", last_seq, " THIS SEQ: ", LID)
        self._info_state_nodes_komwu[player_id][info_state] = info_state_node
        for action in info_state_node.legal_actions:
          self._info_state_nodes_komwu[player_id][info_state].sequences.append(LID)
          self._info_state_nodes_komwu[player_id][info_state].actions_to_sequences[action] = LID
          self._info_state_nodes_komwu[player_id][info_state].sequence_to_infoset[LID] = []
          cur_seq = LID
          LID += 1
          # print("Info_state: ", info_state," INFODI: ", infoid, " par_seq: ", last_seq, " THIS SEQ: ", cur_seq)
          self._initialize_info_state_nodes_komwu(state.child(action), cur_seq, player_id, info_state_node, depth+1)
          
      # If infoset already exists, append history (NOT USED CURRENTLY)
      else:
        self._info_state_nodes_komwu[player_id][info_state].states.append(state)
        for action in info_state_node.legal_actions:
          action_to_seq = self._info_state_nodes_komwu[player_id][info_state].actions_to_sequences[action]
          self._initialize_info_state_nodes_komwu(state.child(action), action_to_seq, player_id, info_state_node, depth+1)


  def _initialize_children_komwu(self, state, last_seq, player_id, parent_infoset):

    if state.is_terminal():
      # Collect terminals?
      return

    if state.is_chance_node():
      # Check later, Leduc has multiple chance nodes
      for action, unused_action_prob in state.chance_outcomes():
        self._initialize_children_komwu(state.child(action), last_seq, player_id, parent_infoset)
      return
 
    current_player = state.current_player()
    info_state = state.information_state_string(current_player)
    info_state_node = self._info_state_nodes_komwu[player_id].get(info_state)

    if current_player != player_id:
      # Add here for relationships for i to -i
      for action in state.legal_actions(current_player):
        if state.child(action) != None:
          if parent_infoset != None:
            if info_state not in self._info_state_nodes_komwu[player_id][parent_infoset].children_other_players:
              self._info_state_nodes_komwu[player_id][parent_infoset].children_other_players.append(info_state)
          self._initialize_children_komwu(state.child(action), last_seq, player_id, parent_infoset)
    
    # Are the player
    else:
      if parent_infoset != None:
        if info_state not in self._info_state_nodes_komwu[player_id][parent_infoset].children:
          # get the sequence ID from parent infoset that led to here
          # Add actions to sequences above
          seq = last_seq#self._info_state_nodes_komwu[player_id][parent_infoset].actions_to_sequences[last_action]
          # Need .children?
          if info_state_node not in self._info_state_nodes_komwu[player_id][parent_infoset].sequence_to_infoset[seq]:
            self._info_state_nodes_komwu[player_id][parent_infoset].sequence_to_infoset[seq].append(info_state_node)
          #   print("Setting ", parent_infoset, " now has: ", info_state_node.info_state)
          # else:
          #   print("Actually ", parent_infoset, " ALREADY HAS: ", info_state_node.info_state)
         
          self._info_state_nodes_komwu[player_id][parent_infoset].children.append(info_state_node)

        
      for action in state.legal_actions(current_player): #info_state_node.legal_actions:
        seq = self._info_state_nodes_komwu[player_id][info_state].actions_to_sequences[action]
        self._initialize_children_komwu(state.child(action), seq, player_id, info_state)
      



  def current_policy(self):
    """Returns the current policy as a TabularPolicy.

    WARNING: The same object, updated in-place will be returned! You can copy
    it (or its `action_probability_array` field).

    For CFR/CFR+, this policy does not necessarily have to converge. It
    converges with high probability for CFR-BR.
    """
    return self._current_policy

  def average_policy(self):
    """Returns the average of all policies iterated.

    WARNING: The same object, updated in-place will be returned! You can copy
    it (or its `action_probability_array` field).

    This average policy converges to a Nash policy as the number of iterations
    increases.

    The policy is computed using the accumulated policy probabilities computed
    using `evaluate_and_update_policy`.

    Returns:
      A `policy.TabularPolicy` object (shared between calls) giving the (linear)
      time averaged policy (weighted by player reach probabilities) for both
      players.
    """
    # These just normalize I think, check again
    _update_average_policy(self._average_policy, self._info_state_nodes_komwu[0])
    _update_average_policy(self._average_policy, self._info_state_nodes_komwu[1])
    
    return self._average_policy


  def _update(self, state, policies, reach_probabilities, player, seqs, depth, update, t, r):
    
    if state.is_terminal():
      return np.asarray(state.returns())

    if state.is_chance_node():
      for action, action_prob in state.chance_outcomes():
        self._update(state.child(action), policies, None, player, seqs, depth, update, t,r)
      return 0

    current_player = state.current_player()
    info_state = state.information_state_string(current_player)
    info_state_node = self._info_state_nodes_komwu[current_player][info_state]
    
    if info_state not in self._update_nodes_komwu[current_player]:
      
      if info_state_node.reach_probs_sum[current_player] >= -0.00005:# -0.00005: #0.000005:
        self._update_nodes_komwu[current_player].append(info_state)#, info_state_node])
        
        other_player = 1 - current_player
        info_state_node.reach_probs_sum[current_player] = 0.0
        info_state_node.reach_probs_sum[other_player] = 0.0
        # print("Added *", info_state, "*  t=", t)
        # Should this call a helper to distribute probs_store here?
        # So add 1-1-2 because its reach_probs_sum is high enough now
        # If info_state is new (wasn't in old update_nodes_komwu)
        # go down the tree and add infosets 

      else:
        # other_player = 1 - current_player
        # print("Not adding: ", info_state, "  t=",t)
        # print("   up0: ", info_state_node.reach_probs_sum[current_player])
        # print("   up0: ", info_state_node.reach_probs_store[current_player]) 
        # print("   up1: ", info_state_node.reach_probs_sum[other_player])
        # print("   up1: ", info_state_node.reach_probs_store[other_player])
        return 0
        # other_player = 1 - current_player
        # info_state_node.reach_probs_store[current_player] = 0.0
        # info_state_node.reach_probs_store[other_player] = 0.0

    for action in state.legal_actions():
      seq = self._info_state_nodes_komwu[current_player][info_state].actions_to_sequences[action]      
      new_seqs = seqs.copy()
      new_seqs[current_player] = seq
      seqs[current_player] = seq

      child_utility = self._update(
          state.child(action),
          policies=policies,
          reach_probabilities=None,
          player=player,
          seqs=new_seqs,
          depth=depth+1,
          update=update, t=t, r=r)



  def _compute_grad(self, state, policies, reach_probabilities, reach_probabilities_ref, player, seqs, depth, update, t):
    
    if state.is_terminal():
      depth = 2.0 #15.0


      self.mu = 0.01#1 / (t + 1) ** (3 / 4)#0.1
      cfr = reach_probabilities[-1]
      # Update player 0
      counterfactual_reach_prob_p0 = (
        np.prod(reach_probabilities[:0]) *
        np.prod(reach_probabilities[0 + 1:]))
      counterfactual_reach_prob_p1 = (
        np.prod(reach_probabilities[:1]) *
        np.prod(reach_probabilities[1 + 1:]))

      counterfactual_reach_prob_p0_ref = (
        np.prod(reach_probabilities_ref[:0]) *
        np.prod(reach_probabilities_ref[0 + 1:]))
      counterfactual_reach_prob_p1_ref = (
        np.prod(reach_probabilities_ref[:1]) *
        np.prod(reach_probabilities_ref[1 + 1:]))
      #  values = np.exp(self.eta * 
      # (utility + self.mu * (self.ref_strategy[a] - self.strategy[a]) / self.strategy[a]))
      # * self.strategy
      util_0 = state.returns()[0] * counterfactual_reach_prob_p0
      if counterfactual_reach_prob_p1 > 0:
        # print("HERE")
        extra_0 = self.mu * (counterfactual_reach_prob_p1_ref - counterfactual_reach_prob_p1) / counterfactual_reach_prob_p1
        if t < 100:
          extra_0 = 0.0
        # print(self.mu, counterfactual_reach_prob_p1_ref, counterfactual_reach_prob_p1, counterfactual_reach_prob_p1_ref - counterfactual_reach_prob_p1) 
        
      else:
        extra_0 = 0.0
      self.grad[0][seqs[0]][0] += util_0 + extra_0
      
      # WORKINGself.grad[0][seqs[0]][0] += state.returns()[0] * counterfactual_reach_prob_p0
      # WORKING self.grad[0][seqs[0]][0] += state.returns()[0] * cfr * self.y_exp[1][seqs[1]]
      
      self.grad[0][seqs[0]][1] = depth

      # Update player 1
      util_1 = state.returns()[1] * counterfactual_reach_prob_p1
      if counterfactual_reach_prob_p0 > 0:
        extra_1 = self.mu * (counterfactual_reach_prob_p0_ref - counterfactual_reach_prob_p0) / counterfactual_reach_prob_p0
        if t < 100:
          extra_1 = 0.0
      else:
        extra_1 = 0.0
      self.grad[1][seqs[1]][0] += util_1 + extra_1
      # WORKING self.grad[1][seqs[1]][0] += state.returns()[1] * counterfactual_reach_prob_p1
      # WORKING self.grad[1][seqs[1]][0] += state.returns()[1] * cfr * self.y_exp[0][seqs[0]]
      self.grad[1][seqs[1]][1] = depth
      
      return np.asarray(state.returns())

    if state.is_chance_node():
      state_value = 0.0
      for action, action_prob in state.chance_outcomes():
        assert action_prob > 0
        new_state = state.child(action)
        new_reach_probabilities = reach_probabilities.copy()
        new_reach_probabilities[-1] *= action_prob

        new_reach_probabilities_ref = reach_probabilities_ref.copy()
        new_reach_probabilities_ref[-1] *= action_prob

        # print("Chance: action: ", action, " action_prob: ", action_prob, " rp: ", reach_probabilities, " nrp: ", new_reach_probabilities)
        state_value += action_prob * self._compute_grad(
            new_state, policies, new_reach_probabilities, new_reach_probabilities_ref, player, seqs, depth, update, t)
      return state_value

    current_player = state.current_player()
    info_state = state.information_state_string(current_player)

    # if all(reach_probabilities[:-1] == 0):
    #   return 0#np.zeros(self._num_players)

    state_value = np.zeros(self._num_players)

    children_utilities = {}

    info_state_node = self._info_state_nodes_komwu[current_player][info_state]
    if policies is None:
      info_state_policy = self._get_infostate_policy_komwu(info_state, current_player)
    else:
      info_state_policy = policies[current_player](info_state)

    # If we aren't going past this infoset, keep track of reaches
    if info_state not in self._update_nodes_komwu[current_player]:
      # This isn't updated, just sum   
  
      # if t == 80:
      # print("Infostate not included: ", info_state, "  t=", t)
      counterfactual_reach_prob_cp = (
        np.prod(reach_probabilities[:current_player]) *
        np.prod(reach_probabilities[current_player + 1:]))
      
      other_player = 1 - current_player

      counterfactual_reach_prob_op = (
        np.prod(reach_probabilities[:other_player]) *
        np.prod(reach_probabilities[other_player + 1:]))

      if seqs[current_player] >= 0:
        info_state_node.reach_probs_sum[current_player] += counterfactual_reach_prob_cp  
        info_state_node.reach_probs_store[current_player] += counterfactual_reach_prob_cp  
        # if t == 80:
        #   print("   p0: ", info_state_node.reach_probs_sum[current_player])
        #   print("   p0: ", info_state_node.reach_probs_store[current_player])
      else:
        info_state_node.reach_probs_sum[current_player] = 1.0
        info_state_node.reach_probs_store[current_player] = 1.0  
      
        
      if seqs[other_player] >= 0:
        info_state_node.reach_probs_sum[other_player] += counterfactual_reach_prob_op
        info_state_node.reach_probs_store[other_player] += counterfactual_reach_prob_op
        # if t == 80:
        #   print("   p1: ", info_state_node.reach_probs_sum[other_player])
        #   print("   p1: ", info_state_node.reach_probs_store[other_player])
      else:
        info_state_node.reach_probs_sum[other_player] = 1.0
        info_state_node.reach_probs_store[other_player] = 1.0
        
      return 0

    counterfactual_reach_prob_cp = (
      np.prod(reach_probabilities[:current_player]) *
      np.prod(reach_probabilities[current_player + 1:]))
    
    other_player = 1 - current_player

    counterfactual_reach_prob_op = (
      np.prod(reach_probabilities[:other_player]) *
      np.prod(reach_probabilities[other_player + 1:]))

    other_player = 1 - current_player
    if seqs[current_player] >= 0:
        info_state_node.reach_probs_sum[current_player] += counterfactual_reach_prob_cp  
    else:
      info_state_node.reach_probs_sum[current_player] = 1.0
        
    if seqs[other_player] >= 0:
      info_state_node.reach_probs_sum[other_player] += counterfactual_reach_prob_op
    else:
        info_state_node.reach_probs_sum[other_player] = 1.0
    
    for action in state.legal_actions():
      action_prob = self._current_policy.action_probability_array[
          info_state_node.index_in_tabular_policy][action]
      new_state = state.child(action)
      new_reach_probabilities = reach_probabilities.copy()
      new_reach_probabilities[current_player] *= action_prob

      action_prob_ref = self._ref_policy.action_probability_array[
          info_state_node.index_in_tabular_policy][action]
      new_state_ref = state.child(action)
      new_reach_probabilities_ref = reach_probabilities_ref.copy()
      new_reach_probabilities_ref[current_player] *= action_prob_ref

      if info_state_node.reach_probs_store[current_player] > 0.0:
        new_reach_probabilities[current_player] *= info_state_node.reach_probs_store[current_player]
        # if t==80:
        #   print("Store0: ", info_state, " nrp: ", new_reach_probabilities[current_player], "  store: ", info_state_node.reach_probs_store[current_player])

      other_player = 1 - current_player 
      if info_state_node.reach_probs_store[other_player] > 0.0:
        new_reach_probabilities[other_player] *= info_state_node.reach_probs_store[other_player]
        # if t == 80:
        #   print("Store1: ", info_state, " nrp: ", new_reach_probabilities[other_player], " store: ", info_state_node.reach_probs_store[other_player])
      
      # if t == 80:
      #   print("infostate: ", info_state, " rps: ", new_reach_probabilities)

      # if t == 80:
      #   print("infostate: ", info_state, " rps: ", new_reach_probabilities)

      seq = self._info_state_nodes_komwu[current_player][info_state].actions_to_sequences[action]     
      reach_prob = reach_probabilities[current_player]
      info_state_node.cumulative_policy[action] += reach_prob * action_prob
      ######## CHANGED NOTHING
      new_seqs = seqs.copy()
      new_seqs[current_player] = seq
      ################

      seqs[current_player] = seq
      child_utility = self._compute_grad(
          new_state,
          policies=policies,
          reach_probabilities=new_reach_probabilities,
          reach_probabilities_ref=new_reach_probabilities_ref,
          player=player,
          seqs=new_seqs,
          depth=depth+1,
          update=update, t=t)

    other_player = 1 - current_player
    info_state_node.reach_probs_store[current_player] = 0.0
    info_state_node.reach_probs_store[other_player] = 0.0   

    return state_value


  def _komwu(self, t):
    

    # Update mutation policy
    if t > 0 and t % 100 == 0:
      self._ref_policy = self._current_policy.__copy__()

    if self.current_block == self.blocks:
      self.current_block = 1
      self._update_nodes_komwu = [[],[]]
      # print("Calling Update")
      r = 0.00005
      self._update(self._root_node,
              policies=None,
              reach_probabilities=np.ones(self._game.num_players() + 1),
              player=None,
              seqs=[-100000,-100000],depth=0, update=None, t=t, r=r)
      # print("Update End")
    else:
      self.current_block += 1

    
    self.nodes_touched += len(self._update_nodes_komwu[0]) + len(self._update_nodes_komwu[1])
    if t % 25 == 0:
      print("NT: t=",t, "  touched: ", self.nodes_touched)

    # Only want to zero out gradients for those in update_nodes_komwu
    for player_id in range(self._num_players):
      for info_str in self._update_nodes_komwu[player_id]:
        info_state_node = self._info_state_nodes_komwu[player_id][info_str]
        for action in info_state_node.legal_actions:
          seq = self._info_state_nodes_komwu[player_id][info_str].actions_to_sequences[action]      
          self.grad[player_id][seq] = [0,2.0]


    
    self._compute_grad(self._root_node,
            policies=None,
            reach_probabilities=np.ones(self._game.num_players() + 1),
            reach_probabilities_ref=np.ones(self._game.num_players() + 1),
            player=None,
            seqs=[-100000,-100000],depth=0, update=None, t=t)

    for player_id in range(self._num_players):
      opt = -100.0 # OPT IS SET IN COMPUTE_GRAD NOW
      # opt_grad = [(self.grad[player_id][i][1] * 1.0) * self.grad[player_id][i][0] - (self.grad[player_id][i][1]-1.0) * self.last_grad[player_id][i][0] for i in range(len(self.grad[player_id]))]
      opt_grad = opt_grad = []
      for i in range(len(self.grad[player_id])):
        opt = self.grad[player_id][i][1] * self.grad[player_id][i][0]
        prev = (self.grad[player_id][i][1]-1.0) * self.last_grad[player_id][i][0]
        opt_grad.append(opt - prev)
        # opt = self.grad[player_id][i][0]
        # opt_grad.append(opt)
      
      
      for i in range(len(self.b[player_id])):
        eta = 0.1 # eta <= 1/8
        t_temp = t ** (3.0 / 2.0)
        self.b[player_id][i] += (eta * opt_grad[i]) #* (t_temp / (t_temp + 1.0))

      # for i in range(len(self.b[player_id])):
      #   eta = 1.0 # eta <= 1/8
      #   t_temp = t ** (3.0 / 2.0)
      #   self.b[player_id][i] *= ((t_temp + 1.0) / (t_temp + 0.0))
    
    self.last_grad = [[],[]]
    for player_id in range(self._num_players):
      for i in range(len(self.grad[player_id])):
        self.last_grad[player_id].append([self.grad[player_id][i][0]])
    
    self._compute_x(t)

  def _compute_x(self,t):  

    # print(self._update_nodes_komwu)
    # print(self._info_state_nodes_komwu)
    new_list = [[],[]]
    for player_id in range(self._num_players):
      for i in range(20):
        for info_str in self._update_nodes_komwu[player_id]:#self._info_state_nodes_komwu[player_id].items():
          infoset = self._info_state_nodes_komwu[player_id][info_str]
        # for info_str, infoset in self._info_state_nodes_komwu[player_id].items():
          if infoset.depth == i:
            # print(info_str)
            # print(id(infoset))
            new_list[player_id].append(info_str)
            
    # Step 1
    for player_id in range(self._num_players):
      for info_str in new_list[player_id][::-1]:
        # info_str = infoset.info_state
        infoset = self._info_state_nodes_komwu[player_id][info_str]
        self.K_j[player_id][infoset.ID] = logsumexp([
            self.b[player_id][seq] + sum([self.K_j[player_id][child_infoset.ID] for child_infoset in self._info_state_nodes_komwu[player_id][info_str].sequence_to_infoset[seq]])
            for seq in infoset.sequences
        ])
      #   print("K_j: ", "(", info_str, ")", self.K_j[player_id][infoset.ID])
      # print("")


      # Step 3
      # Top-down traversal
      # for infoset in new_list[player_id]:
      for info_str in new_list[player_id]:
        # info_str = infoset.info_state
        infoset = self._info_state_nodes_komwu[player_id][info_str]
        for sequence_id in infoset.sequences: #range(infoset.start_sequence_id, infoset.end_sequence_id + 1):
          # Proposition 5.3 in logarithmic form
          if infoset.parent_seq == -1:
            y_new = 0
          else:
            y_new = self.y[player_id][infoset.parent_seq]
          self.y[player_id][sequence_id] = y_new \
          + self.b[player_id][sequence_id] + sum([self.K_j[player_id][child.ID] for child in self._info_state_nodes_komwu[player_id][info_str].sequence_to_infoset[sequence_id]]) \
          - self.K_j[player_id][infoset.ID]
  
    self.y_exp[0] = np.exp(self.y[0])
    self.y_exp[1] = np.exp(self.y[1])
    
    # Normalize sequence form for behavioral form
    for player_id in range(self._num_players):
      # for infoset in new_list[player_id]:
      # for info_str, infoset in self._info_state_nodes_komwu[player_id].items():
      for info_str in new_list[player_id]:
        #info_str = infoset.info_state
        infoset = self._info_state_nodes_komwu[player_id][info_str]
        denom = 0
        for action in infoset.legal_actions:
          seq = self._info_state_nodes_komwu[player_id][info_str].actions_to_sequences[action]      
          denom += self.y_exp[player_id][seq]
        for action in infoset.legal_actions:
          seq = self._info_state_nodes_komwu[player_id][info_str].actions_to_sequences[action]
          if denom < 1e-220:
            do_nothing_to_policy = 1
          else:         
            self._current_policy.action_probability_array[
            infoset.index_in_tabular_policy][action] = self.y_exp[player_id][seq] / denom
           
 
  def _get_infostate_policy(self, info_state_str):
    """Returns an {action: prob} dictionary for the policy on `info_state`."""
    for i in range(10):
      print("HHHEEEEE")
    info_state_node = self._info_state_nodes[info_state_str]
    prob_vec = self._current_policy.action_probability_array[
        info_state_node.index_in_tabular_policy]
    return {
        action: prob_vec[action] for action in info_state_node.legal_actions
    }

  def _get_infostate_policy_komwu(self, info_state_str, current_player):
    """Returns an {action: prob} dictionary for the policy on `info_state`."""
    info_state_node = self._info_state_nodes_komwu[current_player][info_state_str]
    prob_vec = self._current_policy.action_probability_array[
        info_state_node.index_in_tabular_policy]
    return {
        action: prob_vec[action] for action in info_state_node.legal_actions
    }


def _regret_matching(cumulative_regrets, legal_actions):
  """Returns an info state policy by applying regret-matching.

  Args:
    cumulative_regrets: A {action: cumulative_regret} dictionary.
    legal_actions: the list of legal actions at this state.

  Returns:
    A dict of action -> prob for all legal actions.
  """
  regrets = cumulative_regrets.values()
  sum_positive_regrets = sum((regret for regret in regrets if regret > 0))

  info_state_policy = {}
  if sum_positive_regrets > 0:
    for action in legal_actions:
      positive_action_regret = max(0.0, cumulative_regrets[action])
      info_state_policy[action] = (
          positive_action_regret / sum_positive_regrets)
  else:
    for action in legal_actions:
      info_state_policy[action] = 1.0 / len(legal_actions)
  return info_state_policy


class _CFRSolver(_CFRSolverBase):
  r"""Implements the Counterfactual Regret Minimization (CFR) algorithm.

  The algorithm computes an approximate Nash policy for 2 player zero-sum games.

  CFR can be view as a policy iteration algorithm. Importantly, the policies
  themselves do not converge to a Nash policy, but their average does.

  The main iteration loop is implemented in `evaluate_and_update_policy`:

  ```python
      game = pyspiel.load_game("game_name")
      initial_state = game.new_initial_state()

      cfr_solver = CFRSolver(game)

      for i in range(num_iterations):
        cfr.evaluate_and_update_policy()
  ```

  Once the policy has converged, the average policy (which converges to the Nash
  policy) can be computed:
  ```python
        average_policy = cfr_solver.ComputeAveragePolicy()
  ```

  # Policy and average policy

  policy(0) and average_policy(0) are not technically defined, but these
  methods will return arbitrarily the uniform_policy.

  Then, we are expected to have:

  ```
  for t in range(1, N):
    cfr_solver.evaluate_and_update_policy()
    policy(t) = RM or RM+ of cumulative regrets
    avg_policy(t)(s, a) ~ \sum_{k=1}^t player_reach_prob(t)(s) * policy(k)(s, a)

    With Linear Averaging, the avg_policy is proportional to:
    \sum_{k=1}^t k * player_reach_prob(t)(s) * policy(k)(s, a)
  ```
  """

  def evaluate_and_update_policy(self):
    """Performs a single step of policy evaluation and policy improvement."""
    self._iteration += 1
    if self._alternating_updates:
      for player in range(self._game.num_players()):
        self._compute_counterfactual_regret_for_player(
            self._root_node,
            policies=None,
            reach_probabilities=np.ones(self._game.num_players() + 1),
            player=player)
        if self._regret_matching_plus:
          _apply_regret_matching_plus_reset(self._info_state_nodes)
        _update_current_policy(self._current_policy, self._info_state_nodes)
    else:
      self._komwu(self._iteration)
      # _update_average_policy(self._average_policy, self._info_state_nodes_komwu)
      # _update_average_policy(self._current_policy, self._info_state_nodes)

      # self._compute_counterfactual_regret_for_player(
      #     self._root_node,
      #     policies=None,
      #     reach_probabilities=np.ones(self._game.num_players() + 1),
      #     player=None)
      # if self._regret_matching_plus:
      #   _apply_regret_matching_plus_reset(self._info_state_nodes)
      # _update_current_policy(self._current_policy, self._info_state_nodes)


class CFRPlusSolver(_CFRSolver):
  """CFR+ implementation.

  The algorithm computes an approximate Nash policy for 2 player zero-sum games.
  More generally, it should approach a no-regret set, which corresponds to the
  set of coarse-correlated equilibria. See https://arxiv.org/abs/1305.0034

  CFR can be view as a policy iteration algorithm. Importantly, the policies
  themselves do not converge to a Nash policy, but their average does.

  See https://poker.cs.ualberta.ca/publications/2015-ijcai-cfrplus.pdf

  CFR+ is CFR with the following modifications:
  - use Regret Matching+ instead of Regret Matching.
  - use alternating updates instead of simultaneous updates.
  - use linear averaging.

  Usage:

  ```python
      game = pyspiel.load_game("game_name")
      initial_state = game.new_initial_state()

      cfr_solver = CFRSolver(game)

      for i in range(num_iterations):
        cfr.evaluate_and_update_policy()
  ```

  Once the policy has converged, the average policy (which converges to the Nash
  policy) can be computed:
  ```python
        average_policy = cfr_solver.ComputeAveragePolicy()
  ```
  """

  def __init__(self, game):
    super(CFRPlusSolver, self).__init__(
        game,
        regret_matching_plus=True,
        alternating_updates=True,
        linear_averaging=True)


class CFRSolver(_CFRSolver):
  """Implements the Counterfactual Regret Minimization (CFR) algorithm.

  See https://poker.cs.ualberta.ca/publications/NIPS07-cfr.pdf

  NOTE: We use alternating updates (which was not the case in the original
  paper) because it has been proved to be far more efficient.
  """

  def __init__(self, game):
    self.seq_id = 0
    super(CFRSolver, self).__init__(
        game,
        regret_matching_plus=False,
        alternating_updates=False,
        linear_averaging=False)
