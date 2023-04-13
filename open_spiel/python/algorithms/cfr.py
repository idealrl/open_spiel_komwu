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
    self._current_policy = policy.TabularPolicy(game)
    self._average_policy = self._current_policy.__copy__()

    # self._info_state_nodes = {}
    # self._initialize_info_state_nodes(self._root_node)

    self._info_state_nodes_komwu = [{} for _ in range(self._num_players)]
    global LID
    global INFOID
    p = 0
    
    self.seq_ID = 0
    LID = 0
    INFOID = 0
    self.ID = 0 # infoset IDs
    # Arguments(state, last_action, player_id, parent_infoset)
    self._initialize_info_state_nodes_komwu(self._root_node, -1, p,None)
    self._initialize_children_komwu(self._root_node, -1, p, None)
    b0 = [0 for _ in range(LID)]

    p = 1
    self.seq_id = 0 # Reset to make sequences
    self.ID = 0
    INFOID = 0
    LID = 0
    self._initialize_info_state_nodes_komwu(self._root_node, -1, p,None)
    self._initialize_children_komwu(self._root_node, -1, p, None)
    b1 = [0 for _ in range(LID)]

    #print("LID: ", LID)
    #self.seq_id = LID
    # self.grad = [[0 for _ in range(self.seq_id)] for _ in range(self._num_players)]
    # self.last_grad = [[0 for _ in range(self.seq_id)] for _ in range(self._num_players)]
    

    self.b = [b0, b1]#[[0 for _ in range(self.seq_id)] for _ in range(self._num_players)]
    self.grad = [b0, b1]
    self.last_grad = [b0, b1]

    p0 = []
    p1 = []
    self._root_node = self._game.new_initial_state()

    # print("-----State state--------")
    # state = self._root_node
    # print(state)
    # current_player = state.current_player()
    # print("  Current Player: ", current_player)
    # print("  Legal actions:")
    # print("  ", state.legal_actions())
    # print("-------------------------")

    # act = state.legal_actions()[1]
    # print("Taking action: ", act)

    # print("-----Next state--------")
    # state = state.child(act)
    # print(state)
    # current_player = state.current_player()
    # print("  Current Player: ", current_player)
    # print("  Legal actions:")
    # print("  ", state.legal_actions())
    # print("-------------------------")

    # act = state.legal_actions()[1]
    # print("Taking action: ", act)

    # print("-----Next state--------")
    # state = state.child(act)
    # print(state)
    # current_player = state.current_player()
    # print("  Current Player: ", current_player)
    # print("  Legal actions:")
    # print("  ", state.legal_actions())
    # print("-------------------------")

    # # player 0
    # act = state.legal_actions()[1]
    # print("Taking action: ", act)
    # info_state = state.information_state_string(current_player)
    # info_state_node = self._info_state_nodes_komwu[current_player].get(info_state)
    # seq = self._info_state_nodes_komwu[current_player][info_state].actions_to_sequences[act]
    # sti = self._info_state_nodes_komwu[current_player][info_state].sequence_to_infoset[seq]
    # x = [sti[i].sequences for i in range(len(sti))]
    # print(x)
    # print("ID:")
    # print(info_state_node.ID)
    # x = [sti[i].ID for i in range(len(sti))]
    # print(x)
    # print(sti[1].ID)
    # # print(sti[0].sequences, len(sti))#, sti)
    # p0.append(seq)
    # print("Seq: ", seq, " Parent seq: ", info_state_node.parent_seq, " Player ", current_player, " seqs: ", p0)


    # print("-----Next state--------")
    # state = state.child(act)
    # print(state)
    # current_player = state.current_player()
    # print("  Current Player: ", current_player)
    # print("  Legal actions:")
    # print("  ", state.legal_actions())
    # print("-------------------------")

    # # player 1
    # act = state.legal_actions()[1]
    # print("Taking action: ", act)
    # info_state = state.information_state_string(current_player)
    # info_state_node = self._info_state_nodes_komwu[current_player].get(info_state)
    # seq = self._info_state_nodes_komwu[current_player][info_state].actions_to_sequences[act]
    # p1.append(seq)
    # print("Seq: ", seq, " Parent seq: ", info_state_node.parent_seq, " Player ", current_player, " seqs: ", p1)

    # print("-----Next state--------")
    # state = state.child(act)
    # print(state)
    # current_player = state.current_player()
    # print("  Current Player: ", current_player)
    # print("  Legal actions:")
    # print("  ", state.legal_actions())
    # print("-------------------------")

    # # Chance (community card)
    # act = state.legal_actions()[1]
    # print("Taking action: ", act)

    # print("-----Next state--------")
    # state = state.child(act)
    # print(state)
    # current_player = state.current_player()
    # print("  Current Player: ", current_player)
    # print("  Legal actions:")
    # print("  ", state.legal_actions())
    # print("-------------------------")

    # # player 0
    # act = state.legal_actions()[1]
    # print("Taking action: ", act)
    # info_state = state.information_state_string(current_player)
    # info_state_node = self._info_state_nodes_komwu[current_player].get(info_state)
    # seq = self._info_state_nodes_komwu[current_player][info_state].actions_to_sequences[act]
    # p0.append(seq)
    # print("ID:")
    # print(info_state_node.ID)
    # print("Seq: ", seq, " Parent seq: ", info_state_node.parent_seq, " Player ", current_player, " seqs: ", p0)

    # print("-----Next state--------")
    # state = state.child(act)
    # print(state)
    # current_player = state.current_player()
    # print("  Current Player: ", current_player)
    # print("  Legal actions:")
    # print("  ", state.legal_actions())
    # print("-------------------------")

    # # player 1
    # act = state.legal_actions()[1]
    # print("Taking action: ", act)
    # info_state = state.information_state_string(current_player)
    # info_state_node = self._info_state_nodes_komwu[current_player].get(info_state)
    # seq = self._info_state_nodes_komwu[current_player][info_state].actions_to_sequences[act]
    # p1.append(seq)
    # print("Seq: ", seq, " Parent seq: ", info_state_node.parent_seq, " Player ", current_player, " seqs: ", p1)

    # print("-----Next state--------")
    # state = state.child(act)
    # print(state)
    # current_player = state.current_player()
    # print("  Current Player: ", current_player)
    # print("  Legal actions:")
    # print("  ", state.legal_actions())
    # print("-------------------------")

    self._compute_x(t=0)

    self._iteration = 0  # For possible linear-averaging.
    self._linear_averaging = linear_averaging
    self._alternating_updates = alternating_updates
    self._regret_matching_plus = regret_matching_plus

    

  def _initialize_info_state_nodes(self, state):
    """Initializes info_state_nodes.

    Create one _InfoStateNode per infoset. We could also initialize the node
    when we try to access it and it does not exist.

    Args:
      state: The current state in the tree walk. This should be the root node
        when we call this function from a CFR solver.
    """
    if state.is_terminal():
      return

    if state.is_chance_node():
      for action, unused_action_prob in state.chance_outcomes():
        self._initialize_info_state_nodes(state.child(action))
      return

    current_player = state.current_player()
    info_state = state.information_state_string(current_player)

    info_state_node = self._info_state_nodes.get(info_state)
    if info_state_node is None:
      legal_actions = state.legal_actions(current_player)
      info_state_node = _InfoStateNode(
          legal_actions=legal_actions,
          index_in_tabular_policy=self._current_policy.state_lookup[info_state],
          parent_infoset=None,
          parent_seq=None,
          states=None,
          sequences=None,
          children=None,
          children_other_players=None,
          actions_to_sequences=None,
          sequence_to_infoset=None,
          ID=None,
          player=None
          )
      self._info_state_nodes[info_state] = info_state_node

    for action in info_state_node.legal_actions:
      self._initialize_info_state_nodes(state.child(action))

        
        
        
  def _initialize_info_state_nodes_komwu(self, state, last_seq, player_id, parent_infoset):

    if state.is_terminal():
      # Collect terminals?
      return

    if state.is_chance_node():
      # Check later, Leduc has multiple chance nodes
      for action, unused_action_prob in state.chance_outcomes():
        self._initialize_info_state_nodes_komwu(state.child(action), last_seq, player_id, parent_infoset)
      return
     
    current_player = state.current_player()
    info_state = state.information_state_string(current_player)
    info_state_node = self._info_state_nodes_komwu[player_id].get(info_state)
  
    # Collect only infosets that are player_id
    # If not at the current player_id, keep going
    if current_player != player_id:
      for action in state.legal_actions(current_player):
        if state.child(action) != None:
          self._initialize_info_state_nodes_komwu(state.child(action), last_seq, player_id, parent_infoset)
  
    # Are current player
    else:
      global INFOID
      if info_state_node is None:
        legal_actions = state.legal_actions(current_player)
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
            player=current_player
            )
        global LID
        INFOID += 1
        # print("Info_state: ", info_state, " par_seq: ", last_seq, " THIS SEQ: ", LID)
        self._info_state_nodes_komwu[player_id][info_state] = info_state_node
        for action in info_state_node.legal_actions:
          self._info_state_nodes_komwu[player_id][info_state].sequences.append(LID)
          self._info_state_nodes_komwu[player_id][info_state].actions_to_sequences[action] = LID
          self._info_state_nodes_komwu[player_id][info_state].sequence_to_infoset[LID] = []
          cur_seq = LID
          LID += 1
          # print("Info_state: ", info_state, " par_seq: ", last_seq, " THIS SEQ: ", cur_seq)
          self._initialize_info_state_nodes_komwu(state.child(action), cur_seq, player_id, info_state_node)
          
      # If infoset already exists, append history
      else:
        self._info_state_nodes_komwu[player_id][info_state].states.append(state)
        for action in info_state_node.legal_actions:
          action_to_seq = self._info_state_nodes_komwu[player_id][info_state].actions_to_sequences[action]
          self._initialize_info_state_nodes_komwu(state.child(action), action_to_seq, player_id, info_state_node)


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
            # print("Setting ", parent_infoset, " now has: ", self._info_state_nodes_komwu[player_id][parent_infoset].sequence_to_infoset[seq])
          self._info_state_nodes_komwu[player_id][parent_infoset].children.append(info_state_node)

        
      for action in info_state_node.legal_actions:
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
    _update_average_policy(self._average_policy, self._info_state_nodes_komwu[0])
    _update_average_policy(self._average_policy, self._info_state_nodes_komwu[1])
    
    return self._average_policy

  def _compute_grad(self, state, policies, reach_probabilities, player, seqs):
    
    if state.is_terminal():
      
      # Update player 0
      reach_prob = reach_probabilities[0]
      counterfactual_reach_prob = (
        np.prod(reach_probabilities[:0]) *
        np.prod(reach_probabilities[0 + 1:]))

      # print(state.returns()[0] * counterfactual_reach_prob) self.y[0][seqs[0]]
      self.grad[0][seqs[0]] += state.returns()[0] * counterfactual_reach_prob

      # Update player 1
      reach_prob = reach_probabilities[1]
      counterfactual_reach_prob = (
        np.prod(reach_probabilities[:1]) *
        np.prod(reach_probabilities[1 + 1:]))
      self.grad[1][seqs[1]] += state.returns()[1] * counterfactual_reach_prob
      return np.asarray(state.returns())

    if state.is_chance_node():
      state_value = 0.0
      for action, action_prob in state.chance_outcomes():
        assert action_prob > 0
        new_state = state.child(action)
        new_reach_probabilities = reach_probabilities.copy()
        new_reach_probabilities[-1] *= action_prob
        state_value += action_prob * self._compute_grad(
            new_state, policies, new_reach_probabilities, player, seqs)
      return state_value

    current_player = state.current_player()
    info_state = state.information_state_string(current_player)

    # if all(reach_probabilities[:-1] == 0):
    #   return np.zeros(self._num_players)

    state_value = np.zeros(self._num_players)

    # The utilities of the children states are computed recursively. As the
    # regrets are added to the information state regrets for each state in that
    # information state, the recursive call can only be made once per child
    # state. Therefore, the utilities are cached.
    children_utilities = {}

    info_state_node = self._info_state_nodes_komwu[current_player][info_state]
    if policies is None:
      info_state_policy = self._get_infostate_policy_komwu(info_state, current_player)
    else:
      info_state_policy = policies[current_player](info_state)
    i = 0
    for action in state.legal_actions():
      action_prob = info_state_policy.get(action, 0.)
      new_state = state.child(action)
      new_reach_probabilities = reach_probabilities.copy()
      new_reach_probabilities[current_player] *= action_prob
      # seq = self._info_state_nodes_komwu[current_player][info_state].sequences[i]
      seq = self._info_state_nodes_komwu[current_player][info_state].actions_to_sequences[action]      
      # i += 1
      seqs[current_player] = seq
      child_utility = self._compute_grad(
          new_state,
          policies=policies,
          reach_probabilities=new_reach_probabilities,
          player=player,
          seqs=seqs)

   
    return state_value


  def _komwu(self, t):
    
    # seq_id was built up to the total number of seqs
    g0 = [0 for _ in range(len(self.b[0]))]
    g1 = [0 for _ in range(len(self.b[1]))]
    self.grad = [g0, g1]#[[0 for _ in range(self.seq_id)] for _ in range(self._num_players)]
    
    # state, policies, reach_probabilities, player, seqs
    # Should do lazy cfr check here and collect infosets that will be affected
    # And only use them in _compute_x
    self._compute_grad(self._root_node,
            policies=None,
            reach_probabilities=np.ones(self._game.num_players() + 1),
            player=None,
            seqs=[-1,-1])
   
    # 5 3232
    # 6 2610
    # 7 2475
    # 8 1852
    # 18 767
    # 22 633
    # 50 485
    for player_id in range(self._num_players):
      opt = 3.0
      opt_grad = [opt * self.grad[player_id][i] - (opt-1.0) * self.last_grad[player_id][i] for i in range(len(self.grad[player_id]))]
      for i in range(len(self.b[player_id])):
        eta = 1.0 / 10.0 # eta <= 1/8
        self.b[player_id][i] += eta * opt_grad[i]
    self.last_grad = self.grad
  
    self._compute_x(t)

  def _compute_x(self,t):  

    self.y = [[],[]]  
    # Step 1
    y = [[0 for _ in range(len(self.b[i]))] for i in range(self._num_players)]
    self.y = [[0 for _ in range(len(self.b[i]))] for i in range(self._num_players)]  
    for player_id in range(self._num_players):

      K_j = [None] * len(self._info_state_nodes_komwu[player_id])
      
      for info_str, infoset in reversed(list(self._info_state_nodes_komwu[player_id].items())):
        
        seq_values = []
        for seq in infoset.sequences:
          child_values = []
          for child_infoset in self._info_state_nodes_komwu[player_id][info_str].sequence_to_infoset[seq]:
            child_values.append(K_j[child_infoset.ID])
          seq_value = self.b[player_id][seq] + sum(child_values)
          seq_values.append(seq_value)
        K_j[infoset.ID] = logsumexp(seq_values)
        # print("K_j: ", "(", info_str, ")", K_j[infoset.ID])
      # print("")

      # Step 3
      for info_str, infoset in self._info_state_nodes_komwu[player_id].items():
        for seq in infoset.sequences:
          if infoset.parent_seq == -1:
            y_new = 0
          else:
            y_new = y[player_id][infoset.parent_seq]
          y[player_id][seq] = self.b[player_id][seq] + y_new 
          child_values = []
          for child_infoset in self._info_state_nodes_komwu[player_id][info_str].sequence_to_infoset[seq]:
            child_values.append(K_j[child_infoset.ID])
          y[player_id][seq] = y[player_id][seq] + sum(child_values) - K_j[infoset.ID]
      
    self.y[0] = np.exp(y[0])
    self.y[1] = np.exp(y[1])
    
    for player_id in range(self._num_players):
      for info_str, infoset in self._info_state_nodes_komwu[player_id].items():
        denom = 0
        for action, seq in enumerate(infoset.sequences):
          denom += self.y[player_id][seq]
        for action, seq in enumerate(infoset.sequences):
          self._current_policy.action_probability_array[
          infoset.index_in_tabular_policy][action] = self.y[player_id][seq] / denom
          # infoset.cumulative_policy[action] += y[player_id][seq]
    
    # print(self.y)    
      # print(self._current_policy.action_probability_array)
      # print("")
      # print("yyyyyyyyyy")
      # print(y)
      
    
    

      # Next, use y to set the policy
      # for info_str, infoset in self._info_state_nodes_komwu[player_id].items():
      #   info_state_node = infoset
      #   denom = 0
      #   for seq in infoset.sequences:
      #     denom += np.exp(y[player_id][seq])
      #   for action, seq in enumerate(infoset.sequences):
      #     normal = np.exp(y[player_id][seq]) / denom
      #     self._current_policy.action_probability_array[
      #     info_state_node.index_in_tabular_policy][action] = normal
          

  def _compute_counterfactual_regret_for_player(self, state, policies,
                                                reach_probabilities, player):
    """Increments the cumulative regrets and policy for `player`.

    Args:
      state: The initial game state to analyze from.
      policies: A list of `num_players` callables taking as input an
        `info_state_node` and returning a {action: prob} dictionary. For CFR,
          this is simply returning the current policy, but this can be used in
          the CFR-BR solver, to prevent code duplication. If None,
          `_get_infostate_policy` is used.
      reach_probabilities: The probability for each player of reaching `state`
        as a numpy array [prob for player 0, for player 1,..., for chance].
        `player_reach_probabilities[player]` will work in all cases.
      player: The 0-indexed player to update the values for. If `None`, the
        update for all players will be performed.

    Returns:
      The utility of `state` for all players, assuming all players follow the
      current policy defined by `self.Policy`.
    """
    if state.is_terminal():
      return np.asarray(state.returns())

    if state.is_chance_node():
      state_value = 0.0
      for action, action_prob in state.chance_outcomes():
        assert action_prob > 0
        new_state = state.child(action)
        new_reach_probabilities = reach_probabilities.copy()
        new_reach_probabilities[-1] *= action_prob
        state_value += action_prob * self._compute_counterfactual_regret_for_player(
            new_state, policies, new_reach_probabilities, player)
      return state_value

    current_player = state.current_player()
    info_state = state.information_state_string(current_player)

    # No need to continue on this history branch as no update will be performed
    # for any player.
    # The value we return here is not used in practice. If the conditional
    # statement is True, then the last taken action has probability 0 of
    # occurring, so the returned value is not impacting the parent node value.
    if all(reach_probabilities[:-1] == 0):
      return np.zeros(self._num_players)

    state_value = np.zeros(self._num_players)

    # The utilities of the children states are computed recursively. As the
    # regrets are added to the information state regrets for each state in that
    # information state, the recursive call can only be made once per child
    # state. Therefore, the utilities are cached.
    children_utilities = {}

    info_state_node = self._info_state_nodes[info_state]
    if policies is None:
      info_state_policy = self._get_infostate_policy(info_state)
    else:
      info_state_policy = policies[current_player](info_state)
    for action in state.legal_actions():
      action_prob = info_state_policy.get(action, 0.)
      new_state = state.child(action)
      new_reach_probabilities = reach_probabilities.copy()
      new_reach_probabilities[current_player] *= action_prob
      child_utility = self._compute_counterfactual_regret_for_player(
          new_state,
          policies=policies,
          reach_probabilities=new_reach_probabilities,
          player=player)

      state_value += action_prob * child_utility
      children_utilities[action] = child_utility

    # If we are performing alternating updates, and the current player is not
    # the current_player, we skip the cumulative values update.
    # If we are performing simultaneous updates, we do update the cumulative
    # values.
    simulatenous_updates = player is None
    if not simulatenous_updates and current_player != player:
      return state_value

    reach_prob = reach_probabilities[current_player]
    counterfactual_reach_prob = (
        np.prod(reach_probabilities[:current_player]) *
        np.prod(reach_probabilities[current_player + 1:]))
    state_value_for_player = state_value[current_player]

    for action, action_prob in info_state_policy.items():
      cfr_regret = counterfactual_reach_prob * (
          children_utilities[action][current_player] - state_value_for_player)

      info_state_node.cumulative_regret[action] += cfr_regret
      if self._linear_averaging:
        info_state_node.cumulative_policy[
            action] += self._iteration * reach_prob * action_prob
      else:
        info_state_node.cumulative_policy[action] += reach_prob * action_prob

    return state_value

  def _get_infostate_policy(self, info_state_str):
    """Returns an {action: prob} dictionary for the policy on `info_state`."""
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
      x = 1
      self._komwu(self._iteration)
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
