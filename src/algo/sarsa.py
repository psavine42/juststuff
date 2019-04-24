from collections import namedtuple
from collections import defaultdict as ddict


"""
http://incompleteideas.net/book/ebook/node67.html

base:
problem of predicting state-value function Vπ(s) or V
from experience generated using policy π

G = Return after n steps 

problem of predicting state-value function Vπ(s) or V
with action

V(s_t) <- V(s_t) + a(G_t - V(s_t) )

Updating Value

"""


def inc_mean(old_mean, k, new_val):
    return old_mean + (1/k) * (new_val - old_mean)


SarsaTransition = namedtuple(
    "Transition", ('state', 'action', 'reward', 'new_state', 'new_action')
)

QTransition = namedtuple(
    "QTransition", ('state', 'action', 'reward')
)

DQTransition = namedtuple(
    "QTransition", ('state', 'action', 'reward', 'new_state')
)


def td_zero(model, env, args):
    """
    Temporal Difference learning.

    1) make estimate
    2) do some actions
    3) update estimate

    Incremental mean update

    V(s_t) <- V(s_t) + a(R_t+1 + y*V(s_t+1) - V(S_t)))

    """
    for i in range(args.num_episodes):
        state = env.initialize()
        # action = model.policy(Q, state)
        for step in range(args.num_steps):
            reward, new_state = env.action(action)
            new_action = model.policy(Q, new_state)

            Q.update(QTransition(state, action, reward))

            state = new_state
            action = new_action

            model.optimizer()



def sarsa(Q, policy, optimizer, env, num_episodes, num_steps):
    """
    stub for SARSA
    eq:
    Q(s_t, a_t) <- Q(s_t, a_t) + a[r_t+1 + y*Q(s_t+1, a_t+1) - Q(s_t, a_t)

    """
    for i in range(num_episodes):
        state = env.initialize()
        action = policy(Q, state)
        for step in range(num_steps):

            reward, new_state = env.action(action)
            new_action = policy(Q, new_state)

            Q.update(SarsaTransition(state, action, reward, new_state, new_action))

            state = new_state
            action = new_action

            optimizer.step()
    return state, Q





def off_policy_td(Q, policy, optimizer, env, num_episodes, num_steps):
    """

    Q(s_t, a_t) <- Q(s_t, a_t) + a[r_t+1 + y*max_a* Q(s_t+1, a) - Q(s_t, a_t)

    """
    for i in range(num_episodes):
        state = env.initialize()
        action = policy(Q, state)
        for step in range(num_steps):

            reward, new_state = env.action(action)
            new_action = policy(Q, new_state)

            Q.update(QTransition(state, action, reward))

            state = new_state
            action = new_action

            optimizer.step()
    return Q


def rlearn():
    """
    """
    pass

def td_lambda():
    pass


def dynamic_policy():
    pass


def naive_monte_carlo(model, env, args):
    """
    N: counter of states

    V(s_t) <- V(s_t) + (G_t - V(s_t) )/N(s_t)


    """
    N = ddict(int)
    def update_q(state, reward, expected):
        p_delta = (reward - expected) / N[state]
        model.update(p_delta)



def glie_mc(model, env, args):
    """
    greedy in limit with infinte exploration
    all-state action are explored infinitely many times

    Policy will eventually become greedy.

    https://www.youtube.com/watch?v=0g4j2k_Ggc4

    Run MC, while tracking number of times a state is seen

    """
    N = ddict(int)
    def update_q(state, action, reward, expected):
        p_delta = (reward - expected) / N[(state, action)]
        model.update(p_delta)

    for i in range(args.num_episodes):
        state = env.initialize()
        action = model.policy(state)
        for step in range(args.num_steps):
            reward, new_state = env.action(action)
            pred_reward = model.reward(new_state)
            new_action = model.policy(new_state)

            N[(state, action)] += 1

            model.update(state, action, reward, pred_reward, new_state, new_action)

            state = new_state
            action = new_action
    print('complete')



def learn_dq(model, env, num_episodes, num_steps):
    """
    deep Q

    """
    for i in range(num_episodes):
        state = env.initialize()
        action = model.policy(state)
        for step in range(num_steps):
            reward, new_state = env.action(action)
            pred_reward = model.reward(new_state)
            new_action = model.policy(new_state)

            model.update(state, action, reward, pred_reward, new_state, new_action)

            state = new_state
            action = new_action
    return model


def zero_q(model, env, num_episodes, num_steps):
    """
    stub for alpha_zero thing

    """
    for i in range(num_episodes):
        state = env.initialize()
        action = model.policy(state)
        for step in range(num_steps):
            reward, new_state = env.action(action)
            pred_reward = model.reward(new_state)
            new_action = model.policy(new_state)

            model.update(state, action, reward, pred_reward, new_state, new_action)

            state = new_state
            action = new_action
    return model


