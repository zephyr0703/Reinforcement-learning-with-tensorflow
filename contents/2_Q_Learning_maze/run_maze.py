import numpy as np
import pandas as pd

from maze import Maze

ACTIONS = ['up', 'down', 'left', 'right']
EPSILON = 0.9   # 贪婪度 greedy
ALPHA = 0.1     # 学习率
GAMMA = 0.9    # 奖励递减值

class QLearning(object):
    def __init__(self, actions):
        self._actions = actions
        self._q_table = pd.DataFrame(
            columns=actions,  # columns 对应的是行为名称
            dtype=np.float64
        )
    
    def choose_action(self, state):
        self._check_state_exist(state)
        # 在某个 state 地点, 选择行为
        state_actions = self._q_table.loc[state, :]  # 选出这个 state 的所有 action 值
        if np.random.uniform() > EPSILON or state_actions.all() == 0:  # 非贪婪 or 或者这个 state 还没有探索过
            action = np.random.choice(self._actions)
        else:
            action = state_actions.idxmax()  # 贪婪模式
        return action
    
    def learn(self, S, A, S_, R, done):
        q_predict = self._q_table.loc[S, A]  # 估算的(状态-行为)值
        if not done:
            self._check_state_exist(S_)
            q_target = R + GAMMA * self._q_table.loc[S_, :].max()  # 实际的(状态-行为)值 (回合没结束)
        else:
            q_target = R  # 实际的(状态-行为)值 (回合结束)
        self._q_table.loc[S, A] += ALPHA * (q_target - q_predict)  #  q_table 更新
    
    def _check_state_exist(self, state):
        if state not in self._q_table.index:
            # append new state to q table
            self._q_table = self._q_table.append(
                pd.Series(
                    [0] * len(self._actions),
                    index=self._q_table.columns,
                    name=state,
                )
            )
    
    def get_q_table(self):
        return self._q_table


def run():
    for t in range(100):
        s = env.reset()  # initial observation
        print('round:{} coords:{} start...'.format(t, s))
        num_step = 0
        while True:
            env.render()  # fresh env
            # RL choose action based on observation
            a = RL.choose_action(str(s))
            # RL take action and get next observation and reward
            s_, r, done = env.step(a)
            # RL learn from this transition
            RL.learn(str(s), a, str(s_), r, done)
            s = s_  # swap observation
            num_step += 1
            if done:
                result = 'win' if r > 0 else 'lose'
                print('round:{} steps:{} coords:{} {}.'.format(t, num_step, s, result))
                break
    print('Q-table:\n{}'.format(RL.get_q_table()))
    env.destroy()
    # end of game
    print('game over')

if __name__ == '__main__':
    env = Maze()
    RL = QLearning(actions=ACTIONS)
    env.after(100, run)
    env.mainloop()