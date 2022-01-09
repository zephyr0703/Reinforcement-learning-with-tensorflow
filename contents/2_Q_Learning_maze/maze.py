import numpy as np
import time
import tkinter as tk

from numpy.lib.function_base import append

UNIT = 80   # pixels
MAZE_H = 4  # grid height
MAZE_W = 4  # grid width
ACTIONS = ['up', 'down', 'left', 'right']

class Maze(tk.Tk):
    def __init__(self):
        super(Maze, self).__init__()
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_maze()
    
    def _build_maze(self):
        self.canvas = tk.Canvas(self,
            bg='white', height=MAZE_H * UNIT, width=MAZE_W * UNIT)
        # create grids
        for c in range(UNIT, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            # print('w:{} {} {} {}'.format(x0, y0, x1, y1))
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(UNIT, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            # print('h:{} {} {} {}'.format(x0, y0, x1, y1))
            self.canvas.create_line(x0, y0, x1, y1)
        # create origin
        origin = np.array([UNIT / 2, UNIT / 2])
        offset = UNIT / 2 - UNIT / 16
        # create rect
        self.rect = self.canvas.create_rectangle(
            origin[0] - offset, origin[1] - offset,
            origin[0] + offset, origin[1] + offset,
            fill='red')
        # create oval
        oval_center = origin + UNIT * 2
        oval = self.canvas.create_oval(
            oval_center[0] - offset, oval_center[1] - offset,
            oval_center[0] + offset, oval_center[1] + offset,
            fill='yellow')
        self.oval_coords = self.canvas.coords(oval)
        # create hells
        hell_centers = []
        hell_centers.append(origin + np.array([UNIT * 2, UNIT]))
        hell_centers.append(origin + np.array([UNIT, UNIT * 2]))
        self.hell_coords = []
        for center in hell_centers:
            hell = self.canvas.create_rectangle(
                center[0] - offset, center[1] - offset,
                center[0] + offset, center[1] + offset,
                fill='black')
            self.hell_coords.append(self.canvas.coords(hell))
        # pack all
        self.canvas.pack()
    
    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        # create origin
        origin = np.array([UNIT / 2, UNIT / 2])
        offset = UNIT / 2 - UNIT / 16
        # create rect
        self.rect = self.canvas.create_rectangle(
            origin[0] - offset, origin[1] - offset,
            origin[0] + offset, origin[1] + offset,
            fill='red')
        return self.canvas.coords(self.rect)

    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 'up':
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 'down':
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 'left':
            if s[0] > UNIT:
                base_action[0] -= UNIT
        elif action == 'right':
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent
        s_ = self.canvas.coords(self.rect)  # next state
        # reward function
        if s_ == self.oval_coords:
            r = 1
            done = True
        elif s_ in self.hell_coords:
            r = -1
            done = True
        else:
            r = 0
            done = False
        return s_, r, done

    def render(self):
        time.sleep(0.1)
        self.update()


def run():
    for t in range(10):
        s = env.reset()
        print('round:{} coords:{} start...'.format(t, s))
        num_step = 0
        while True:
            env.render()
            a = np.random.choice(ACTIONS)
            s, r, done = env.step(a)
            num_step += 1
            if done:
                result = 'win' if r > 0 else 'lose'
                print('round:{} steps:{} coords:{} {}.'.format(t, num_step, s, result))
                break
    env.destroy()


if __name__ == '__main__':
    env = Maze()
    env.after(100, run)
    env.mainloop()
