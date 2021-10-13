import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import math

class Simu():
    def __init__(self):
        super(Simu, self).__init__()
        self.done = False
        self.current_position = None
        self.next_position = None
        self.current_direction = None
        self.all_positions = []
        self.corners_pool = np.array([[0,0],[1,0],[0,1],[1,1]])
        self.centre_pool = np.array([0.5,0.5])
        self.centre_border_pool = np.array([1,0.5])  # could be chosen at random among 4 centres of borders
        self.centre_platform = self.centre_pool + (self.centre_border_pool - self.centre_pool)/2
        self.corners_rectangel = np.array([[0.4,0.7],[0.6,0.7],[0.4,0.8],[0.6,0.8]])
        self.time = 0
        self.reward = 0

    def is_in_platform(self, point):
        # check if a point is in the plateform
        if (point[0] - self.centre_platform[0])**2 + (point[1] - self.centre_platform[1])**2 <= 0.1**2 :
            return True
        else:
            return False
    def hits_wall(self, point):
        if (point[0]>= 1 or point[0] <=0) or (point[1]>= 1 or point[1] <=0):# or (point[0] >= 0.4 and point[0]<= 0.6 and point[1]<=0.8 and point[1]>=0.7):
            return True
        else:
            return False
    def hits_rectangle(self,point):
        if (point[0] >= 0.4 and point[0]<= 0.6 and point[1]<=0.8 and point[1]>=0.7):
            return True
        else:
            return False

    def reinit(self):
        self.done = False
        self.time = 0
        self.reward = 0
        self.all_positions = []
        self.current_position = np.random.rand(2)
        while self.is_in_platform(self.current_position) or self.hits_wall(self.current_position):
            self.current_position = np.random.rand(2)
        self.all_positions.append(self.current_position)

    def step(self, action):
        self.time += 0.125
        action = action+1
        self.current_direction = 2*math.pi*action/8
        next_pos_x = self.current_position[0] + 0.02*math.cos(self.current_direction)
        next_pos_y = self.current_position[1] + 0.02*math.sin(self.current_direction)
        self.next_position = np.array([next_pos_x, next_pos_y])
        if self.hits_wall(self.next_position):
            self.reward = -5
            self.current_position = np.clip(self.next_position, 1e-10,1-1e-10)
        elif self.is_in_platform(self.next_position):
            self.done = True
            self.reward = 20
            self.current_position = self.next_position
        # elif self.hits_rectangle(self.next_position):
        #     self.reward = 0
        else:
            self.current_position = self.next_position
            # self.reward = -self.time/10
        self.all_positions.append(self.current_position)

        return  self.reward, self.done
    def pos_to_neurones(self):
        s = self.current_position*400
        s_x = int(s[0]//20)
        s_y = int(s[1]//20)
        # s = int(s_x*10 + s_y*10)
        s = int((s_y)*20+s_x)
        return s,s_x,s_y
    def point_to_neurones(self,point):
        s = point*400
        s_x = int(s[0]//20)
        s_y = int(s[1]//20)
        s = int((s_y)*20+s_x)
        return s,s_x,s_y

    def plot_trajectory(self,positions,name_file):
        positions=np.array(positions)
        x = positions[:,0]
        y = positions[:,1]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_facecolor((0, 0.5, 1, 0.3))
        major_ticks = np.linspace(0.0, 1.0, num = 21)
        ax.set_xticks(major_ticks, minor = True)
        ax.set_yticks(major_ticks, minor = True)
        ax.grid(which='both', alpha=0.3)
        platform = plt.Circle((self.centre_platform[0], self.centre_platform[1]), 0.1, color='black')
        # rect = patches.Rectangle((0.4, 0.7), 0.2, 0.1, color= 'purple')


        ax.add_patch(platform)
        # ax.add_patch(rect)

        plt.plot(x,y, color = "red")

        plt.xlim(0,1)
        plt.ylim(0,1)

        ax.set_aspect('equal', adjustable='box')
        plt.savefig(name_file + '.pdf',format = "pdf")
        plt.clf()

    def plot_from_Q(self,Q,time_lim,eps,name_file):
        self.reinit()
        print(Q)
        self.current_position = [0.5, 0.9]
        self.all_positions[-1] = self.current_position
        while (not self.done):
            if self.time>=time_lim:
                break
            s,_,_ = self.pos_to_neurones()
            if np.random.rand() < 0.2 :
                a = np.random.randint(8)    # action aleatoire, exploration
            else:
                a = np.argmax(Q[s,:])
            self.step(a)
        self.plot_trajectory(self.all_positions,name_file)

    def plot_cells_from_Q(self,Q, eps,name_file):

        self.reinit()
        self.current_position = [0.5, 0.9]
        self.all_positions[-1] = self.current_position

        while (not self.done):
            if self.time>=360:
                break
            s,_,_ = self.pos_to_neurones()
            if np.random.rand() < eps :
                a = np.random.randint(8)    # action aleatoire, exploration
            else:
                a = np.argmax(Q[s,:])
            self.step(a)





        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_facecolor((0, 0.5, 1, 0.3))
        major_ticks = np.linspace(0.0, 1.0, num = 21)
        ax.set_xticks(major_ticks, minor = True)
        ax.set_yticks(major_ticks, minor = True)
        ax.grid(which='both', alpha=0.3)

# And a corresponding grid
        platform = plt.Circle((self.centre_platform[0], self.centre_platform[1]), 0.1, color='black')
        # rect = patches.Rectangle((0.4, 0.7), 0.2, 0.1, color= 'purple')


        ax.add_patch(platform)
        # ax.add_patch(rect)

        for i in range(len(self.all_positions)):
            if i%50 == 0:
                pos = self.all_positions[i]
                pos = np.array(pos)
                s,s_x,s_y = self.point_to_neurones(pos)
                X = np.linspace(0, 0.05, 50)
                Y = np.linspace(0, 0.05, 50)
                X_to_mesh = pos[0] - X
                X_to_mesh = np.concatenate((X_to_mesh, pos[0] + X))
                X_to_mesh = np.clip(X_to_mesh,s_x*0.05, (s_x+1)*0.05 )
                Y_to_mesh = pos[1] - Y
                Y_to_mesh = np.concatenate((Y_to_mesh, pos[1] + Y))
                Y_to_mesh = np.clip(Y_to_mesh,s_y*0.05, (s_y+1)*0.05 )
                x,y = np.meshgrid(X_to_mesh,Y_to_mesh)
                f = np.exp((-(x - pos[0])**2-(y - pos[1])**2)/(2*(0.05**2)))
                plt.contour(x,y,f)


        plt.xlim(0,1)
        plt.ylim(0,1)

        ax.set_aspect('equal', adjustable='box')
        plt.savefig(name_file + '.pdf',format = "pdf")
        plt.clf()
