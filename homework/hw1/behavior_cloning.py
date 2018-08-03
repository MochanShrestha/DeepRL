import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import load_policy
import random


#Expert which will be called by playground. Will be called by playground
class Expert:
    def __init__(self, expertPolicyFile):
        self.policy = load_policy.load_policy(expertPolicyFile)
        print("Loaded expert policy")
    
    def react(self, observation):
        return self.policy(observation)

#Clone copies the behavior of the expert. Will be called by playground
class Clone:
    def __init__(self, dimension):
        self.model = Sequential()
        self.model.add(Dense(121, input_dim=dimension[0], activation='relu'))
        self.model.add(Dense(60, activation='relu'))
        self.model.add(Dense(29, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(11, activation='relu'))
        self.model.add(Dense(units=dimension[1]))
        self.model.compile(optimizer='sgd', loss='mse')
    
    def train(self, training_data, epochs, batch_size):
        loss =[]
        lossFunction = self.model.fit(training_data[0], training_data[1], epochs=epochs, batch_size=batch_size)
        loss.append(lossFunction.history['loss'])
        return loss

    def react(self, observation):
        return self.model.predict(observation, batch_size=5000, verbose=0)

class Playground:
    def __init__(self, environment):
        self.environment_name = environment
        self.initialize()
    
    def initialize(self):
        self.env = gym.make(self.environment_name)
    
    def simulate(self, agent, num_rollouts, max_timeSteps, render):
        with tf.Session():
            tf_util.initialize()
            observations = []
            actions = []
            max_timeSteps = max_timeSteps or self.env.spec.timestep_limit
            for i in range(num_rollouts):
                print('iter', i)
                obs = self.env.reset()
                done = False
                steps = 0
                sumreward = 0
                while not done:
                    action = agent.react(obs[None,:])
                    observations.append(obs)
                    actions.append(np.squeeze(action))
                    obs, r, done, _ = self.env.step(action)
                    sumreward += r
                    steps += 1
                    #done=False
                    if render:
                        self.env.render()
                    if steps >= max_timeSteps:
                        break
                print("Reward:" + str(sumreward))
            expert_observations = np.array(observations)
            expert_actions =  np.array(actions)
            return expert_observations, expert_actions
    
    def DAgger(self, expert, observations):
        with tf.Session():
            actions = []
            for obs in observations:
                action = expert.react(obs[None,:])
                actions.append(np.squeeze(action))
            expert_observations = np.array(observations)
            expert_actions =  np.array(actions)
            return expert_observations, expert_actions
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=2, help='Number of expert roll outs')
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    args = parser.parse_args()

    expertAgent = Expert(args.expert_policy_file)
    simulator = Playground(args.envname)
    expert_observations, expert_actions = simulator.simulate(expertAgent, args.num_rollouts, args.max_timesteps, False)    

    observationDimension = expert_observations.shape[-1]
    actionDimension = expert_actions.shape[-1]    
    dimensions = [observationDimension, actionDimension]
    clone = Clone(dimensions)    
    lossValue = clone.train([expert_observations, expert_actions], args.num_epochs, args.batch_size)

    #simulator = Playground(args.envname)
    agentObservations, agentActions = simulator.simulate(clone, args.num_rollouts, args.max_timesteps, False)
    
    cloneobservation, expertLabeledactions = simulator.DAgger(expertAgent, agentObservations)
    #name = input("Finished Running DAgger")
    print(cloneobservation.shape)
    print(expertLabeledactions.shape)

    print(expert_observations)
    print(expertLabeledactions)

    cloneobservation = np.concatenate((cloneobservation, expert_observations), axis=0)
    expertLabeledactions = np.concatenate((expertLabeledactions, expert_actions), axis=0)
    #print(shape(cloneobservation))
    #print(shape(expertLabeledactions))
    daggerIter = 10
    for i in range(daggerIter):
        #dimensions = [cloneobservation.shape[-1], expertLabeledactions.shape[-1]]
        #clone1 = Clone(dimensions)
        lossValue = clone.train([cloneobservation, expertLabeledactions], args.num_epochs, args.batch_size)

        agentObservations1, agentActions1 = simulator.simulate(clone, args.num_rollouts, args.max_timesteps, True)
        
        cloneobservation1, expertLabeledactions1 = simulator.DAgger(expertAgent, agentObservations1)
        
        cloneobservation = np.concatenate((cloneobservation, cloneobservation1), axis=0)
        expertLabeledactions = np.concatenate((expertLabeledactions, expertLabeledactions1), axis=0)

        print(cloneobservation.shape)
        print(expertLabeledactions.shape)
            

    name = input("Done")


if __name__ == '__main__':
    main()