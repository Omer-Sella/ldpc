import numpy as np
import torch
torch.cuda.is_available = lambda : False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.optim import Adam
import gym
import time
import os
import copy
projectDir = os.environ.get('LDPC')
if projectDir == None:
    import pathlib
    projectDir = pathlib.Path(__file__).parent.absolute()
import openAIcore as core
from logx import EpochLogger
#from mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
#from mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from utilityFunctions import plotter as ossplotter
from utilityFunctions import logger as osslogger
#OSS 26/11/2021 moved beffur to a separate module so importing everything from there.
from buffer import *

#OSS trying spawn instead of fork
import multiprocessing




OBSERVATION_DATA_TYPE = np.float32
INTERNAL_ACTION_DATA_TYPE = np.float32

#seed = 7134066
#localRandom = np.random.RandomState(seed)
#maximumEpisodeLength = 3
clipRatio = 0.2
policyLearningRate = 3e-4
valueFunctionLearningRate = 1e-3
#loggerKeyWords = ['value', 'EpRet', 'Episode length', ]
policyTrainIterations = 80
targetKL = 1.5 * 0.01
valueFunctionTrainIterations = 80
loggerPath = str(projectDir) + "/temp/"
MAXIMUM_NUMBER_OF_HOT_BITS = 7
INTERNAL_ACTION_SPACE_SIZE = 1 + 1 + 1 + MAXIMUM_NUMBER_OF_HOT_BITS
SAVE_MODEL_FREQUENCY = 10
NUMBER_OF_GPUS_PER_NODE = 2

# Number of entropy elements is depends on the model. At the this time we have i,j,k and we don't include entropy for number of coordinates selection.
NUMBER_OF_ENTROPY_ELEMENTS = 3 
OPEN_AI_PPO_NUMBER_OF_BUFFERS = 1

import models


# OSS 26/11/2021 I'm temporarily placin the buffer function under comment, since I'm migrating it to a separate module
#class PPOBuffer:
#    """
#    A buffer for storing trajectories experienced by a PPO agent interacting
#    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
#    for calculating the advantages of state-action pairs.
#    """

#    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
#        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
#        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
#        self.adv_buf = np.zeros(size, dtype=np.float32)
#        self.rew_buf = np.zeros(size, dtype=np.float32)
#        self.ret_buf = np.zeros(size, dtype=np.float32)
#        self.val_buf = np.zeros(size, dtype=np.float32)
#        self.ent_buf = np.zeros(size, dtype=np.float32)
#        self.entropyList_buf = np.zeros((size, NUMBER_OF_ENTROPY_ELEMENTS), dtype=np.float32)
        
#        self.logp_buf = np.zeros(size, dtype=np.float32)
#        self.gamma, self.lam = gamma, lam
#        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

#    def store(self, obs, act, rew, val, logp, ent, entropyArray):
#        """
#        Append one timestep of agent-environment interaction to the buffer.
#        """
#        assert self.ptr < self.max_size     # buffer has to have room so you can store
#        self.obs_buf[self.ptr] = obs
#        self.act_buf[self.ptr] = act
#        self.rew_buf[self.ptr] = rew
#        self.val_buf[self.ptr] = val
#        self.logp_buf[self.ptr] = logp
#        self.ent_buf[self.ptr] = ent
#        self.entropyList_buf[self.ptr] = entropyArray
#        self.ptr += 1

#    def finish_path(self, last_val=0):
#        """
#        Call this at the end of a trajectory, or when one gets cut off
#        by an epoch ending. This looks back in the buffer to where the
#        trajectory started, and uses rewards and value estimates from
#        the whole trajectory to compute advantage estimates with GAE-Lambda,
#        as well as compute the rewards-to-go for each state, to use as
#        the targets for the value function.

#        The "last_val" argument should be 0 if the trajectory ended
#        because the agent reached a terminal state (died), and otherwise
#        should be V(s_T), the value function estimated for the last state.
#        This allows us to bootstrap the reward-to-go calculation to account
#        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
#        """

#        path_slice = slice(self.path_start_idx, self.ptr)
#        rews = np.append(self.rew_buf[path_slice], last_val)
#        vals = np.append(self.val_buf[path_slice], last_val)
        
#        # the next two lines implement GAE-Lambda advantage calculation
#        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
#        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
#        # the next line computes rewards-to-go, to be targets for the value function
#        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
#        self.path_start_idx = self.ptr

#    def get(self):
#        """
#        Call this at the end of an epoch to get all of the data from
#        the buffer, with advantages appropriately normalized (shifted to have
#        mean zero and std one). Also, resets some pointers in the buffer.
#        """
#        assert self.ptr == self.max_size    # buffer has to be full before you can get
#        self.ptr, self.path_start_idx = 0, 0
#        # the next two lines implement the advantage normalization trick
#        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
#        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
#        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
#                    adv=self.adv_buf, logp=self.logp_buf, ent=self.ent_buf)
#        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}



def ppo(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        #Omer Sella: I replaced this: steps_per_epoch=4000, with this:
        steps_per_epoch=64,
        epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, 
        max_ep_len=1000,
        target_kl=0.01, logger_kwargs=dict(), 
        save_freq=10, envCudaDevices = 4, experimentDataDir = None,
        entropyCoefficient0 = 0.01, entropyCoefficient1 = 0.01, entropyCoefficient2 = 0.01, entropyCoefficientNoAction = 0.01, policyCoefficient = 1.0, resetType = 'WORST_CODES', actionInvalidator = 'Disabled'):
    """
    Proximal Policy Optimization (by clipping), 

    with early stopping based on approximate KL

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a 
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` 
            module. The ``step`` method should accept a batch of observations 
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of 
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing 
                                           | the log probability, according to 
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical: 
                                           | make sure to flatten this!)
            ===========  ================  ======================================


        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while 
            still profiting (improving the objective function)? The new policy 
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`. 

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take 
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """


    # Special function to avoid certain slowdowns from PyTorch + MPI combo.  
    #setup_pytorch_for_mpi() #OSS 07/01/2022 commented this since no mpi will be used.

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())


    # Omer Sella: this is my logger and plotter:
    if actionInvalidator == 'Enabled':
        simpleKeys = ['Observation', 'iAction', 'jAction', 'kAction', 'hotBitsAction', 'Reward', 'epochNumber', 'stepNumber', 'actorEntropy', 'logP', 'logpI', 'logpJ', 'logpK', 'iEntropy', 'jEntropy', 'kEntropy', 'coordinatesEntropy', 'vValue', 'noAction']
    else:
        simpleKeys = ['Observation', 'iAction', 'jAction', 'kAction', 'hotBitsAction', 'Reward', 'epochNumber', 'stepNumber', 'actorEntropy', 'logP', 'logpI', 'logpJ', 'logpK', 'iEntropy', 'jEntropy', 'kEntropy', 'coordinatesEntropy', 'vValue']
    myLogger = osslogger(keys = simpleKeys, logPath = experimentDataDir, fileName = str(seed))
    #logger.save_config(locals())
    myPlotter = ossplotter(50)

    # Random seed
    seed += 10000 #OSS: 10/01/2022 removed: * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)


    # Instantiate environment
    #Omer Sella: I changed the environment seed to be the same as the ppo seed.
    env = env_fn(x = seed, y = envCudaDevices, z = resetType)
    obs_dim = env.observation_space.shape
    if actionInvalidator == 'Enabled':
        act_dim = 1 + 1 + 1 + MAXIMUM_NUMBER_OF_HOT_BITS + 1#env.action_space.shape
    else:
        act_dim = 1 + 1 + 1 + MAXIMUM_NUMBER_OF_HOT_BITS #env.action_space.shape

    # Create actor-critic module
    #OVERRIDE_OBSERVATION_SPACE_DIM = np.zeros(2048)
    #OVERRIDE_ACTION_SPACE_DIM = np.zeros(516)
    
    # Omer Sella: this is where I need to plant my AC
    #ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    #ac = actor_critic(OVERRIDE_OBSERVATION_SPACE_DIM, OVERRIDE_ACTION_SPACE_DIM, **ac_kwargs)
    ac = models.openAIActorCritic(int, 2048, int, INTERNAL_ACTION_SPACE_SIZE, 64, MAXIMUM_NUMBER_OF_HOT_BITS, [64,64] , actorCriticDevice = 'cpu', actionInvalidator = actionInvalidator)
    # Sync params across processes
    #sync_params(ac) #OSS 07/01/2022 commenting out since removing mpi functionality

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch) #OSS 10/01/2022 removed mpi stuff / num_procs())
    #OSS 29/11/2021 commented the buffer and switched to buffer container
    #OSS 07/01/2021 reinstated the regular buffer in an attempt to isolate conc fututres bug.
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)
    #buf = PPOBufferContainer(obs_dim, act_dim, local_steps_per_epoch, OPEN_AI_PPO_NUMBER_OF_BUFFERS, gamma, lam)
    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old, entropy_old = data['obs'], data['act'], data['adv'], data['logp'], data['ent']

        # Policy loss
        # Omer Sella: This is where we need ac.pi to accept both observations AND actions
        #pi, logp = ac.pi(obs, act)
        _, _, logp, actorEntropy, _, actorEntropyList = ac.step(obs, act)
        ratio = torch.exp(logp - logp_old)
        
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        
        ent = actorEntropy.mean().item()
        #print(actorEntropyList[0])
        #print(actorEntropyList[0].mean())
        #print(actorEntropyList[0].mean().item())
        
        iEntropy = actorEntropyList[0].mean().item()
        jEntropy = actorEntropyList[1].mean().item()
        kEntropy = actorEntropyList[2].mean().item()
        if actionInvalidator == 'Enabled':
            noActionEntropy = actorEntropyList[3].mean().item()

        #coordinatesEntropy = actorEntropyList[3].mean().item()
       
        
        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)
        #myLogger.keyValue('kl', kl)
        #myLogger.keyValue('entropy', ent)
        #myLogger.keyValue('clippedFrac', clipfrac)
        
        #########
        #OSS: we are testing a hypothesis, that the entropy for choice of i collapses too fast. So I'm replacing ent with iEntropy and let's see what happens
        #totalLoss = policyCoefficient * loss_pi + entropyCoefficient * ent
        if actionInvalidator == 'Enabled':
            totalLoss = policyCoefficient * loss_pi + entropyCoefficient0 * iEntropy + entropyCoefficient1 * jEntropy + entropyCoefficient2 * kEntropy + entropyCoefficientNoAction * noActionEntropy
        else:
            totalLoss = policyCoefficient * loss_pi + entropyCoefficient0 * iEntropy + entropyCoefficient1 * jEntropy + entropyCoefficient2 * kEntropy
        
        return totalLoss, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret)**2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update():
        #OSS 29/11/2021 I'm moving the code from buffer to buffer container, so updtae() merges the buffers into a single one and resets the bufferContainer.
        # OSS 07/01/2022 I'm reinstating a simple buffer implementation.
        #flatBuffer = buf.flattenBuffer()
        #data = flatBuffer.get()
        data = buf.get() #OSS 07/01/2022 double check for correctness.

        #############################
        ## For debug puposes only ! 
        ## Debugging conc futures
        #next_o, r, d, _ = env.step(a[-1])
        #print("*** debugging conc futures - did I make it inside the update, after the get() ?") #YES !
        ## Did it work ?  YES !
        #############################


        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        #############################
        ## For debug puposes only ! 
        ## Debugging conc futures
        #next_o, r, d, _ = env.step(a[-1])
        #print("*** debugging conc futures - did I make it inside the update, after computing loss ?") #Yes !
        ## Did it work ? Yes !
        #############################

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            #print("*** Policy training step %d..."%i)
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = pi_info['kl']
            assert(kl == pi_info['kl'])
            #print("*** Understanding early stopping to due kl:")
            #print(kl)
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
            #############################
            ## For debug puposes only ! 
            ## Debugging conc futures
            #next_o, r, d, _ = env.step(a[-1])
            #print("*** debugging conc futures - did I make it inside the update, before loss_pi.backwards() ?")
            ## Did it work ?  
            #############################
            #print(loss_pi)
            #loss_pi.to('cpu')
            #print(loss_pi)
            loss_pi.backward()

            #OSS 07/01/2022 commented since no mpi is used.
            #mpi_avg_grads(ac.pi)    # average grads across MPI processes
            #############################
            ## For debug puposes only ! 
            ## Debugging conc futures
            #next_o, r, d, _ = env.step(a[-1])
            #print("*** debugging conc futures - did I make it inside the update, before optimizer ?")
            ## Did it work ?  
            #############################
            pi_optimizer.step()
        #############################
        ## For debug puposes only ! 
        ## Debugging conc futures
        #next_o, r, d, _ = env.step(a[-1])
        #print("*** debugging conc futures - did I make it inside the update, after train_pi_iters?")
        ## Did it work ?  No !
        #############################
        logger.store(StopIter=i)

        # Value function learning
        for i in range(train_v_iters):
            #print("*** Value function training step %d..."%i)
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            #OSS 10/01/2022 I commented mpi_avg_grads
            #mpi_avg_grads(ac.v)    # average grads across MPI processes
            vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))

    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    saveNumber = 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            
            myLogger.keyValue('Observation', o)
            a, v, logp, actorEntropy, logpList, entropyList = ac.step(torch.as_tensor(o, dtype=torch.float32))
            myLogger.keyValue('actorEntropy', actorEntropy)
            myLogger.keyValue('logP', logp)
            myLogger.keyValue('iAction', a[0])
            myLogger.keyValue('jAction', a[1])
            myLogger.keyValue('kAction', a[2])
            myLogger.keyValue('hotBitsAction', a[3])
            if actionInvalidator == 'Enabled':
                myLogger.keyValue('noAction', a[4])
            myLogger.keyValue('logpI', logpList[0].item())
            myLogger.keyValue('logpJ', logpList[1].item())
            myLogger.keyValue('logpK', logpList[2].item())
            myLogger.keyValue('iEntropy', entropyList[0].item())
            myLogger.keyValue('jEntropy', entropyList[1].item())
            myLogger.keyValue('kEntropy', entropyList[2].item())
            # Omer Sella: this is a bug ! entropyList[3] is not a 1 element tensor (coordinatesEntropy) so can't take it as an item. I'm keeping it as a placeholder.
            myLogger.keyValue('coordinatesEntropy', entropyList[2].item())
            next_o, r, d, _ = env.step(a[-1])
            myLogger.keyValue('Reward', r)
            myLogger.keyValue('epochNumber', epoch)
            myLogger.keyValue('stepNumber', t)
            myLogger.keyValue('vValue', v)
            ################################
            myLogger.dumpLogger()
            ################################
            ep_ret += r
            ep_len += 1

            # save and log
            entropyArray = np.array([entropyList[0].item(), entropyList[1].item(), entropyList[2].item()])
            #entropyArray = np.hstack((entropyArray, entropyList[3]].detach().numpy()))
            if  OPEN_AI_PPO_NUMBER_OF_BUFFERS == 1:
                # OSS: 07/01/2021 commented and reverted to regular buffer (no container) when debigging conc futures
                #buf.store([o], [a[-2]], [r], [v], [logp], [actorEntropy], [entropyArray])
                buf.store(o, a[-2], r, v, logp, actorEntropy, entropyArray)
            else:
                buf.store(o, a[-2], r, v, logp, actorEntropy, entropyArray)
            logger.store(VVals=v)
            
            # Update obs (critical!)
            o = next_o

            # timeout means we used the maximal number of episodes per epoch
            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch-1

            if terminal or epoch_ended:
                #myPlotter.step(ep_ret)
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _, _, _, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0
                if OPEN_AI_PPO_NUMBER_OF_BUFFERS == 1:
                    buf.finish_path([v])
                else:
                    buf.finish_path(v)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    #print("*** PPO acknowledges that the episode terminated")
                    #print("*** EpRet debug. ep_ret == " + str(ep_ret))
                    #logger.store(EpRet=ep_ret, EpLen=ep_len)
                    pass
                o, ep_ret, ep_len = env.reset(), 0, 0
            # OSS22 15/06/2022 cheasy and dirty fix. It's just to investigate if done flag can help the agent perform better.
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            
            

                #############################
                ## For debug puposes only ! 
                ## Debugging conc futures
                #next_o, r, d, _ = env.step(a[-1])
                ## Did it work ? Yes !
                #############################
            
            





        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1) or (epoch == 0):
            #logger.save_state({'env': env}, None)
            logger.save_state({'env': env, 'model': ac}, itr = saveNumber)
            acFileName = experimentDataDir + "/acSave" + str(saveNumber) + ".pt"
            torch.save(ac.state_dict(), acFileName)
            saveNumber = saveNumber + 1


        # Perform PPO update!
        #############################
        ## For debug puposes only ! 
        ## Debugging conc futures
        #next_o, r, d, _ = env.step(a[-1])
        #print("*** debugging conc futures - did I make it BEFORE the update ?") YES !!!
        ## Did it work ? Yes !
        #############################
        update()
        #############################
        ## For debug puposes only ! 
        ## Debugging conc futures
        #next_o, r, d, _ = env.step(a[-1])
        #print("*** debugging conc futures - did I make it after the update ?") NO !
        ## Did it work ? No !
        #############################


        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        #OSS 10/01/2022 changed because getting rid of mpi stuff
        logger.log_tabular('VVals', with_min_and_max=False)
        #logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.log_tabular('EpLen', average_only=True)
        #logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpRet', with_min_and_max=False)
        logger.dump_tabular()
    

if __name__ == '__main__':
    import argparse
    # Omer Sella: this is critical - we are setting forking to spawn, otherwise utilisation of multiple GPUs doesn't work properly
    multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default= 'gym_ldpc:ldpc-v0')
    parser.add_argument('--resetType', type=str, default= 'WORST_CODES')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=30)
    #parser.add_argument('--cpu', type=int, default=2) #Omer Sella: was 4 instead of 1
    parser.add_argument('--cpu', type=int, default=1) #Omer Sella: was 4 instead of 1
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--envCudaDevices', type=int, default=1)
    parser.add_argument('--entropyCoefficient0', type=float, default = 0.01)
    parser.add_argument('--entropyCoefficient1', type=float, default = 0.01)
    parser.add_argument('--entropyCoefficient2', type=float, default = 0.01)
    parser.add_argument('--entropyCoefficientNoAction', type=float, default = 0.01)
    parser.add_argument('--policyCoefficient', type=float, default = 1.0)
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()
    #OSS 10/01/2022 commented mpi_fork since mpi is not used.
    #mpi_fork(args.cpu)  # run parallel code with mpi

    from run_utils import setup_logger_kwargs
    import os
    experimentTime = time.time()
    PROJECT_PATH = os.environ.get('LDPC')
    experimentDataDir = PROJECT_PATH + "/temp/experiments/%i" %int(experimentTime)
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, data_dir = experimentDataDir)

    ppo(lambda x = 8200, y = 0, z = 'WORST_CODES': gym.make(args.env, seed = x, numberOfCudaDevices = y, resetType = z), #Omer Sella: Actor_Critic is now embedded and thus commented actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs, envCudaDevices = args.envCudaDevices, experimentDataDir = experimentDataDir, resetType = 'WORST_CODES', actionInvalidator = 'Disabled')
