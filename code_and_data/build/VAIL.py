import argparse
import gym
import roboschool
import os
import sys
import pickle
import time
import torch.autograd as autograd
proj_direc = 'directory to VAKLIL project'
sys.path.append(os.path.abspath(os.path.join(proj_direc+'build', '..')))

from utils import *
from models.mlp_policy import Policy
from models.mlp_critic import Value
from models.mlp_policy_disc import DiscretePolicy
from models.sn_discriminator import SNDiscriminator
from models.mlp_discriminator import Discriminator
from models.ae_discriminator import AEDiscriminator
from models.aesn_discriminator import AESNDiscriminator
from models.vae_discriminator import VAEDiscriminator
from models.noise_net import NoiseNet
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from core.ppo import ppo_step
from core.common import estimate_advantages
from core.agent import Agent
from mmd import mix_rbf_mmd2,mix_imp_mmd2,z_value_mmd,exp_kernel,mix_imp_no_rbf_mmd2,mix_imp_with_bw_mmd2
# new kernel
from torch_kernel import KernelNet
import mmd_util
import pandas as pd

from torch.autograd import Variable

class ARGS():
    def __init__(self):
        self.render = False
        self.log_std = 1.0 #0.5
        self.l2_reg = 1e-3
        self.clip_epsilon = 0.2
        self.gamma = 0.99
        self.tau = 0.95
        self.prog = 'pydevconsole.py',
        self.usage = None,
        self.conflict_handler='error',
        self.add_help=True
        self.seed = 1
        self.log_interval = 1
        self.gpu_index = 0
        self.save_model_interval = 0
        self.k = 2
        self.p = 6
        # hyper-parameters
        self.WGAN = False
        self.env_name = 'RoboschoolHalfCheetah-v1' #'RoboschoolHumanoidFlagrun-v1' #'RoboschoolWalker2d-v1' #'RoboschoolHalfCheetah-v1' #'RoboschoolHumanoid-v1' #'RoboschoolAnt-v1' #'MountainCarContinuous-v0' #'CartPole-v1' #'LunarLanderContinuous-v2' #'BipedalWalker-v2'
        self.expert_traj_path = proj_direc+"assets/"+self.env_name+"/exp_traj.pkl"
        self.learning_rate = 3e-4  # 5e-4, GAIL: 2e-4, WGAIL: 1e-3, LS-GAIL: 1e-3
        self.milestones = [1500,3000] #[1000,2000] #[400,900,1500] #[300,700]
        self.lr_decay = 0.3
        self.lr_kernel_decay = 0.1
        self.discriminator_epochs = 10
        self.generator_epochs = 5
        self.ppo_batch_size = 64
        self.min_batch_size = 2000 #2048
        self.num_threads = 8
        self.max_iter_num = 4000 #2050
        self.epochs = 6
        # MMD-GAIL
        self.MMDGAN = False
        self.sigma_list = [sigma / 1.0 for sigma in [1, 2, 4, 8, 16]]
        self.lambda_AE_X = 8.0
        self.lambda_AE_Y = 8.0
        self.lambda_rg = 16.0
        # GEOM-GAIL
        self.GEOMGAN = False
        # VDB-GAIL
        self.VAIL = True
        self.alpha=1e-6
        self.beta=0
        # EB-GAIL
        self.EBGAN = False
        self.margin = max(1, self.min_batch_size / 512.)    #64
        self.r_margin = 1
        # LS-GAIL
        self.slope = 0.0    #'slope for function c proposed in generalized lsgan, when slope is 0, gls-gan is lsgan, when slope is 1 gls-gan is wgan.'
        self.lamb = 2e-2    #'the scale of the distance metric used for adaptive margins. This is actually tau in the original paper. L2: 0.05/L1: 0.001, temporary best 0.008 before applying scaling'
        self.LSGAN = False
        # save path
        self.description = 'VAIL',
        self.save_data_path = proj_direc+"build/data/"+self.env_name+"_VAIL.pkl"
args = ARGS()

dtype = torch.float64
torch.set_default_dtype(dtype)
cuda = True if torch.cuda.is_available() else False
#if args.MMDGAN:
#    cuda = False
device = torch.device('cuda', index=args.gpu_index) if cuda else torch.device('cpu')
if cuda:
    torch.cuda.set_device(args.gpu_index)
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

"""environment"""
env = gym.make(args.env_name)
state_dim = env.observation_space.shape[0]
is_disc_action = len(env.action_space.shape) == 0
action_dim = 1 if is_disc_action else env.action_space.shape[0]
running_state = ZFilter((state_dim,), clip=5)
# running_reward = ZFilter((1,), demean=False, clip=10)

"""seeding"""
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.seed(args.seed)

class ONE_SIDED(nn.Module):
    def __init__(self):
        super(ONE_SIDED, self).__init__()
        main = nn.ReLU()
        self.main = main
    def forward(self, input):
        output = self.main(-input)
        output = -output.mean()
        return output
one_sided = ONE_SIDED()
LeakyReLU = nn.LeakyReLU(args.slope)
l1dist = nn.PairwiseDistance(1)
elementwise_loss = nn.MSELoss()
discrim_criterion = nn.BCELoss()
to_device(device, discrim_criterion, elementwise_loss, l1dist, LeakyReLU,one_sided)

# functions
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.1)
        torch.nn.init.constant_(m.bias.data, 0.0)
    if classname.find('Linear') != -1:
        m.weight.data = init.kaiming_normal(m.weight.data)
        #m.weight.data.normal_(0.0, 0.1)
        #m.bias.data.fill_(0)

# load trajectory
expert_traj = pd.read_pickle(args.expert_traj_path)
expert_traj = expert_traj['state-action'].values
expert_traj = np.array(list(expert_traj))
expert_traj.dtype='float'

def create_networks():
    """define actor and critic"""
    if is_disc_action:
        policy_net = DiscretePolicy(state_dim, env.action_space.n, hidden_size=(64, 32), activation='relu')
    else:
        policy_net = Policy(state_dim, env.action_space.shape[0], log_std=args.log_std, hidden_size=(64, 32),
                            activation='relu')
    value_net = Value(state_dim, hidden_size=(32, 16), activation='relu')
    if args.WGAN:
        discrim_net = SNDiscriminator(state_dim + action_dim, hidden_size=(32, 16), activation='relu')
    elif args.EBGAN or args.MMDGAN:
        #discrim_net = AEDiscriminator(state_dim + action_dim, hidden_size=(32,), encode_size=16, activation='relu')
        discrim_net = AESNDiscriminator(state_dim + action_dim, hidden_size=(32,), encode_size=64, activation='relu',dropout=False)
        #discrim_net.apply(weights_init_normal)
    elif args.GEOMGAN:
        # new kernel
        #discrim_net = KernelNet(state_dim + action_dim,state_dim + action_dim)
        noise_dim = 64
        discrim_net = AESNDiscriminator(state_dim + action_dim, hidden_size=(32,), encode_size=noise_dim, activation='relu',dropout=False)
        kernel_net = NoiseNet(noise_dim, hidden_size=(32,), encode_size=noise_dim, activation='relu',dropout=False)
        optimizer_kernel = torch.optim.Adam(kernel_net.parameters(), lr=args.learning_rate)
        scheduler_kernel = MultiStepLR(optimizer_kernel, milestones=args.milestones, gamma=args.lr_kernel_decay)
    elif args.VAIL:
        mid_dim = 64
        discrim_net = VAEDiscriminator(state_dim + action_dim, num_outputs=1, hidden_size=(32,), encode_size=mid_dim, activation='relu',dropout=False)
    else:
        discrim_net = Discriminator(state_dim + action_dim, hidden_size=(32, 16), activation='relu')

    optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=args.learning_rate)
    optimizer_value = torch.optim.Adam(value_net.parameters(), lr=args.learning_rate)
    optimizer_discrim = torch.optim.Adam(discrim_net.parameters(), lr=args.learning_rate)

    scheduler_policy = MultiStepLR(optimizer_policy, milestones=args.milestones, gamma=args.lr_decay)
    scheduler_value = MultiStepLR(optimizer_value, milestones=args.milestones, gamma=args.lr_decay)
    scheduler_discrim = MultiStepLR(optimizer_discrim, milestones=args.milestones, gamma=args.lr_kernel_decay)

    if args.WGAN:
        class ExpertReward():
            def __init__(self):
                self.a = 0
            def expert_reward(self,state, action):
                state_action = tensor(np.hstack([state, action]), dtype=dtype)
                with torch.no_grad():
                    return -discrim_net(state_action)[0].item()
                    # return -discrim_net(state_action).sum().item()
    elif args.EBGAN:
        class ExpertReward():
            def __init__(self):
                self.a = 0
            def expert_reward(self,state, action):
                state_action = tensor(np.hstack([state, action]), dtype=dtype)
                with torch.no_grad():
                    _,recon_out = discrim_net(state_action)
                    return -elementwise_loss(recon_out, state_action).item() + args.r_margin
    elif args.MMDGAN or args.GEOMGAN:
        class ExpertReward():
            def __init__(self):
                #self.g_o_enc = ones((10, discrim_net.encode_size), device=torch.device('cpu'))  #torch.device('cpu')
                #self.e_o_enc = ones((10, discrim_net.encode_size), device=torch.device('cpu'))
                #self.XX = torch.diag(torch.mm(self.e_o_enc,self.e_o_enc.t()))
                #self.YY = torch.diag(torch.mm(self.g_o_enc,self.g_o_enc.t()))
                self.r_bias = 0
            def expert_reward(self,state, action):
                #state_action = tensor(np.hstack([state, action]), dtype=dtype)
                with torch.no_grad():
                    #z_enc,z_dec = discrim_net(state_action)
                    #k_yz_mean,k_xz_mean = z_value_mmd(self.e_o_enc, self.g_o_enc, self.XX, self.YY, z_enc, args.sigma_list)
                    # tbd
                    #pred_val = k_yz_mean - 2*k_xz_mean
                    #pred_val = k_yz_mean - k_xz_mean
                    #return -(pred_val+self.r_bias)
                    return self.r_bias
            def update_XX_YY(self):
                self.XX = torch.diag(torch.mm(self.e_o_enc, self.e_o_enc.t()))
                self.YY = torch.diag(torch.mm(self.g_o_enc, self.g_o_enc.t()))
    elif args.VAIL:
        class ExpertReward():
            def __init__(self):
                self.a = 0
            def expert_reward(self,state, action):
                state_action = tensor(np.hstack([state, action]).reshape(1,-1), dtype=dtype)
                with torch.no_grad():
                    out,_,_ = discrim_net(state_action)
                    return -math.log(out.item())
    else:
        class ExpertReward():
            def __init__(self):
                self.a = 0
            def expert_reward(self,state, action):
                state_action = tensor(np.hstack([state, action]), dtype=dtype)
                with torch.no_grad():
                    return -math.log(discrim_net(state_action)[0].item())
    learned_reward = ExpertReward()

    """create agent"""
    agent = Agent(env, policy_net, device, custom_reward=learned_reward,
                  running_state=None, render=args.render, num_threads=args.num_threads)

    def update_params(batch, i_iter):
        dataSize = min(args.min_batch_size,len(batch.state))
        states = torch.from_numpy(np.stack(batch.state)[:dataSize,]).to(dtype).to(device)
        actions = torch.from_numpy(np.stack(batch.action)[:dataSize,]).to(dtype).to(device)
        rewards = torch.from_numpy(np.stack(batch.reward)[:dataSize,]).to(dtype).to(device)
        masks = torch.from_numpy(np.stack(batch.mask)[:dataSize,]).to(dtype).to(device)
        with torch.no_grad():
            values = value_net(states)
            fixed_log_probs = policy_net.get_log_prob(states, actions)

        """estimate reward"""

        """get advantage estimation from the trajectories"""
        advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau, device)

        """update discriminator"""
        for _ in range(args.discriminator_epochs):
            #dataSize = states.size()[0]
            # expert_state_actions = torch.from_numpy(expert_traj).to(dtype).to(device)
            exp_idx = random.sample(range(expert_traj.shape[0]), dataSize)
            expert_state_actions = torch.from_numpy(expert_traj[exp_idx, :]).to(dtype).to(device)

            dis_input_real = expert_state_actions
            if len(actions.shape) == 1:
                actions.unsqueeze_(-1)
                dis_input_fake = torch.cat([states, actions], 1)
                actions.squeeze_(-1)
            else:
                dis_input_fake = torch.cat([states, actions], 1)

            if args.EBGAN or args.MMDGAN or args.GEOMGAN:
                g_o_enc,g_o = discrim_net(dis_input_fake)
                e_o_enc,e_o = discrim_net(dis_input_real)
            elif args.VAIL:
                g_o,g_mu,g_sigma = discrim_net(dis_input_fake,mean_mode=False)
                e_o,e_mu,e_sigma = discrim_net(dis_input_real,mean_mode=False)
            else:
                g_o = discrim_net(dis_input_fake)
                e_o = discrim_net(dis_input_real)

            optimizer_discrim.zero_grad()
            if args.GEOMGAN:
                optimizer_kernel.zero_grad()
            
            if args.WGAN:
                if args.LSGAN:
                    pdist = l1dist(dis_input_real, dis_input_fake).mul(args.lamb)
                    discrim_loss = LeakyReLU(e_o - g_o + pdist).mean()
                    #discrim_loss = LeakyReLU(torch.sum((e_o - g_o),1) + pdist).mean()
                else:
                    discrim_loss = torch.mean(e_o) - torch.mean(g_o)
                    #discrim_loss = torch.mean(torch.sum(e_o,1)) - torch.mean(torch.sum(g_o,1))
            elif args.EBGAN:
                e_recon = elementwise_loss(e_o, dis_input_real)
                g_recon = elementwise_loss(g_o, dis_input_fake)
                discrim_loss = e_recon
                if (args.margin - g_recon).item() > 0:
                    discrim_loss += (args.margin - g_recon)
            elif args.MMDGAN:
                mmd2_D,K = mix_rbf_mmd2(e_o_enc, g_o_enc, args.sigma_list)
                rewards = K[1]-K[2]             # -(exp - gen): -(kxy-kyy)=kyy-kxy
                rewards = -rewards.detach()     # exp - gen, maximize (gen label negative)
                errD = mmd2_D #+ args.lambda_rg * one_side_errD
                discrim_loss = -errD            # maximize errD
                # prep for generator
                advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau, device)
            elif args.GEOMGAN:
                # larger, better, but slower
                noise_num = 20000
                mmd2_D_net,K_net = mix_imp_with_bw_mmd2(e_o_enc,g_o_enc,noise_num, noise_dim, kernel_net, cuda,args.sigma_list)
                mmd2_D_rbf,K_rbf = mix_rbf_mmd2(e_o_enc, g_o_enc, args.sigma_list)
                mmd2_D = (mmd2_D_net+mmd2_D_rbf)/2
                K = [sum(x)/2 for x in zip(K_net,K_rbf)]
                rewards = K[1]-K[2]             # -(exp - gen): -(kxy-kyy)=kyy-kxy
                rewards = -rewards.detach()
                errD = mmd2_D #+ args.lambda_rg * one_side_errD
                discrim_loss = -errD            # maximize errD
                # prep for generator
                advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau, device)
            elif args.VAIL:
                recon_loss = discrim_criterion(g_o, ones((states.shape[0], 1), device=device)) + \
                               discrim_criterion(e_o, zeros((e_o.shape[0], 1), device=device))
                mus = torch.cat((e_mu, g_mu), dim=0)
                sigmas = torch.cat((e_sigma, g_sigma), dim=0)
                # 1e-8: small number for numerical stability
                kl_divergence = (0.5 * torch.sum((mus ** 2) + (sigmas ** 2)
                                      - torch.log((sigmas ** 2) + 1e-8) - 1, dim=1))
                i_c = 0.2
                bottleneck_loss = (torch.mean(kl_divergence) - i_c)
                discrim_loss = recon_loss/2 + (args.beta * bottleneck_loss)
                # update beta
                args.beta = max(0, args.beta + (args.alpha * bottleneck_loss.detach()))
            else:
                discrim_loss = discrim_criterion(g_o, ones((states.shape[0], 1), device=device)) + \
                               discrim_criterion(e_o, zeros((e_o.shape[0], 1), device=device))

            discrim_loss.backward()
            optimizer_discrim.step()
            if args.GEOMGAN:
                optimizer_kernel.step()

        """perform mini-batch PPO update"""
        optim_iter_num = int(math.ceil(states.shape[0] / args.ppo_batch_size))
        for _ in range(args.generator_epochs):
            perm = np.arange(states.shape[0])
            np.random.shuffle(perm)
            perm = LongTensor(perm).to(device)

            states, actions, returns, advantages, fixed_log_probs = \
                states[perm].clone(), actions[perm].clone(), returns[perm].clone(), advantages[perm].clone(), \
                fixed_log_probs[perm].clone()

            for i in range(optim_iter_num):
                ind = slice(i * args.ppo_batch_size, min((i + 1) * args.ppo_batch_size, states.shape[0]))
                states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = \
                    states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind]

                ppo_step(policy_net, value_net, optimizer_policy, optimizer_value, 1, states_b, actions_b, returns_b,
                         advantages_b, fixed_log_probs_b, args.clip_epsilon, args.l2_reg)

        return rewards
    if args.GEOMGAN:
        return policy_net,value_net,discrim_net,kernel_net,optimizer_policy,optimizer_value,optimizer_discrim,optimizer_kernel,agent,update_params \
            ,scheduler_policy,scheduler_value,scheduler_discrim,scheduler_kernel
    else:
        return policy_net,value_net,discrim_net,optimizer_policy,optimizer_value,optimizer_discrim,agent,update_params \
            ,scheduler_policy,scheduler_value,scheduler_discrim

def main_loop():
    time_list = list()
    reward_list = list()
    label_list = list()
    rMean_list = list()
    rStd_list = list()
    for e_iter in range(args.epochs):
        rList = list()
        stdList = list()
        if args.GEOMGAN:
            policy_net, value_net, discrim_net,kernel_net, optimizer_policy, optimizer_value, optimizer_discrim,optimizer_kernel,agent, update_params \
                , scheduler_policy, scheduler_value, scheduler_discrim,scheduler_kernel \
                = create_networks()
        else:
            policy_net, value_net, discrim_net, optimizer_policy, optimizer_value, optimizer_discrim,agent, update_params \
                , scheduler_policy, scheduler_value, scheduler_discrim \
                = create_networks()
        to_device(device, policy_net, value_net, discrim_net)
        if args.GEOMGAN:
            to_device(device,kernel_net)

        for i_iter in range(args.max_iter_num):
            scheduler_policy.step()
            scheduler_value.step()
            scheduler_discrim.step()
            if args.GEOMGAN:
                scheduler_kernel.step()
            """generate multiple trajectories that reach the minimum batch_size"""
            discrim_net.to(torch.device('cpu'))
            batch, log = agent.collect_samples(args.min_batch_size)
            discrim_net.to(device)

            t0 = time.time()
            rewards = update_params(batch, i_iter)
            #print('min_r {:.4f}\t mean_r {:.4f} \t max_r{:.4f}'.format(rewards.min().item(),rewards.mean().item(),rewards.max().item()))
            t1 = time.time()

            time_list.append(i_iter)
            reward_list.append(log['avg_reward'])
            label_list.append(args.description[0])

            if i_iter % args.log_interval == 0:
                r_mean = np.mean(log['reward_list'])
                r_std = np.std(log['reward_list'])
                r_dif = max(log['max_reward'] - r_mean, r_mean - log['min_reward'])
                print('{}\tT_sample {:.4f}\tT_update {:.4f}\tR_tot {:.2f}\tR_avg {:.2f} +- {:.2f}\t R_std {:.2f}\tEpisodes {:.2f}\tSteps {:.2f}'.format(
                    i_iter, log['sample_time'], t1 - t0, log['total_reward'], r_mean, r_dif, r_std,\
                    log['num_episodes'], log['num_steps']))
            rList.append(r_mean)
            stdList.append(r_std)
            if args.save_model_interval > 0 and (i_iter + 1) % args.save_model_interval == 0:
                to_device(torch.device('cpu'), policy_net, value_net, discrim_net)
                pickle.dump((policy_net, value_net, discrim_net),
                            open(os.path.join(assets_dir(), 'learned_models/{}_gail.p'.format(args.env_name)), 'wb'))
                to_device(device, policy_net, value_net, discrim_net)
            
            """clean up gpu memory"""
            torch.cuda.empty_cache()
        
        rLarge = np.array(rList[-10:])
        rMean = np.mean(rLarge[np.argsort(rLarge)[-7:]])
        stdSmall = np.array(stdList[-10:])
        rStd = np.mean(stdSmall[np.argsort(stdSmall)[:7]])
        rMean_list.append(rMean)
        rStd_list.append(rStd)
        print('rMean {:.2f}\trStd {:.2f}'.format(rMean,rStd))
        
        data_dic = {'time': time_list, 'reward': reward_list, 'Algorithms': label_list}
        df = pd.DataFrame(data_dic)
        df.to_pickle(args.save_data_path)
    print('Epochs rMean {:.2f}\trStd {:.2f}'.format(np.mean(rMean_list),np.mean(rStd_list)))

main_loop()


