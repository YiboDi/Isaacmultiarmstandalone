from BaseNet import StochasticActor, Q
from testNet import testpolicy, testq
from transformer import tf_policy, tf_q



def create_lstm(training_config, actor_obs_dim = 107, action_dim = 6, critic_obs_dim = 107):
    policy_net = StochasticActor(
            obs_dim=actor_obs_dim,
            action_dim=action_dim,
            action_variance_bounds=training_config['action_variance'],
            network_config=training_config['network']['actor'])
    
    # Q1 = Q(obs_dim=critic_obs_dim + 6
    #         if training_config['centralized_critic'] # default to be false
    #         else critic_obs_dim, 
    #         action_dim=action_dim,
    #         network_config=training_config['network']['critic'])
    Q1 = Q(obs_dim=critic_obs_dim, 
            action_dim=action_dim,
            network_config=training_config['network']['critic'])
    
    Q2 = Q(obs_dim=critic_obs_dim, 
            action_dim=action_dim,
            network_config=training_config['network']['critic'])
    
    network = {
        'policy':policy_net,
        'Q1' : Q1,
        'Q2' : Q2,
    }

    return network

def create_testnet(actor_obs_dim = 107, action_dim = 6, critic_obs_dim = 107):
    policy_net = testpolicy(
        # obs_dim=actor_obs_dim,
        # action_dim=action_dim,
                ).to('cuda')
    Q1 = testq(
        # obs_dim=critic_obs_dim,
        #        action_dim=action_dim
                ).to('cuda')
    Q2 = testq(
        # obs_dim=critic_obs_dim,
        #        action_dim=action_dim
                ).to('cuda')
    network = {
        'policy':policy_net,
        'Q1' : Q1,
        'Q2' : Q2,
    }

    return network


def create_transformer(training_config, actor_obs_dim = 107, action_dim = 6, critic_obs_dim = 107):
    policy_net = tf_policy( obs_dim=actor_obs_dim,
                            action_dim=action_dim,
                            action_variance_bounds=training_config['action_variance'],
                            network_config=training_config['network']['actor'])

    Q1 = tf_q(   obs_dim=critic_obs_dim,
                action_dim=action_dim,
                network_config=training_config['network']['critic'])

    Q2 = tf_q(   obs_dim=critic_obs_dim,
                action_dim=action_dim,
                network_config=training_config['network']['critic'])

    network = {
        'policy':policy_net,
        'Q1' : Q1,
        'Q2' : Q2,
    }

    return network