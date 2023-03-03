import torch as t
import torch.nn as nn
from .base_model import PyMADModel
from ..utils.utils import de_unfold


class _LSTMVAE(nn.Module):
    def __init__(self, n_features, hidden_size, latent_size, num_layers):
        super(_LSTMVAE, self).__init__()

        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.latent_size = latent_size

        self.lstm_encoder = nn.LSTM(input_size=self.n_features, hidden_size=self.hidden_size, num_layers=self.num_layers)
        self.lstm_decoder = nn.LSTM(input_size=self.latent_size, hidden_size=self.hidden_size, num_layers=self.num_layers)

        self.hidden_mu = nn.Linear(in_features=hidden_size, out_features=latent_size)
        self.hidden_logvar = nn.Linear(in_features=hidden_size, out_features=latent_size) 
         
        self.x_mu = nn.Linear(in_features=hidden_size, out_features=n_features)
        self.x_sigma = nn.Linear(in_features=hidden_size, out_features=n_features)
        self.softplus = nn.Softplus()

    def forward(self, input):
        """
        input:
            input sequence (batch_size, input_size, len_sequence)
        """
        batch_size, input_size, len_sequence = input.shape
        assert input_size==self.n_features, 'Input size of X should be equal to self.input_size'

        input = input.permute(2,0,1)

        # ENCODER
        hidden, _ = self.lstm_encoder(input)
        z_mu = self.hidden_mu(hidden)
        z_sigma = self.hidden_logvar(hidden)
        z_sigma = t.exp(0.5 * z_sigma)

        # REPARAMETRIZATION
        z = t.randn([len_sequence, batch_size, self.latent_size]).to(z_mu.device)
        z = z * z_sigma + z_mu

        # DECODER
        hidden, _ = self.lstm_decoder(z)
        x_mu = self.x_mu(hidden)
        x_sigma = self.softplus(self.x_sigma(hidden))

        # RESHAPE
        # From (len_sequence, batch_size, n_features) -> (batch_size, n_features, len_sequence)
        x_mu = x_mu.permute(1,2,0)
        x_sigma = x_sigma.permute(1,2,0)

        return x_mu, x_sigma, z_mu, z_sigma, z

def kl_multivariate_gaussian(mu, sigma):
    
    D = mu.shape[-1]

    # KL per timestamp
    # assumes that the prior has prior_mu:=0
    # (T, batch, D) KL collapses D
    trace = t.sum(sigma**2, dim=2)
    mu = t.sum(mu**2, dim=2)
    log_sigma = t.sum(t.log(sigma**2 + 1e-5), dim=2) # torch.log(sigma**2) is the determinant for diagonal + log properties
    kl = 0.5*(trace + mu - log_sigma - D)

    # Mean KL
    kl = t.mean(kl)

    if t.isnan(kl):
        print('kl', kl)
        print('trace', trace)
        print('mu', mu)
        print('log_sigma', log_sigma)
        assert 1<0

    return kl

class LSTMVAE(PyMADModel):
    def __init__(self, n_features, window_size, window_step, hidden_size,
                 latent_size, num_layers, noise_std, random_seed, device=None):
        super(LSTMVAE, self).__init__()

        # LSTM parameters
        self.n_features = n_features
        self.window_size = window_size
        self.window_step = window_step
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers

        # Training
        self.noise_std = noise_std

        # Generator
        t.manual_seed(random_seed)

        self.model = _LSTMVAE(n_features=self.n_features,
                              hidden_size=self.hidden_size,
                              latent_size=self.latent_size,
                              num_layers=self.num_layers).to(self.device)

        self.training_type = 'sgd'

        # For anomaly score
        self.gaussianNLL = nn.GaussianNLLLoss(reduction='none')

        # For training
        self.reconstruction_loss = nn.GaussianNLLLoss()

    def forward(self, input):

        Y = input['Y'].to(self.device)
        mask = input['mask'].to(self.device)

        # Hide with mask
        Y = Y * mask

        # Add gaussian noise (TODO: move to training_step)
        if self.model.train():
            Y_noisy = Y + self.noise_std*(t.randn(Y.shape).to(self.device))
        else:
            Y_noisy = Y

        # Forward
        Y_mu, Y_sigma, Z_mu, Z_sigma, Z = self.model(input=Y_noisy)

        return Y, Y_mu, mask, Y_sigma, Z_mu, Z_sigma, Z

    def training_step(self, input):
        
        # Forward
        Y, Y_mu, mask, Y_sigma, Z_mu, Z_sigma, Z = self.forward(input=input)

        # Loss
        Y_mu = Y_mu.flatten()
        Y_sigma = Y_sigma.flatten()
        Y = Y.flatten()
        mask = mask.flatten()

        # Remove masked values
        Y_mu = Y_mu[mask>0]
        Y_sigma = Y_sigma[mask>0]
        Y = Y[mask>0]
        
        kl_loss = kl_multivariate_gaussian(mu=Z_mu, sigma=Z_sigma)
        recon_loss = self.reconstruction_loss(input=Y_mu, target=Y, var=Y_sigma**2)
        loss = kl_loss + recon_loss

        return loss

    def eval_step(self, input):
        loss = self.training_step(input)
        return loss

    def window_anomaly_score(self, input, return_detail: bool=False):

        # Forward
        Y, Y_mu, _, Y_sigma, _, _, _ = self.forward(input=input)
        
        # (batch_size, n_features, len_sequence)
        anomaly_score = self.gaussianNLL(input=Y_mu, target=Y, var=Y_sigma**2)

        # Anomaly Score
        if return_detail:
            return anomaly_score
        else:
            return t.mean(anomaly_score, dim=1)

    def final_anomaly_score(self, input, return_detail: bool=False):
        
        # Average anomaly score for each feature per timestamp
        anomaly_scores = de_unfold(windows=input, window_step=self.window_step)

        if return_detail:
            return anomaly_scores
        else:
            anomaly_scores = t.mean(anomaly_scores, dim=0)
            return anomaly_scores

