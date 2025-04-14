import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, Dataset,TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import os
import math
import logging
import time

class AdaptiveBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, max_features, eps=1e-6, momentum=0.05, affine=True, track_running_stats=True):
        """
        Adaptive 1D Batch Normalization supporting dynamic feature dimensions
        
        Args:
            max_features (int): Maximum possible number of features/channels
            eps (float, optional): Value added to denominator for numerical stability. Default: 1e-5
            momentum (float, optional): Value used for running mean/var computation. Default: 0.1
            affine (bool, optional): Whether to learn affine parameters. Default: True
            track_running_stats (bool, optional): Whether to track running statistics. Default: True
        """
        super().__init__(max_features, eps, momentum, affine, track_running_stats)
        self.max_features = max_features

    def _get_shape_info(self, x):
        """
        Reshapes input tensor to 2D and preserves original shape information
        
        Args:
            x (torch.Tensor): Input tensor of any shape with last dimension as features
            
        Returns:
            tuple: (reshaped_tensor (N, num_features), original_shape)
        """
        original_shape = x.shape
        num_features = x.size(-1)
        x_reshaped = x.reshape(-1, num_features)
        return x_reshaped, original_shape

    def forward(self, x, current_dim=None):
        """
        Forward pass with adaptive feature processing
        
        Args:
            x (torch.Tensor): Input tensor of shape (..., features)
            current_dim (int, optional): Effective feature dimension for this forward pass.
                If None, uses previously set dimension. Default: None
                
        Returns:
            torch.Tensor: Normalized tensor with same shape as input
        """
        if current_dim is not None:
            active_dim = current_dim
        else:
            active_dim = x[-1].shape[-1]

        x_reshaped, original_shape = self._get_shape_info(x)
        

        valid_features = x_reshaped[:, :active_dim]
        mean = valid_features.mean(dim=0)
        var = valid_features.var(dim=0, unbiased=False)
        self.running_mean[:active_dim] = (
                (1 - self.momentum) * self.running_mean[:active_dim] 
                + self.momentum * mean.detach()
            )
        self.running_var[:active_dim] = (
                (1 - self.momentum) * self.running_var[:active_dim] 
                + self.momentum * var.detach()
            )


        if active_dim > 0:
            valid_part = x_reshaped[:, :active_dim]
            norm_part = (valid_part - mean) / torch.sqrt(var + self.eps)
            
            if self.affine:
                norm_part = norm_part * self.weight[:active_dim] + self.bias[:active_dim]

            if active_dim < self.max_features:
                remaining = x_reshaped[:, active_dim:]
                output = torch.cat([norm_part, remaining], dim=1)
            else:
                output = norm_part
        else:
            output = x_reshaped

        return output.reshape(original_shape)

    def denormalize(self,x):
        """
        De-normalizes the input tensor using the stored mean and standard deviation.
        
        Args:
            x (torch.Tensor): The input tensor to be de-normalized.
            
        Returns:
            torch.Tensor: The de-normalized tensor.
        """
        return x * self.running_var[:x.shape[-1]] + self.running_mean[:x.shape[-1]]
    
class DynamicPositionalEncoding(nn.Module):
    """
    A PyTorch module that dynamically generates positional encoding for sequences of arbitrary length.
    The formula for the embeddings is:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Attribute:
        d_model (int): The dimensionality of the input embeddings.
    Method:
        forward(x): Dynamically generates and adds positional encoding to the input embeddings.

    Example:
        >>> pos_encoder = DynamicPositionalEncoding(d_model=512)
        >>> input_embeddings = torch.randn(100, 32, 512)  
        >>> encoded_embeddings = pos_encoder(input_embeddings)
        >>> print(encoded_embeddings.shape)
        torch.Size([100, 32, 512])
    """

    def __init__(self, d_model=512,max_len=5000, device=None):
        """
        Arg:
            d_model (int): The dimensionality of the input embeddings.
        """
        super(DynamicPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.device = device
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # 不需要计算梯度

        pos = torch.arange(0, max_len, device=device).float().unsqueeze(dim=1)
        _2i = torch.arange(0, d_model, step=2, device=device).float()

        # 计算位置编码
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))


    def forward(self, x):
        """
        Dynamically generates and adds positional encoding to the input embeddings.

        Arg:
            x (torch.Tensor): The input embeddings with shape (seq_len, batch_size, d_model).
        Returns:
            torch.Tensor: The input embeddings with position encodings added. Shape (seq_len, batch_size, d_model).
        """
        #print("x.shape",x.shape)
        
        batch_size, seq_len, dim = x.size()
        #print("self.encoding.shape",self.encoding[:seq_len, :].unsqueeze(0).repeat(batch_size, 1, 1)[0,1,:])
        return self.encoding[:seq_len, :dim].unsqueeze(0).repeat(batch_size, 1, 1).to(self.device)+x


def he_init(module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

class OnlineTransformer(nn.Module):
    """
    An Online Transformer-based model for sequence reconstruction tasks.

    This model processes input sequences one data point at a time, maintaining a memory buffer
    to store previous inputs. It leverages the Transformer encoder to encode the input sequence,
    applies a Diffusion Model to enhance the encoded representation, and then uses a Transformer
    decoder to reconstruct the input sequence.

    Attributes:
        d_model (int): The dimensionality of the model's embeddings and FEATURES. In this task it is same as input_dim to reduce lineaar layers.
        nhead (int): The number of attention heads in the Transformer.
        num_encoder_layers (int): The number of layers in the Transformer encoder.
        num_decoder_layers (int): The number of layers in the Transformer decoder.
        dim_feedforward (int): The dimensionality of the feedforward network in the Transformer.
        dropout (float): The dropout probability.
        memory_size (int): The size of the memory buffer to store previous inputs.
        positional_encoding (DynamicPositionalEncoding): Module to add dynamic positional encoding to the input embeddings.
        transformer_encoder (nn.TransformerEncoder): The Transformer encoder.
        transformer_decoder (nn.TransformerDecoder): The Transformer decoder.
        output_layer (nn.Linear): Linear layer to project the Transformer output back to the input dimension.
        memory (torch.Tensor): The memory buffer to store previous input embeddings.
    """
    def __init__(self, input_dim,output_dim,d_model, nhead, num_encoder_layers, num_decoder_layers, 
                 dim_feedforward=128, dropout=0.1, memory_size=100,batch_size=32, device = "cuda",
                 ):
        super(OnlineTransformer, self).__init__()
        self.device = device
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.memory_size = memory_size
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.pre_layer = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.positional_encoding = DynamicPositionalEncoding(d_model,device=device)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, output_dim)
        )
        
        self.memory = torch.zeros((self.batch_size, self.memory_size, input_dim)).to(device) # (batch_size, memory_size, input_dim)
        self.l_output = nn.Sequential(nn.Linear(d_model, d_model),
                                      nn.ReLU(),
                                      nn.Linear(d_model, 64))
        self.m_output = nn.Sequential(nn.Linear(d_model, d_model),
                                      nn.ReLU(),
                                      nn.Linear(d_model, 64))
        self.bn_p = AdaptiveBatchNorm1d(input_dim)
        self.bn_o = AdaptiveBatchNorm1d(output_dim)
        
        self.apply(he_init)
        self.to(device)
    
    def normalize_x(self,x):
        """
        Normalizes the length parameters using the stored mean and standard deviation. Avg mean and std is updated.
        Args:
            x (torch.Tensor): The input tensor to be normalized.
        Returns:
            torch.Tensor: The normalized tensor.
        """
        try:
            a,b,c = x.shape
            reshaped_tensor = x.view(-1, c)
            normalized_tensor = self.bn_p(reshaped_tensor)
            return normalized_tensor.view(a, b, c)
        except:
            return self.bn_p(x)
    
    def normalize_noise(self, x):
        """
        Normalizes the noises using the stored mean and standard deviation. Avg mean and std is updated.
        Args:
            x (torch.Tensor): The input tensor to be normalized.
        Returns:
            torch.Tensor: The normalized tensor.
        """
        try:
            a,b,c = x.shape
            reshaped_tensor = x.view(-1, c)
            normalized_tensor = self.bn_o(reshaped_tensor)
            return normalized_tensor.view(a, b, c)
        except:
            return self.bn_o(x)
    
    
    def forward(self, x):
        """
        Processes a new input data point and generates a reconstructed output based on the current input and memory.
        Args:
            x (torch.Tensor): The new input data point with shape (batch_size, input_dim).
        Returns:
            torch.Tensor: The reconstructed output with shape (batch_size, input_dim).
        """
        res = self.pre_layer(x)
        res = self.positional_encoding(res).squeeze(0)# (batch_size, seq_length, d_model)
        res = res.permute(1, 0, 2)
        
        feature = self.transformer_encoder(res)  # (seq_length, batch_size, d_model)
        self.m_feature = feature[-1]
        decoder_output = self.transformer_decoder(feature, feature)  # (seq_length, batch_size, d_model)
        self.last_feature = decoder_output.permute(1, 0, 2)
        reconstructed = self.output_layer(decoder_output) # (seq_length, batch_size, output_dim)
        
        res = reconstructed[-1]
        return reconstructed

        
    def get_feature(self,type_index = 1):
        """
        Returns the feature representation of the input data.
        Args:
            type_index (int): The type of feature to return. 
            0 for reconstructed input, 41 dim, [cap_pos, ten_len, cap_vel, damping, friction],
            1 for reconstructed input + m feature, (41+64)dim, [cap_pos, ten_len, cap_vel, damping, friction, m_feature].
            2 for observation + last_feature, (18+64)dim, [cap_pos_noisy, ten_len_noisy, cap_vel_noisy, last_feature].
        """
        if type_index == 0:
            self.feature = self.input_estimate.detach() # (batch_size, output_dim)
        elif type_index == 1:
            m_output = self.m_output(self.m_feature.detach())
            #print("m_output.shape",m_output.shape,"self.input_estimate.shape",self.input_estimate.shape)
            self.feature = torch.cat((self.input_estimate.detach() ,m_output),dim=1)   # (batch_size, output_dim+64)
        elif type_index == 2:
            l_output = self.l_output(self.last_feature.detach())
            self.feature = torch.cat((self.input.detach() ,l_output ),dim=2) # (batch_size, output_dim+64)
        return self.feature
    
    def train_all(self,priviledge, obs_with_noise,batch_size=32, epochs=10000, trajectory = 128,
              learning_rate=1e-3, validation_split=0.2, save_dir="saved_models",save_model = True):
        """
        Trains the model using the given training data.
        Args:
            priviledge (torch.Tensor): The priviledge data with shape (batch_size, input_dim).
            obs_with_noise (torch.Tensor): The observed data with noise with shape (batch_size, input_dim).
            batch_size (int): The batch size for training.
            epochs (int): The number of training epochs.
            trajectory (int): The length of the trajectory for training.
            learning_rate (float): The learning rate for the optimizer.
            validation_split (float): The proportion of the dataset to include in the validation split.
            save_dir (str): The directory to save the model checkpoints.
            save_model (bool): Whether to save the model checkpoints.
        """
        logging.basicConfig(filename=os.path.join(save_dir, 'training.log'),
                            level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info(f"self.d_model:{self.d_model}, self.n_head: {self.nhead}, self.num_encoder_layers: {self.num_encoder_layers}, self.num_decoder_layers: {self.num_decoder_layers}, \
                     self.dim_feedforward: {self.dim_feedforward}, self.dropout: {self.dropout}, self.memory_size: {self.memory_size}")
        
        device = self.device
        self.to(device)
        priviledge_tensor = torch.tensor(priviledge, dtype=torch.float32).to(device)
        obs_with_noise_tensor = torch.tensor(obs_with_noise, dtype=torch.float32).to(device)

        # Create dataset and data loader
        dataset = TensorDataset(obs_with_noise_tensor, priviledge_tensor)
        train_size = int((1 - validation_split) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Define optimizer and loss function
        optimizer = Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,          
            T_mult=2,        
            eta_min=1e-4     
        )

        # Training loop
        for epoch in range(epochs):
            epoch_start_time = time.time()
            #self.train(priviledge_tensor , obs_with_noise_tensor)
            train_loss = 0.0
            train_start_time = time.time()
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                optimizer.zero_grad()
                output_t = self(inputs)  
                total_loss = criterion(output_t, targets)
                total_loss.backward(retain_graph=True)
                optimizer.step()
                scheduler.step(epoch + batch_idx / len(train_loader))
                train_loss += total_loss.item()

            train_loss /= len(train_loader.dataset)
            train_end_time = time.time()

            # Validation
            self.eval()
            val_loss = 0.0
            val_start_time = time.time()
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    output = self(inputs)
                    loss = criterion(output, targets)
                    val_loss += loss

            val_loss /= len(val_loader.dataset)
            val_end_time = time.time()  # 记录验证结束时间
            epoch_end_time = time.time()
            logging.info(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Train Time: {train_end_time-train_start_time:.2f}s, "
                         f"Val Loss: {val_loss:.6f}, Val Time: {val_end_time-val_start_time:.2f}s, Total Time: {epoch_end_time-epoch_start_time:.2f}s")
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Train Time: {train_end_time-train_start_time:.2f}s, "
                  f"Val Loss: {val_loss:.6f}, Val Time: {val_end_time-val_start_time:.2f}s, Total Time: {epoch_end_time-epoch_start_time:.2f}s")
            
            #save model every 100 steps
            if (epoch+1) % 5 == 0 and save_model is True:
                checkpoint_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss
                }, checkpoint_path)
                print(f"Model checkpoint saved to {checkpoint_path}")
                logging.info(f"Model checkpoint saved to {checkpoint_path}")
    
    def eval_latest(self, priviledge, obs_with_noise, batch_size=32, save_dir="saved_models",trajectory=128):
        model_files = [f for f in os.listdir(save_dir) if f.startswith("model_epoch_") and f.endswith(".pth")]
        
        model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        best_model_path = os.path.join(save_dir, model_files[-1])
        
        # Load the model
        checkpoint = torch.load(best_model_path, map_location=torch.device('cpu'))
        self.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        logging.info(f"Loaded model from epoch {epoch}")
        print(f"Loaded model from epoch {epoch}")
        device = self.device
        self.to(device)
        priviledge_tensor = torch.tensor(priviledge, dtype=torch.float32).to(device)
        obs_with_noise_tensor = torch.tensor(obs_with_noise, dtype=torch.float32).to(device)

        # Create dataset and data loader
        dataset = TensorDataset(obs_with_noise_tensor, priviledge_tensor)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.MSELoss()
        torch.set_printoptions(linewidth=1000)
        with torch.no_grad():
            for inputs, targets in train_loader:
                    inputs = inputs.permute(1, 0, 2)  # 调整形状为 (trajectory, batch_size, 49)
                    targets = targets.permute(1, 0, 2)
                    total_loss = 0.0
                    for t in range(trajectory):
                        input_t = inputs[t]  
                        target_t = targets[t]  
                        if input_t.shape[0]<self.memory.shape[0]:
                            input_t = torch.nn.functional.pad(input_t, 
                                        (0, 0, 0, self.memory.shape[1]-input_t.shape[0]), mode='constant', value=0)
                            target_t = torch.nn.functional.pad(target_t, 
                                        (0, 0, 0, self.memory.shape[1]-target_t.shape[0]), mode='constant', value=0)
                        #print("input_t.shape",input_t.shape)
                        if input_t.shape[1]<32:
                            break
                        output_t = self(input_t)
                        print(input_t[0],"\n",output_t[0],"\n",target_t[0])
                        time.sleep(1)
                        loss_t = criterion(output_t, target_t)
                        total_loss += loss_t
                    self.clear_memory()
                    train_loss += total_loss.item()

            train_loss /= len(train_loader.dataset)
    
    def update(self, noised_input, previledge,epoch,learning_rate=1e-3,optimizer = None,creiterion = None,scheduler = None):
        """
        Updates the memory buffer with the current input TORCH data.
        noised_input: The current input data with noise, (batch_size, input_dim).
        previledge: The real parameters with shape (batch_size, input_dim).
        """
        if optimizer is None:
            optimizer = Adam(self.parameters(), lr=learning_rate)
        if creiterion is None:
            creiterion = nn.MSELoss()
        if scheduler is None:
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=10,          
                T_mult=2,        
                eta_min=1e-4     
            )
        noise_modified_previledge = torch.cat((noised_input-previledge[...,:self.input_dim]
                                        ,previledge[...,self.input_dim:]),dim=-1)
        self.memory = torch.cat((self.memory[:,1:], noised_input.unsqueeze(1).to(self.device)), dim=1) # (batch_size, memory_size, d_model)
        input = self.normalize_x(self.memory)
        label = self.normalize_noise(noise_modified_previledge.to(self.device))
        
        res = self(input)
        reconstructed = res[-1,...]
        res_denorm = self.bn_o.denormalize(reconstructed)
        self.input_estimate = torch.cat((res_denorm[...,:self.input_dim]+ self.bn_p.denormalize(input[:,-1,:])
                                        ,res_denorm[...,self.input_dim:]),dim=-1) # (batch_size, input_dim)
        #print(reconstructed.shape,label.shape)
        loss = creiterion(reconstructed, label)
        loss.backward()
        optimizer.step()
        scheduler.step(epoch)
        return reconstructed, loss.item()
    
    def clear_memory(self):
        """
        Clears the memory buffer by setting it to zeros.
        """
        self.memory = torch.zeros((self.memory_size, self.batch_size, self.d_model)).to(self.device)
        
    def save_model(self, model_name):
        """
        Saves the model state to a file.
        Args:
            model_name (str): The name of the model file.
        """
        checkpoint_path = os.path.join("saved_models", f"{model_name}.pth")
        torch.save({
            'model_state_dict': self.state_dict(),
            'param': self.param
        }, checkpoint_path)
        print(f"Model saved to {checkpoint_path}")
        logging.info(f"Model saved to {checkpoint_path}")
        return checkpoint_path
    
    def load_model(self, model_name):
        """
        Loads the model state from a file.
        Args:
            model_name (str): The name of the model file.
        """
        checkpoint_path = os.path.join("saved_models", f"{model_name}.pth")
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.load_state_dict(checkpoint['model_state_dict'])
        self.param = checkpoint['param']
        print(f"Model loaded from {checkpoint_path}")
        logging.info(f"Model loaded from {checkpoint_path}")
        return checkpoint_path
    
    



def add_segmented_noise_to_tensor(input_array, segment_lengths, noise_stds):
    import numpy as np
    """
    Add different Gaussian noise to different segments of the input array.

    Parameters:
        input_array: The input array with shape (batch_size, input_size).
        segment_lengths: A list of integers specifying the lengths of each segment.
                         The sum of segment_lengths must equal input_size.
        noise_stds: The standard deviation of the noise for each segment.
    Returns:
        The array with added noise in different segments and noise array.
    Examples:
        >>> batch_size = 4
        >>> input_size = 10
        >>> input_tensor = np.random.randn(batch_size, input_size)
        >>> segment_lengths = [3, 4, 3]  # The sum of these lengths must equal input_size (10 in this case)
        >>> noise_stds = [0.1, 0.2, 0.3]  # Different noise standard deviations for each segment
        >>> noisy_tensor = add_segmented_noise_to_tensor(input_tensor, segment_lengths, noise_stds)
    """

    if not isinstance(input_array, np.ndarray):
        raise ValueError("Input must be a numpy array")
    if sum(segment_lengths) != input_array.shape[1]:
        raise ValueError("The sum of segment_lengths must equal input_size")
    
    start_idx = 0
    noisy_segments = []
    noise_segments = []
    for length, noise_std in zip(segment_lengths, noise_stds):
        end_idx = start_idx + length
        segment = input_array[:, start_idx:end_idx]
        noise = np.random.randn(*segment.shape) * noise_std
        noise_segments.append(noise)
        noisy_segment = segment + noise
        noisy_segments.append(noisy_segment)
        start_idx = end_idx
    noise_array = np.concatenate(noise_segments, axis=1)
    noisy_array = np.concatenate(noisy_segments, axis=1)
    
    return noisy_array, noise_array



if __name__ == '__main__':
    model = OnlineTransformer(input_dim=45,output_dim=12, d_model=512, nhead=8, 
                                         num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=2048)

    x = torch.randn(32, 6)  # (batch_size, input_dim)

    reconstructed = model(x)

    print("Reconstructed shape:", reconstructed.shape)  # (batch_size, input_dim)