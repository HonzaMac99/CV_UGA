
import numpy as np
import torch
from torch import nn

from skimage import measure
import trimesh

# Training
Max_epoch=10
Batch_size=100

# MLP class
class MLP(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Linear(3, 60),
        nn.Tanh(),
        nn.Linear(60, 120),
        nn.ReLU(),
        nn.Linear(120, 60),
        nn.ReLU(),
        nn.Linear(60, 30),
        nn.ReLU(), 
        nn.Linear(30, 1),
        #nn.Sigmoid()
    )

  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)

  
# MLP Training
def nif_train(data_in, data_out, batch_size):

    #GPU or not GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ",device)
 
    # Initialize the MLP
    mlp = MLP()
    mlp=mlp.float()
    mlp.to(device)

    # Normalize cost between 0 and 1 in the grid
    n_one = (data_out == 1).sum()
    p_weight = (data_out.size()[0] - n_one)/n_one  # loss for positives will be multiplied by this factor in the loss function
    print("Pos. Weight: ", p_weight)

    # Define the loss function and optimizer
    #loss_function = nn.CrossEntropyLoss()
    loss_function = nn.BCEWithLogitsLoss(pos_weight=p_weight)  #sigmoid included in this loss function
    optimizer = torch.optim.SGD(mlp.parameters(), lr=1e-2)

    # Run the training loop
    for epoch in range(0, Max_epoch): 
    
        print(f'Starting epoch {epoch+1}/{Max_epoch}')

        # Creating batch indices
        permutation = torch.randperm(data_in.size()[0])
        
        # Set current loss value
        current_loss = 0.0
        accuracy = 0
    
        # Iterate over batches
        for i in range(0,data_in.size()[0], batch_size):
       
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = data_in[indices], data_out[indices]
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Zero the gradient
            optimizer.zero_grad()
      
            # Perform forward pass
            outputs = mlp(batch_x.float())
            
            # Compute loss
            loss     = loss_function(outputs, batch_y.float())
          
            # Perform backward pass
            loss.backward()
      
            # Perform optimization
            optimizer.step()
      
            # Print current loss so far
            current_loss += loss.item()
            if (i/batch_size) % 500 == 499:
                print('Loss after mini-batch %5d: %.3f' %((i/batch_size) + 1, current_loss / (i/batch_size) + 1))
       
        outputs = torch.sigmoid(mlp(data_in.float()))
        acc = binary_acc(outputs,data_out)
        print("Binary accuracy: ",acc)     
 
    # Training is complete.
    print('MLP trained.')
    return(mlp)


# IOU evaluation between binary grids
def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(y_pred)    
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    accuracy = correct_results_sum/y_test.shape[0]
    accuracy = torch.round(accuracy * 100)    
    return accuracy


####### MAIN #########  
if __name__ == "__main__":
    
    # Load occupancy
    occupancy = np.load("output/occupancy.npy")
    resolution = occupancy.shape[0]

    # Build 3D grids with the proper resolution
    # 3D Grids are of size: resolution x resolution x resolution/2
    step = 2/resolution

    # The grids X, Y, Z allow to query the world position inside the volumetric representation:
    # given a volume location i,j,k, then X[i,j,k] gives it's x-world-coordinate
    X, Y, Z = np.mgrid[-1:1:step, -1:1:step, -0.5:0.5:step]

    # Format data for PyTorch
    data_in=np.stack((X,Y,Z),axis=-1)
    data_in=np.reshape(data_in,(int(resolution * resolution * resolution/2),3))
    data_out=np.reshape(occupancy,(int(resolution * resolution * resolution/2),1))
   
    # Pytorch format
    data_in=torch.from_numpy(data_in)
    data_out = torch.from_numpy(data_out)

    # Train mlp
    mlp=nif_train(data_in,data_out,Batch_size)  # data_out.size()[0])
    
    # Apply the network to the input coordinates
    outputs = mlp(data_in.float())
    occ=outputs.detach().cpu().numpy() # from torch format to numpy
    newocc= np.reshape(occ,(resolution,resolution,int(resolution/2))) # Go back to 3D grid
    newocc=np.around(newocc)    
    verts, faces, normals, values = measure.marching_cubes(newocc, 0.25) # Marching cubes    
    surf_mesh = trimesh.Trimesh(verts, faces, validate=True) # Export in a standard file format 
    surf_mesh.export('output/alimplicit.off')
