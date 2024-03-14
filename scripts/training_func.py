import torch
from mpol import precomposed, losses
import numpy as np
import matplotlib.pyplot as plt

def seed_from_dirty_image(learning_rate_dim, n_iter_dim, coords, img, dset, plot_loss_per_iteration=True, plot_final_seed=True):
    """
    use the dirty image as the initial model image in BaseCube
    create a loss function corresponding to the mean squared error (MSE) between the RML model image pixel fluxes
    and the dirty image pixel fluxes and then optimize this RML model. It calculates the loss based off of the
    image-plane distance between the dirty image and the state of the ImageCube in order to make the state of the
    ImageCube closer to the dirty image.
    """
    print(f"Starting the optimisation loop with {n_iter_dim} iterations to optimise the initial model image (BaseCube) based on the dirty image...")
    dirty_image = torch.tensor(img.copy())  # converts the dirty image into a pytorch tensor
    rml_dim = precomposed.SimpleNet(coords=coords, nchan=dset.nchan) # initialise SimpleNet
    optimizer_dim = torch.optim.SGD(rml_dim.parameters(), lr=learning_rate_dim) # instantiate the SGD optimizer

    loss_tracker_dim = []
    for i_dim in range(n_iter_dim):
        optimizer_dim.zero_grad() # zero out any gradients attached to the tensor components so that they aren’t counted twice
        rml_dim() # calculate the model visibilities from the current model image
        sky_cube_dim = rml_dim.icube.sky_cube # get the model image from the BaseCube object
        lossfunc_dim = torch.nn.MSELoss(reduction="sum")  # the MSELoss calculates mean squared error (squared L2 norm)
        loss_dim = (lossfunc_dim(sky_cube_dim, dirty_image)) ** 0.5 # square root of the MSE is our loss value
        loss_tracker_dim.append(loss_dim.item()) # append the loss value to the loss tracker list
        loss_dim.backward() # calculate the gradients of the loss with respect to the model parameters
        optimizer_dim.step() # subtract the gradient image to the base image in order to advance base parameters in the direction of the minimum loss value

    # plot the loss per iteration
    if plot_loss_per_iteration:
        fig, ax = plt.subplots(nrows=1)
        ax.plot(loss_tracker_dim)
        ax.set_xlabel("iteration")
        ax.set_ylabel("loss")
        ax.set_title("loss per iteration - L2 norm (MSE) between dirty image and BaseCube model image")
        plt.savefig('RML_loop_outputs/loss_per_iteration_dim.pdf', format='pdf', bbox_inches='tight')
        print(f'Loss per iteration (for optimising the BaseCube image to the dirty image) plot saved to: RML_loop_outputs/loss_per_iteration_dim.pdf')
        plt.close()

    # plot the final model image (BaseCube) after the last iteration
    if plot_final_seed:
        img_dim = np.squeeze(rml_dim.icube.sky_cube.detach().numpy())
        fig, ax = plt.subplots(nrows=1)
        im_dim = ax.imshow(img_dim, origin="lower", interpolation="none", extent=rml_dim.icube.coords.img_ext)
        plt.colorbar(im_dim)
        plt.savefig('RML_loop_outputs/optimised_input_model_image_based_on_dirty_image.pdf', format='pdf', bbox_inches='tight')
        print(f'Loss per iteration (for optimising the initial BaseCube to the dirty image) plot saved to: RML_loop_outputs/optimised_input_model_image_based_on_dirty_image.pdf')
        plt.close()
        print(f'The optimised initial image based on the dirty image contains {np.sum(img_dim < 0)} negative pixels.')

        # save the optimised initial model image (BaseCube) to a .pt file
    torch.save(rml_dim.state_dict(), "RML_loop_outputs/dirty_image_model.pt")
    print('Optimised initial model image (BaseCube) based on the dirty image saved to: RML_loop_outputs/dirty_image_model.pt\n')
    return rml_dim


def train(hyperparams_config, dset, rml, optimizer, writer=None):
    # initiate a list to store the loss values at each iteration
    loss_tracker = []
    # NOTE: there was no rml.train here, but in the tutorial, it's written model.train()
    rml.train() # set the model to training mode
    for i in range(hyperparams_config["epochs"]):
        # NOTE: it was rml.zero_grad() before, but in the tutorial it is written as optimizer.zero_grad()
        optimizer.zero_grad()
        # rml.zero_grad()

        # STEP 1: calculate the model visibilities from the current model image
        vis = rml() # calculate model visibilities
        sky_cube = rml.icube.sky_cube # get the current model 'sky' image

        # STEP 2: calculate the loss between the model visibilities and the data visibilities
        # loss = losses.nll_gridded(vis, dset) # loss function without regularizers, using only the NLL
        loss = (
            losses.nll_gridded(vis, dset)
            + hyperparams_config["lambda_sparsity"] * losses.sparsity(sky_cube)
            + hyperparams_config["lambda_TV"] * losses.TV_image(sky_cube)
            + hyperparams_config["entropy"] * losses.entropy(sky_cube, hyperparams_config["prior_intensity"])
            + hyperparams_config["TSV"] * losses.TSV(sky_cube)
        ) # loss function with regularizers

        if writer is not None:
            writer.add_scalar("loss", loss.item(), i)

        loss_tracker.append(loss.item()) # append the loss value to the loss tracker list

        # STEP 3: calculate the gradients of the loss with respect to the model parameters
        loss.backward()

        # STEP 4: subtract the gradient image to the base image in order to advance base parameters in the direction of the minimum loss value
        optimizer.step()
    return loss_tracker