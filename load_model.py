import torch
from model import BaseVAE, MuteVAE, MuteGenreLatentVAE, MuteLatentGenreInputVAE
 
def load_model(model_path, model_class, params_dict=None, is_evaluating=True, device=None):
    try:
        if device is not None:
            loaded_dict = torch.load(model_path, map_location=device)
        else:
            loaded_dict = torch.load(model_path)
    except:
        loaded_dict = torch.load(model_path, map_location=torch.device('cpu'))

    if params_dict is None:
        if 'params' in loaded_dict:
            params_dict = loaded_dict['params']
        else:
            raise Exception(f"Could not instantiate model as params_dict is not found. "
                            f"Please provide a params_dict either as a json path or as a dictionary")

    if isinstance(params_dict, str):
        import json
        with open(params_dict, 'r') as f:
            params_dict = json.load(f)

    model = model_class(params_dict)
    model.load_state_dict(loaded_dict["model_state_dict"])
    if is_evaluating:
        model.eval()

    return model

if __name__ == "__main__":
    beta = 0.2  # Replace with the appropriate beta value for your model
    model_path = f"base_vae_beta_{beta}.pth"
    model_class = BaseVAE  # Replace with the appropriate model class
    params_dict = None  # Replace with your parameters dictionary or JSON path if needed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_base = load_model(model_path, BaseVAE)
    # model_mute = load_model("path/to/mute_vae_beta_0_2.pth", MuteVAE)
    # model_mute_genre1 = load_model("path/to/mute_genre_latent_vae_beta_0_2.pth", MuteGenreLatentVAE)
    # model_mute_genre2 = load_model("path/to/mute_latent_genre_input_vae_beta_0_2.pth", MuteLatentGenreInputVAE)

    print("Model loaded successfully.")