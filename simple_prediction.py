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
    model_path = f"base_vae_beta_{str(beta).replace('.', '_')}.pth"
    model_class = BaseVAE  # Replace with the appropriate model class
    params_dict = None  # Replace with your parameters dictionary or JSON path if needed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_base = load_model(model_path, BaseVAE)
    # model_mute = load_model("path/to/mute_vae_beta_0_2.pth", MuteVAE)
    # model_mute_genre1 = load_model("path/to/mute_genre_latent_vae_beta_0_2.pth", MuteGenreLatentVAE)
    # model_mute_genre2 = load_model("path/to/mute_latent_genre_input_vae_beta_0_2.pth", MuteLatentGenreInputVAE)

    print("Model loaded successfully.")

    groove_hits = torch.tensor([1, 0, 0, 0] * 8, dtype=torch.float).view(1, 32, 1).float()
    groove_velocities = torch.rand((1, 32, 1)) * groove_hits        # values between 0 and 1 at hits == 1
    groove_offsets = (torch.rand((1, 32, 1)) - 0.5) * groove_hits   # values between -0.5 and 0.5 at hits == 1
    input_groove = torch.cat([groove_hits, groove_velocities, groove_offsets], dim=-1)

    

    # simple prediction
    hvo, latent_z = model_base.predict(input_groove)
    # hvo, latent_z = model_mute.predict(input_groove, kick_is_muted, snare_is_muted, hat_is_muted, tom_is_muted, cymbal_is_muted)
    # hvo, latent_z = model_mute_genre1.predict(input_groove, genre_ix, kick_is_muted, snare_is_muted, hat_is_muted, tom_is_muted, cymbal_is_muted)
    # hvo, latent_z = model_mute_genre2.predict(input_groove, genre_ix, kick_is_muted, snare_is_muted, hat_is_muted, tom_is_muted, cymbal_is_muted)
    print("Prediction completed successfully.")


    # # forward pass 
    # h_logits, v_logits, o_logits, mu, log_var, latent_z = model_base.forward(input_groove)
    # # h_logits, v_logits, o_logits, mu, log_var, latent_z = model_mute.forward(input_groove, kick_is_muted, snare_is_muted, hat_is_muted, tom_is_muted, cymbal_is_muted)
    # # h_logits, v_logits, o_logits, mu, log_var, latent_z = model_mute_genre1.forward(input_groove, genre_ix, kick_is_muted, snare_is_muted, hat_is_muted, tom_is_muted, cymbal_is_muted)
    # # h_logits, v_logits, o_logits, mu, log_var, latent_z = model_mute_genre2.forward(input_groove, genre_ix, kick_is_muted, snare_is_muted, hat_is_muted, tom_is_muted, cymbal_is_muted)
    
    # # activate outputs
    # hits = torch.sigmoid(h_logits)
    # velocities = torch.tanh(v_logits) + 0.5     # Make sure you use 0.5
    # offsets = torch.tanh(o_logits)