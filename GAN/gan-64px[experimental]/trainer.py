import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt

# -------- optional LPIPS --------
try:
    import lpips
    _HAS_LPIPS = True
except Exception:
    _HAS_LPIPS = False


# ---------------- GP ----------------
def gradient_penalty(critic, real_images, fake_images, audio_embeddings, device):
    b = real_images.size(0)
    eps = torch.rand(b, 1, 1, 1, device=device, requires_grad=True)
    x_hat = eps * real_images + (1 - eps) * fake_images
    x_hat.requires_grad_(True)
    d_hat = critic(x_hat, audio_embeddings)
    grads = grad(outputs=d_hat, inputs=x_hat,
                 grad_outputs=torch.ones_like(d_hat),
                 create_graph=True, retain_graph=True, only_inputs=True)[0]
    grads = grads.view(b, -1)
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()


# ---------------- plots ----------------
def plot_losses(d_losses, g_losses, save_path="loss_plot.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(d_losses, label='Discriminator Loss', alpha=0.7)
    plt.plot(g_losses, label='Generator Loss', alpha=0.7)
    plt.xlabel('Training Steps'); plt.ylabel('Loss'); plt.title('WCGAN Training Losses')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight'); plt.close()


# ---------------- EMA ----------------
from copy import deepcopy

def make_ema(model):
    ema = deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema

@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    msd = model.state_dict()
    for k, v in ema_model.state_dict().items():
        if v.dtype.is_floating_point:
            v.copy_(v * decay + msd[k] * (1.0 - decay))


# ---------------- inference ----------------
@torch.no_grad()
def run_inference(generator, device, noise_dim, audio_embed_dim, epoch, output_folder, num_samples=16):
    generator.eval()
    noise = torch.randn(num_samples, noise_dim, device=device)
    audio_embeds = torch.randn(num_samples, audio_embed_dim, device=device)
    fake_images = generator(noise, audio_embeds)
    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, f"epoch_{epoch}_inference.png")
    save_image(fake_images, save_path, nrow=4, normalize=True)


# ---------------- train ----------------
def train(dataloader, generator, discriminator, device,
          noise_dim=128, audio_embed_dim=512, n_epochs=100,
          g_lr=2e-4, d_lr=1e-4, betas=(0.0, 0.9),
          lambda_gp=10.0, n_critic=4,
          lambda_fm=10.0, use_lpips=True, lambda_lpips=0.2,
          ema_decay=0.999, ema_start_step=1000,
          print_every=100, plot_every=200, inference_every=500,
          save_path="models", output_folder="outputs", loss_plot_path="loss_plot.png"):

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    g_opt = optim.Adam(generator.parameters(), lr=g_lr, betas=betas)
    d_opt = optim.Adam(discriminator.parameters(), lr=d_lr, betas=betas)

    d_losses, g_losses = [], []
    generator.to(device); discriminator.to(device)
    torch.randn(1, device=device)  # warm CUDA
    torch.backends.cudnn.benchmark = True

    # EMA setup
    ema_G = make_ema(generator)
    global_step = 0

    # LPIPS setup (optional)
    if use_lpips and _HAS_LPIPS:
        lpips_fn = lpips.LPIPS(net='vgg').to(device).eval()
        for p in lpips_fn.parameters(): p.requires_grad_(False)
    else:
        lpips_fn = None
        if use_lpips and not _HAS_LPIPS:
            print("LPIPS not installed; continuing without perceptual loss.")

    for epoch in range(n_epochs):
        loop = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{n_epochs}")

        for i, (real_images, audio_embeds) in loop:
            real_images = real_images.to(device, non_blocking=True)
            audio_embeds = audio_embeds.to(device, non_blocking=True)
            b = real_images.size(0)

            # ---- train D n_critic times ----
            for _ in range(n_critic):
                noise = torch.randn(b, noise_dim, device=device)
                with torch.no_grad():
                    fake_images = generator(noise, audio_embeds)

                d_real = discriminator(real_images, audio_embeds).mean()
                d_fake = discriminator(fake_images, audio_embeds).mean()
                gp = gradient_penalty(discriminator, real_images, fake_images, audio_embeds, device)

                d_loss = d_fake - d_real + lambda_gp * gp

                d_opt.zero_grad(set_to_none=True)
                d_loss.backward()
                d_opt.step()

            # ---- train G once (adv + FM + optional LPIPS) ----
            noise = torch.randn(b, noise_dim, device=device)
            gen_images = generator(noise, audio_embeds)

            # adversarial
            adv_loss = -discriminator(gen_images, audio_embeds).mean()

            # feature matching (use D penultimate pooled features)
            with torch.no_grad():
                real_feats = discriminator.features(real_images)   # [B, C]
            fake_feats = discriminator.features(gen_images)        # [B, C]
            fm_loss = torch.mean(torch.abs(real_feats - fake_feats))

            # optional perceptual (paired to same real batch)
            if lpips_fn is not None:
                perc_loss = lpips_fn(gen_images, real_images).mean()
            else:
                perc_loss = torch.tensor(0.0, device=device)

            g_loss = adv_loss + lambda_fm * fm_loss + lambda_lpips * perc_loss

            g_opt.zero_grad(set_to_none=True)
            g_loss.backward()
            g_opt.step()

            # EMA update after G step
            global_step += 1
            if global_step >= ema_start_step:
                update_ema(ema_G, generator, decay=ema_decay)

            # ---- bookkeeping ----
            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())

            if i % print_every == 0:
                loop.set_postfix(
                    D_loss=d_loss.item(),
                    G_loss=g_loss.item(),
                    D_real=d_real.item(),
                    D_fake=d_fake.item(),
                    GP=gp.item(),
                    FM=fm_loss.item(),
                    LPIPS=(perc_loss.item() if lpips_fn else 0.0)
                )

            if i % plot_every == 0 and i > 0:
                plot_losses(d_losses, g_losses, save_path=loss_plot_path)

            if i % inference_every == 0:
                # sample from EMA if available, else raw G
                run_inference(ema_G if global_step >= ema_start_step else generator,
                              device, noise_dim, audio_embed_dim, epoch, output_folder)

        # save at epoch end (both raw G and EMA)
        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'g_optimizer_state_dict': g_opt.state_dict(),
            'd_optimizer_state_dict': d_opt.state_dict(),
            'ema_G_state_dict': ema_G.state_dict(),
            'global_step': global_step
        }, os.path.join(save_path, f"ckpt_epoch{epoch}.pth"))
