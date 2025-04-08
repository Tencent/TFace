import os
import sys
from timeit import default_timer as timer

import hydra
import torch
import torchmetrics as tm
from diffusion.ddpm import DenoisingDiffusionProbabilisticModel
from hydra.utils import instantiate
from models.autoencoder.vqgan import VQDecoderInterface, VQEncoderInterface
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.lite import LightningLite
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from utils.helpers import (count_model_parameters, denormalize_to_zero_to_one,
                           ensure_path_join, normalize_to_neg_one_to_one,
                           print_status)

sys.path.insert(0, "uiface/")

os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"


class DiffusionTrainerLite(LightningLite):

    @staticmethod
    @rank_zero_only
    def save_checkpoint(
        ema_model,
        diffusion_model,
        optimizer,
        global_step,
        epoch,
        steps_of_checkpoints=None,
        lr_scheduler=None,
    ):

        optimization_ckpt = {
            "global_step": global_step,
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
        }

        if lr_scheduler:
            optimization_ckpt["lr_scheduler"] = lr_scheduler.state_dict()

        torch.save(
            ema_model.averaged_model.state_dict(),
            ensure_path_join(
                os.getcwd(), "checkpoints", f"ema_averaged_model_{global_step}.ckpt"
            ),
        )
        torch.save(
            diffusion_model.state_dict(),
            ensure_path_join(os.getcwd(), "checkpoints", f"model_{global_step}.ckpt"),
        )
        torch.save(
            optimization_ckpt,
            ensure_path_join(os.getcwd(), "checkpoints", f"optimization.ckpt"),
        )

        print(f"Successfully saved checkpoint (global_step: {global_step})")

    @staticmethod
    def restore_checkpoint(model, optimizer, path, lr_scheduler=None):
        model_ckpt = torch.load(
            os.path.join(path, "checkpoints", "model.ckpt"), map_location="cpu"
        )
        optimization_ckpt = torch.load(
            os.path.join(path, "checkpoints", "optimization.ckpt"), map_location="cpu"
        )

        global_step = optimization_ckpt["global_step"]
        epoch = optimization_ckpt["epoch"]

        model.load_state_dict(model_ckpt)
        optimizer.load_state_dict(optimization_ckpt["optimizer"])

        if "lr_scheduler" in optimization_ckpt:
            lr_scheduler.load_state_dict(optimization_ckpt["lr_scheduler"])

        print(
            f"Successfully restored checkpoint (global_step: {global_step}) from: {path}"
        )

        return global_step, epoch

    @rank_zero_only
    def create_and_save_sample_grid(
        self, model, size, x=None, context=None, latent_decoder=None, save_path=""
    ):
        N_PER_ROW = 2
        PAD_VALUE = 1.0
        PADDING = 4
        N_PER_BLOCK = 4
        while not isinstance(model, DenoisingDiffusionProbabilisticModel):
            model = model.module
        model.eval()
        with torch.no_grad():

            # 1. empty identity embs
            samples_uncond = model.sample_ddim(N_PER_BLOCK, size).cpu()
            if latent_decoder is not None:
                samples_uncond = latent_decoder(samples_uncond)
            samples_uncond = denormalize_to_zero_to_one(samples_uncond)

            # grid block with only the unconditional samples
            grid = make_grid(
                samples_uncond, nrow=N_PER_ROW, pad_value=PAD_VALUE, padding=PADDING
            )
            divider = torch.full(
                (grid.shape[0], PADDING, grid.shape[2]), PAD_VALUE
            ).cpu()

            if context is not None:
                context = context.float()
                # 2. random gaussian
                syn_context = torch.nn.functional.normalize(
                    torch.randn_like(context)
                ).cuda()
                samples_syn_cond = model.sample_ddim(
                    N_PER_BLOCK, size, context=syn_context[:N_PER_BLOCK]
                ).cpu()
                if latent_decoder is not None:
                    samples_syn_cond = latent_decoder(samples_syn_cond)
                samples_syn_cond = denormalize_to_zero_to_one(samples_syn_cond)
                block = make_grid(
                    samples_syn_cond,
                    nrow=N_PER_ROW,
                    pad_value=PAD_VALUE,
                    padding=PADDING,
                )
                grid = torch.cat([grid, divider, block], dim=1)

                # and fixed synthetic context but non-fixed noise conditional
                # generation
                context_fixed = syn_context[0].repeat(N_PER_ROW, 1).cuda()
                samples_fixed_syn_context = model.sample_ddim(
                    N_PER_ROW, size, context=context_fixed
                ).cpu()
                if latent_decoder is not None:
                    samples_fixed_syn_context = latent_decoder(
                        samples_fixed_syn_context
                    )
                samples_fixed_context = denormalize_to_zero_to_one(
                    samples_fixed_syn_context
                )
                block = make_grid(
                    samples_fixed_context,
                    nrow=N_PER_ROW,
                    pad_value=PAD_VALUE,
                    padding=PADDING,
                )
                grid = torch.cat([grid, divider, block], dim=1)

                samples_cond = model.sample_ddim(
                    N_PER_BLOCK, size, context=context[:N_PER_BLOCK]
                ).cpu()
                if latent_decoder is not None:
                    samples_cond = latent_decoder(samples_cond)
                samples_cond = denormalize_to_zero_to_one(samples_cond)
                block = make_grid(
                    samples_cond, nrow=N_PER_ROW, pad_value=PAD_VALUE, padding=PADDING
                )
                grid = torch.cat([grid, divider, block], dim=1)

                # and fixed context but non-fixed noise conditional generation
                context_fixed = context[0].repeat(N_PER_ROW, 1).cuda()
                samples_fixed_context = model.sample_ddim(
                    N_PER_ROW, size, context=context_fixed
                ).cpu()
                if latent_decoder is not None:
                    samples_fixed_context = latent_decoder(samples_fixed_context)
                samples_fixed_context = denormalize_to_zero_to_one(
                    samples_fixed_context
                )
                block = make_grid(
                    samples_fixed_context,
                    nrow=N_PER_ROW,
                    pad_value=PAD_VALUE,
                    padding=PADDING,
                )
                grid = torch.cat([grid, divider, block], dim=1)

            # add additional block with real samples
            if x is not None:
                grid = torch.cat(
                    [
                        grid,
                        divider,
                        make_grid(
                            x[:N_PER_BLOCK].cpu(),
                            nrow=N_PER_ROW,
                            pad_value=PAD_VALUE,
                            padding=PADDING,
                        ),
                    ],
                    dim=1,
                )

            save_image(grid, save_path)

        model.train()

    def run(self, cfg):
        # seed for reproducibility
        self.seed_everything(cfg.constants.seed)

        # create diffusion model from config
        diffusion_model = instantiate(cfg.diffusion)
        # count number of parameters
        trainable_params, _, total_params = count_model_parameters(diffusion_model)
        print(f"#Params Diffusion Model: {trainable_params} (Total: {total_params})")
        # create optimizer from config
        partial_optimizer = instantiate(cfg.training.optimizer)
        optimizer = partial_optimizer(params=diffusion_model.parameters())

        if cfg.training.lr_scheduler is not None:
            partial_lr_scheduler = instantiate(cfg.training.lr_scheduler)
            lr_scheduler = partial_lr_scheduler(optimizer=optimizer)
        else:
            lr_scheduler = None

        # registrate model and optimizer in lite
        diffusion_model, optimizer = self.setup(diffusion_model, optimizer)

        # (optional) load pre-existing weights
        if cfg.training.checkpoint.restore:
            global_step, epoch = self.restore_checkpoint(
                diffusion_model.module,
                optimizer,
                cfg.training.checkpoint.path,
                lr_scheduler,
            )
        else:
            global_step, epoch = 0, 0

        # create exponential moving average (ema) model
        partial_ema = instantiate(cfg.training.ema)
        ema_model = partial_ema(model=diffusion_model.module)
        ema_model.optimization_step += global_step

        if cfg.training.checkpoint.restore:
            ema_model_ckpt = torch.load(
                os.path.join(
                    cfg.training.checkpoint.path,
                    "checkpoints",
                    "ema_averaged_model.ckpt",
                ),
                map_location="cpu",
            )
            ema_model.averaged_model.load_state_dict(ema_model_ckpt)

        if cfg.latent_diffusion:
            # create VQGAN encoder and decoder for training in its latent space
            latent_encoder = VQEncoderInterface(
                first_stage_config_path=os.path.join(
                    "./models", "autoencoder", "first_stage_config.yaml"
                ),
                encoder_state_dict_path=cfg.training.checkpoint.VQEncoder,
            )
            latent_decoder = VQDecoderInterface(
                first_stage_config_path=os.path.join(
                    "./models", "autoencoder", "first_stage_config.yaml"
                ),
                decoder_state_dict_path=cfg.training.checkpoint.VQDecoder,
            )

            # sampling and not during training)
            self.setup(latent_encoder)

            # set both into evaluation mode
            latent_encoder.eval()
            latent_decoder.eval()

            # display their number of parameters
            trainable_params, _, total_params = count_model_parameters(latent_encoder)
            print(
                f"#Params Latent Encoder Model: {trainable_params} (Total: {total_params})"
            )
            trainable_params, _, total_params = count_model_parameters(latent_decoder)
            print(
                f"#Params Latent Decoder Model: {trainable_params} (Total: {total_params})"
            )
        else:
            latent_encoder, latent_decoder = None, None

        # create dataset and dataloader from config
        dataset = instantiate(cfg.dataset.dataset)
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.training.num_workers,
            pin_memory=cfg.training.pin_memory,
        )
        # registrate dataloader in lite
        dataloader = self.setup_dataloaders(dataloader)

        # keep one batch fixed for visualisations
        x_visualisation = None
        context_visualisation = None

        # training loop
        metrics = {}
        loss_metric = tm.MeanMetric().cuda()
        while True:
            count = 0
            for x, context in dataloader:
                t_start = timer()
                count += x.shape[0]
                if x_visualisation is None:
                    x_visualisation = x.detach().clone()
                    context_visualisation = (
                        context.detach().clone()
                        if cfg.model.is_context_conditional
                        else None
                    )

                    if cfg.latent_diffusion:
                        with torch.no_grad():
                            sample_size = latent_encoder(x_visualisation[:1]).shape[-3:]
                    else:
                        sample_size = x_visualisation.shape[-3:]

                x = normalize_to_neg_one_to_one(x)

                if context is None or not cfg.model.is_context_conditional:
                    context = None
                    dropout_mask = None
                elif cfg.training.context_dropout > 0:
                    dropout_mask = torch.rand(len(x)) < cfg.training.context_dropout
                    if cfg.model.learn_empty_context and dropout_mask.sum() == 0:
                        dropout_mask[0] = True
                else:
                    dropout_mask = None

                if (
                    context is not None
                    and cfg.model.is_context_conditional
                    and cfg.training.context_permutation > 0
                ):
                    n_permuted = int(
                        cfg.training.batch_size * cfg.training.context_permutation
                    )
                    permutation = torch.randperm(n_permuted)
                    context[:n_permuted] = context[permutation]

                if global_step in cfg.training.steps_of_checkpoints:
                    self.save_checkpoint(
                        ema_model,
                        diffusion_model,
                        optimizer,
                        global_step,
                        epoch,
                        cfg.training.steps_of_checkpoints,
                        lr_scheduler,
                    )

                if global_step == 10:
                    self.create_and_save_sample_grid(
                        ema_model.averaged_model,
                        size=sample_size,
                        x=x_visualisation,
                        context=context_visualisation,
                        latent_decoder=latent_decoder,
                        save_path=ensure_path_join(
                            os.getcwd(), "samples", f"sample_{global_step:06d}.png"
                        ),
                    )

                # sampling
                if (
                    global_step % cfg.training.steps_between_sampling == 0
                    and global_step != 0
                ):
                    self.create_and_save_sample_grid(
                        ema_model.averaged_model,
                        size=sample_size,
                        x=x_visualisation,
                        context=context_visualisation,
                        latent_decoder=latent_decoder,
                        save_path=ensure_path_join(
                            os.getcwd(), "samples", f"sample_{global_step:06d}.png"
                        ),
                    )

                # training step
                diffusion_model.train()

                # map to latent_space
                if cfg.latent_diffusion:
                    with torch.no_grad():
                        x = latent_encoder(x)
                optimizer.zero_grad()

                loss = diffusion_model(
                    x, context=context.float(), dropout_mask=dropout_mask
                )
                self.backward(loss)
                optimizer.step()

                if lr_scheduler is not None:
                    lr_scheduler.step()

                # update ema model
                ema_model.step(diffusion_model.module)

                # aggregate losses from parallel devices
                loss_metric.update(loss)

                # monitoring and logging
                t_elapsed = timer() - t_start
                log_file_path = os.path.join(os.getcwd(), "main.log")
                print_status(
                    epoch=epoch,
                    global_step=global_step,
                    loss=loss_metric.compute().item(),
                    metrics=metrics,
                    time=t_elapsed,
                    lr=(
                        lr_scheduler.get_last_lr()[0]
                        if lr_scheduler is not None
                        else None
                    ),
                    log_file_path=(
                        log_file_path
                        if global_step % cfg.training.steps_between_logging == 0
                        else None
                    ),
                )

                # reset losses from parallel devices
                loss_metric.reset()

                global_step += 1
                if global_step >= cfg.training.steps:
                    self.save_checkpoint(
                        ema_model,
                        diffusion_model,
                        optimizer,
                        global_step,
                        epoch,
                        cfg.training.steps_of_checkpoints,
                        lr_scheduler,
                    )
                    self.create_and_save_sample_grid(
                        ema_model.averaged_model,
                        size=sample_size,
                        x=x_visualisation,
                        context=context_visualisation,
                        latent_decoder=latent_decoder,
                        save_path=ensure_path_join(
                            os.getcwd(), "samples", f"sample_{global_step:06d}.png"
                        ),
                    )
                    return
            epoch += 1


@hydra.main(config_path="configs", config_name="train_config", version_base=None)
def train(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    trainer = DiffusionTrainerLite(
        devices="auto", accelerator="gpu", precision=cfg.training.precision
    )
    trainer.run(cfg)


if __name__ == "__main__":
    train()
