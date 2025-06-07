import enum
import math
from typing import Callable, TypedDict, Optional, Any

import torch
from tqdm import tqdm

class BetaSchedule(enum.Enum):
    LINEAR = enum.auto()
    COSINE = enum.auto()


class ModelMeanType(enum.Enum):
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()


class PMeanVariance(TypedDict):
    mean: torch.Tensor
    variance: torch.Tensor
    log_variance: torch.Tensor
    pred_x_0: torch.Tensor
    additional_output: Optional[Any]


def mean_flat(tensor, mask=None):
    if mask is None:
        return tensor.mean(dim=list(range(1, len(tensor.shape))))
    mask = mask.reshape(mask.shape + (1,) * (len(tensor.shape) - len(mask.shape)))
    return (tensor * mask).mean(dim=list(range(1, len(tensor.shape)))) / mask.sum(dim=list(range(1, len(tensor.shape))))

def expand(tensor, shape):
    return tensor.reshape(tensor.shape + (1,) * (len(shape) - len(tensor.shape))).expand(shape)

class GaussianDiffusion:
    LIN_BETA_START = 0.0001
    LIN_BETA_END = 0.02
    COS_BETA_1 = 0.008
    COS_BETA_2 = 1.008

    def _betas_for_alpha_bar(self, num_timesteps: int, alpha_bar: Callable[[float], float],
                             max_beta=0.999) -> torch.Tensor:
        betas = []
        for i in range(num_timesteps):
            t1 = i / num_timesteps
            t2 = (i + 1) / num_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return torch.tensor(betas, dtype=torch.float32, device=self.device)

    def _beta_schedule(self, schedule: BetaSchedule, num_timesteps: int):
        if schedule == BetaSchedule.LINEAR:
            scale = 1000 / num_timesteps
            beta_start = scale * self.LIN_BETA_START
            beta_end = scale * self.LIN_BETA_END
            return torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32, device=self.device)
        elif schedule == BetaSchedule.COSINE:
            return self._betas_for_alpha_bar(
                num_timesteps,
                lambda t: math.cos((t + self.COS_BETA_1) / self.COS_BETA_2 * math.pi / 2) ** 2,
            )
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

    def scale_timesteps(self, t: torch.Tensor) -> torch.Tensor:
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def unscale_timesteps(self, t: torch.Tensor) -> torch.Tensor:
        if self.rescale_timesteps:
            return (t * self.num_timesteps / 1000.0).long()
        return t

    def predict_x_0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        assert x_t.shape == eps.shape
        return expand(self.sqrt_recip_alphas_cumprod[t], x_t.shape) * x_t - expand(self.sqrt_recipm1_alphas_cumprod[t], x_t.shape) * eps

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            expand(self.sqrt_recip_alphas_cumprod[t], x_t.shape) * x_t
            - pred_xstart
        ) / expand(self.sqrt_recipm1_alphas_cumprod[t], x_t.shape)

    def __init__(self, num_timesteps: int, schedule: BetaSchedule, model_mean_type: ModelMeanType,
                 model_var_type: ModelVarType, device: torch.device = torch.device('cpu'),
                 rescale_timesteps: bool = False, force_betas: Optional[torch.Tensor] = None):
        self.device = device
        self.rescale_timesteps = rescale_timesteps
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        if force_betas is not None:
            self.betas = force_betas
        else:
            self.betas = self._beta_schedule(schedule, num_timesteps)
        assert (self.betas > 0).all() and (self.betas < 1).all()

        self.num_timesteps = len(self.betas)

        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=self.device), self.alphas_cumprod[:-1]])
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0], device=self.device)])
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_1m_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1.0)

        self.posterior_variance = (
                self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        self.posterior_log_variance_clipped = torch.log(
            torch.concat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]])
        )

        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> tuple[
        torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x_0, device=self.device)
        else:
            if noise.shape != x_0.shape:
                raise ValueError("Noise must have the same shape as x_0")
            if noise.device != self.device:
                noise = noise.to(self.device)
        if t.device != self.device:
            t = t.to(self.device)

        return expand(self.sqrt_alphas_cumprod[t], x_0.shape) * x_0 + expand(self.sqrt_1m_alphas_cumprod[t], x_0.shape) * noise, noise

    def q_posterior_mean_variance(self, x_0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        if x_0.shape != x_t.shape:
            print(x_0.shape, x_t.shape)
        assert x_0.shape == x_t.shape
        if x_0.device != self.device:
            x_0 = x_0.to(self.device)
        if t.device != self.device:
            t = t.to(self.device)

        posterior_mean = expand(self.posterior_mean_coef1[t], x_t.shape) * x_0 + expand(self.posterior_mean_coef2[t], x_t.shape) * x_t
        posterior_variance = expand(self.posterior_variance[t], x_t.shape)
        posterior_log_variance_clipped = expand(self.posterior_log_variance_clipped[t], x_t.shape)
        assert posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] == \
               x_0.shape[0]
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, x: torch.Tensor, t: torch.Tensor, clip_denoised: bool = True,
                        model_kwargs=None) -> PMeanVariance:
        if model_kwargs is None:
            model_kwargs = {}

        num_batches = x.shape[0]
        assert t.shape == (num_batches,)

        model_output: torch.Tensor = model(x, self.scale_timesteps(t), **model_kwargs)
        if model_kwargs.get('additional_loss', None) is not None:
            model_output, additional_output = model_output
        else:
            additional_output = None

        if model_kwargs.get('filter_output', None) is not None:
            model_output = model_kwargs['filter_output'](model_output, t)

        if 'prediction_branch' in model_kwargs:
            assert t.shape[0] == 1
            prediction_branch_output = model_kwargs['prediction_branch'](model_output, t)
            model_output, _ = model_output
        else:
            prediction_branch_output = None

        if self.model_var_type == ModelVarType.FIXED_SMALL:
            model_variance, model_log_variance = self.posterior_variance, self.posterior_log_variance_clipped
        elif self.model_var_type == ModelVarType.FIXED_LARGE:
            model_variance, model_log_variance = torch.concat(
                [self.posterior_variance[1], self.betas[1:]]), torch.log(
                torch.concat([self.posterior_variance[1], self.betas[1:]]))
        else:
            raise ValueError(f"Unknown model variance type: {self.model_var_type}")
        model_variance = expand(model_variance[t], x.shape)
        model_log_variance = expand(model_log_variance[t], x.shape)

        def process_x_0(_x):
            if clip_denoised:
                return torch.clamp(_x, -1, 1)
            return _x

        if self.model_mean_type == ModelMeanType.START_X:
            pred_x_0 = process_x_0(model_output)
        elif prediction_branch_output is not None:
            pred_x_0 = process_x_0(prediction_branch_output)
        else:
            pred_x_0 = process_x_0(self.predict_x_0_from_eps(x, t, model_output))
        model_mean, _, _ = self.q_posterior_mean_variance(pred_x_0, x, t)

        assert model_mean.shape == model_log_variance.shape == pred_x_0.shape == x.shape

        return {
            'mean': model_mean,
            'variance': model_variance,
            'log_variance': model_log_variance,
            'pred_x_0': pred_x_0,
            'additional_output': additional_output,
        }

    def p_sample(self, model, x: torch.Tensor, t: torch.Tensor, clip_denoised: bool = True, model_kwargs=None) -> tuple[
        torch.Tensor, torch.Tensor, Optional[Any]]:
        if model_kwargs is None:
            model_kwargs = {}

        out = self.p_mean_variance(model, x, t, clip_denoised, model_kwargs)
        noise = torch.randn_like(x, device=self.device)
        nonzero_mask = (
            (t != 0).float().view(-1, *[1] * (len(x.shape) - 1))
        )
        sample = out['mean'] + nonzero_mask * torch.exp(0.5 * out['log_variance']) * noise
        return sample, out['pred_x_0'], out['additional_output']

    def p_sample_loop(self, model, shape, noise: Optional[torch.Tensor] = None, clip_denoised: bool = True,
                      model_kwargs=None, progress: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        final = None
        model.eval()
        for sample, _, additional_sample in tqdm(self.p_sample_loop_progressive(model, shape, noise, clip_denoised, model_kwargs, progress)):
            final = sample, additional_sample
        model.train()
        return final

    def p_sample_loop_progressive(self, model, shape, noise: Optional[torch.Tensor] = None, clip_denoised: bool = True,
                                  model_kwargs=None, progress: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is not None:
            assert shape == noise.shape
            x = noise
        else:
            x = torch.randn(*shape, device=self.device)

        indices = reversed(range(self.num_timesteps))

        if progress:
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = torch.tensor([i] * shape[0], device=self.device)
            with torch.no_grad():
                out = self.p_sample(model, x, t, clip_denoised, model_kwargs)
                yield out
                x = out[0]

    def training_losses(self, model, x_0: torch.Tensor, t: torch.Tensor, model_kwargs=None,
                        noise: Optional[torch.Tensor] = None):
        if model_kwargs is None:
            model_kwargs = {}

        additional_loss: Optional[Callable] = model_kwargs.get("additional_loss", None)

        if noise is not None:
            assert x_0.shape == noise.shape
        x_t, noise = self.q_sample(x_0, t, noise)

        terms = {}

        model_output = model(x_t, self.scale_timesteps(t), **model_kwargs)
        if additional_loss is not None:
            model_output, additional_output = model_output
        else:
            additional_output = None

        target = {
            ModelMeanType.START_X: x_0,
            ModelMeanType.EPSILON: noise,
        }[self.model_mean_type]
        assert model_output.shape == target.shape == x_0.shape
        if 'src_key_padding_mask' in model_kwargs:
            mask = ~model_kwargs['src_key_padding_mask']
        else:
            mask = None
        if self.model_mean_type == ModelMeanType.EPSILON and 'custom_eps_loss' in model_kwargs:
            terms['loss'] = model_kwargs['custom_eps_loss'](model_output, target, x_0, x_t, t, **model_kwargs)
        else:
            terms["mse"] = mean_flat((target - model_output) ** 2, mask)
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
            if additional_loss is not None and additional_output is not None:
                terms["additional_loss"] = additional_loss(additional_output)
                terms["loss"] = terms["loss"] + terms["additional_loss"]

        return terms

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
        )

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_x_0"])

        alpha_bar = expand(self.alphas_cumprod[t], x.shape)
        alpha_bar_prev = expand(self.alphas_cumprod_prev[t], x.shape)
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = torch.randn_like(x)
        mean_pred = (
            out["pred_x_0"] * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return sample, out["pred_x_0"], out["additional_output"]

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample, _, additional_sample in tqdm(self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        )):
            final = sample, additional_sample
        return final

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        model.eval()

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out[0]


        model.train()

