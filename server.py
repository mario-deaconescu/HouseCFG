import json
from contextlib import asynccontextmanager
from typing import Generator
import asyncio

import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRoute
from fastapi.openapi.utils import get_openapi
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.backend.types import PlanSerializer
from src.backend.models import RoomTypeInputParameters
from src.backend.types import NumpyArray1DSerializer
from src.backend.models import BubblesInputParameters, BaseInputParameters
from src.backend.types import NumpyArray3DSerializer
from src.gaussian_noise import BetaSchedule, ModelMeanType, ModelVarType
from src.respace import SpacedDiffusion, space_timesteps
from src.rplan.types import ImagePlan
from src.rplan_masks.karras.cfg import CFGUnetWithScale, CFGUnet
from src.rplan_masks.openai.unet import UNetModel


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', validate_default=False)
    bubbles_old_model_path: str = "./models/bubbles/model_9_1000.pt"
    room_types_model_path: str = "./models/room_types_cfg/model_15_1500.pt"
    bubbles_model_1_path: str = './models/combined/model_15_0.pt'
    bubbles_model_2_path: str = './models/combined/model_19_1000.pt'


class ModelsCache:

    def _load_model(self, model, model_path: str, additional_state = None):
        state_dict = torch.load(model_path, map_location=self.device)
        if additional_state is None:
            additional_state = {}
        for key, value in additional_state.items():
            state_dict[key] = value
        model.load_state_dict(state_dict)
        full_model = CFGUnetWithScale(model).to(self.device)
        # full_model.eval()
        return full_model

    def __init__(self, settings: Settings, device: torch.device):
        self.device = device
        room_types_model = CFGUnet(dim=64, channels=4, out_dim=3, cond_drop_prob=0).to(device)
        bubbles_model_1 = CFGUnet(dim=64, channels=6, out_dim=3, cond_drop_prob=0, bubble_dim=1).to(device)
        bubbles_model_2 = CFGUnet(dim=64, channels=6, out_dim=3, cond_drop_prob=0, bubble_dim=1).to(device)

        self._bubbles_model_1 = self._load_model(bubbles_model_1, settings.bubbles_model_1_path, {"null_bubble_diagram": torch.zeros_like(bubbles_model_1.null_bubble_diagram, device=device)})
        self._bubbles_model_2 = self._load_model(bubbles_model_2, settings.bubbles_model_2_path, {"null_bubble_diagram": torch.zeros_like(bubbles_model_1.null_bubble_diagram, device=device)})
        self._room_types_model = self._load_model(room_types_model, settings.room_types_model_path)
        self._bubbles_old_model = None

        self._settings = settings

    def bubbles_1_model(self):
        return self._bubbles_model_1

    def bubbles_2_model(self):
        return self._bubbles_model_2

    def room_types_model(self):
        return self._room_types_model

    def bubbles_old_model(self):
        if self._bubbles_old_model is not None:
            return self._bubbles_old_model

        bubbles_old_model = UNetModel(image_size=64, in_channels=6, model_channels=192, out_channels=3, num_res_blocks=3,
                                  attention_resolutions=[32, 16, 8], num_head_channels=64, resblock_updown=True,
                                  use_scale_shift_norm=True,
                                  use_new_attention_order=True, use_fp16=False, dropout=0.1, cond_drop_prob=1).to(
            self.device)

        self._bubbles_old_model = self._load_model(bubbles_old_model, self._settings.bubbles_old_model_path)

        return self._bubbles_old_model

def make_diffusion(num_steps: int, device: torch.device):
    T = 1000

    diffusion = SpacedDiffusion(space_timesteps(T, f'ddim{num_steps}'), T, BetaSchedule.COSINE,
                                ModelMeanType.EPSILON,
                                ModelVarType.FIXED_SMALL,
                                device=device)
    return diffusion


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    models_cache = ModelsCache(settings, device)
    yield {
        "settings": settings,
        "device": device,
        "models_cache": models_cache,
        "diffusion_cache": {
            100: make_diffusion(100, device),
        },
    }
    # Cleanup code can be added here if needed


app = FastAPI(lifespan=lifespan)


def make_kwargs(parameters: BaseInputParameters, device: torch.device):
    return {
        'masks': torch.tensor(NumpyArray3DSerializer.deserialize(parameters.mask), device=device,
                              dtype=torch.float32).unsqueeze(1).repeat(parameters.num_samples, 1, 1, 1),
        'cond_scale': parameters.condition_scale,
        'rescaled_phi': parameters.rescaled_phi,
    }


def sample(diffusion: SpacedDiffusion, model, parameters: BaseInputParameters, model_kwargs: dict,
           device: torch.device) -> Generator[tuple[np.ndarray, bool], None, None]:
    num_samples = parameters.num_samples
    mask_size = 64
    ddim = parameters.ddim
    shape = (num_samples, 3, mask_size, mask_size)

    generator = diffusion.ddim_sample_loop_progressive if ddim else diffusion.p_sample_loop_progressive

    total_steps = diffusion.num_timesteps
    for step, (samples, pred_x_0, _) in enumerate(generator(model, shape, model_kwargs=model_kwargs)):
        if step == total_steps - 1:
            final = True
            output = samples.detach().cpu().numpy()
        else:
            final = False
            output = pred_x_0.detach().cpu().numpy()
        yield output, final


def response_generator(sample_generator: Generator[tuple[np.ndarray, bool], None, None],
                       parameters: BaseInputParameters) -> Generator[str, None, None]:
    for samples, final in sample_generator:
        plans = [ImagePlan(walls=walls, image=rooms, door_image=doors) for
                 rooms, walls, doors in samples]
        original_images = [plan.to_image() for plan in plans]
        processed_plans = [plan.to_plan(thin_walls=parameters.skeletonize, simplify=parameters.simplify,
                                        use_felzenszwalb=parameters.felzenszwalb, target_size=(256, 256), image_only=(not final)) for
                           plan in plans]
        images = [image_plan.to_image() for _, image_plan in processed_plans]
        output_plans = [PlanSerializer.serialize(plan) if plan is not None else None for plan, _ in processed_plans]
        data = {
            'final': final,
            'images': [NumpyArray3DSerializer.serialize(image) for image in images],
            'original_images': [NumpyArray3DSerializer.serialize(image) for image in original_images],
            'plans': output_plans,
        }
        yield f"data: {json.dumps(data)}\n\n"

def generate_bubbles_helper(input_params: BubblesInputParameters, request: Request, model):
    device = request.state.device
    diffusion_cache: dict[int, SpacedDiffusion] = request.state.diffusion_cache

    diffusion = diffusion_cache.get(input_params.num_steps, make_diffusion(input_params.num_steps, device))
    if input_params.num_steps not in diffusion_cache:
        diffusion_cache[input_params.num_steps] = diffusion
    kwargs = make_kwargs(input_params, device)
    kwargs['bubbles'] = torch.tensor(NumpyArray3DSerializer.deserialize(input_params.bubbles), device=device,
                                     dtype=torch.float32).unsqueeze(1).repeat(input_params.num_samples, 1, 1,
                                                                              1) if input_params.bubbles is not None else None
    generator = sample(diffusion, model, input_params, kwargs, device)
    response = response_generator(generator, input_params)
    return StreamingResponse(response, media_type="text/event-stream")


@app.post("/generate/room_types")
def generate_room_types(input_params: RoomTypeInputParameters, request: Request):
    device = request.state.device
    diffusion_cache: dict[int, SpacedDiffusion] = request.state.diffusion_cache

    diffusion = diffusion_cache.get(input_params.num_steps, make_diffusion(input_params.num_steps, device))
    if input_params.num_steps not in diffusion_cache:
        diffusion_cache[input_params.num_steps] = diffusion
    kwargs = make_kwargs(input_params, device)
    kwargs['room_types'] = torch.tensor(NumpyArray1DSerializer.deserialize(input_params.room_types), device=device,
                                        dtype=torch.float32).unsqueeze(0).repeat(input_params.num_samples,
                                                                                 1) if input_params.room_types is not None else None
    generator = sample(diffusion, request.state.models_cache.room_types_model(), input_params, kwargs, device)
    response = response_generator(generator, input_params)
    return StreamingResponse(response, media_type="text/event-stream")


@app.post("/generate/bubbles")
def generate_bubbles(input_params: BubblesInputParameters, request: Request):
    return generate_bubbles_helper(input_params, request, request.state.models_cache.bubbles_1_model())


@app.post("/generate/bubbles_old")
def generate_bubbles_old(input_params: BubblesInputParameters, request: Request):
    return generate_bubbles_helper(input_params, request, request.state.models_cache.bubbles_old_model())


@app.post("/generate/bubbles_new")
def generate_bubbles_new(input_params: BubblesInputParameters, request: Request):
    return generate_bubbles_helper(input_params, request, request.state.models_cache.bubbles_2_model())


async def health_check():
    while True:
        status = {"status": "ok"}
        await asyncio.sleep(1)
        yield f"data: {json.dumps(status)}\n\n"
        await asyncio.sleep(1)

@app.get("/health")
async def health():
    return StreamingResponse(health_check(), media_type="text/event-stream")


def use_route_names_as_operation_ids(app: FastAPI) -> None:
    """
    Simplify operation IDs so that generated API clients have simpler function
    names.

    Should be called only after all routes have been added.
    """
    for route in app.routes:
        if isinstance(route, APIRoute):
            route.operation_id = route.name  # in this case, 'read_items'


use_route_names_as_operation_ids(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # Generate the schema for BaseRoom and add it
    base_schema = BaseInputParameters.model_json_schema(ref_template="#/components/schemas/{model}")
    openapi_schema["components"]["schemas"]["BaseInputParameters"] = base_schema

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=8000)
