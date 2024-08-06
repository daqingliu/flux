import os
import re
import time
from glob import iglob
from io import BytesIO

import torch
from einops import rearrange
from PIL import ExifTags, Image

from flux.cli import SamplingOptions
from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from flux.util import (
    load_ae,
    load_clip,
    load_flow_model,
    load_t5,
)


def get_models(name: str, device: torch.device):
    t5 = load_t5(device, max_length=512)
    clip = load_clip(device)
    model = load_flow_model(name, device)
    ae = load_ae(name, device)
    return model, ae, t5, clip


@torch.inference_mode()
def main(
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: str = "output",
):
    torch_device = torch.device(device)
    name = 'flux-dev'

    model, ae, t5, clip = get_models(
        name,
        device=torch_device,
    )

    # allow for packing and conversion to latent space
    width = 1024
    height = 1024
    num_steps = 50
    guidance = 3.5
    seed = 0
    save_samples = True

    prompt = (
        "a photo of a forest with mist swirling around the tree trunks. The word "
        '"FLUX" is painted over it in big, red brush strokes with visible texture'
    )

    output_name = os.path.join(output_dir, "img_{idx}.jpg")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        idx = 0
    else:
        fns = [fn for fn in iglob(output_name.format(idx="*")) if re.search(r"img_[0-9]\.jpg$", fn)]
        if len(fns) > 0:
            idx = max(int(fn.split("_")[-1].split(".")[0]) for fn in fns) + 1
        else:
            idx = 0

    rng = torch.Generator(device="cpu")

    opts = SamplingOptions(
        prompt=prompt,
        width=width,
        height=height,
        num_steps=num_steps,
        guidance=guidance,
        seed=seed,
    )

    opts.seed = rng.seed()
    print(f"Generating '{opts.prompt}' with seed {opts.seed}")
    t0 = time.perf_counter()

    # prepare input
    x = get_noise(
        1,
        opts.height,
        opts.width,
        device=torch_device,
        dtype=torch.bfloat16,
        seed=opts.seed,
    )
    # divide pixel space by 16**2 to acocunt for latent space conversion
    timesteps = get_schedule(
        opts.num_steps,
        (x.shape[-1] * x.shape[-2]) // 4,
        shift=True,
    )

    inp = prepare(t5=t5, clip=clip, img=x, prompt=opts.prompt)

    # denoise initial noise
    x = denoise(model, **inp, timesteps=timesteps, guidance=opts.guidance)

    # decode latents to pixel space
    x = unpack(x.float(), opts.height, opts.width)
    with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
        x = ae.decode(x)

    t1 = time.perf_counter()

    fn = output_name.format(idx=idx)
    print(f"Done in {t1 - t0:.1f}s.")
    # bring into PIL format and save
    x = x.clamp(-1, 1)
    x = rearrange(x[0], "c h w -> h w c")

    img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

    buffer = BytesIO()
    img.save(buffer, format="jpeg", quality=95, subsampling=0)

    img_bytes = buffer.getvalue()
    if save_samples:
        print(f"Saving {fn}")
        with open(fn, "wb") as file:
            file.write(img_bytes)
        idx += 1


if __name__ == "__main__":
    main()
