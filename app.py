import torch
import gradio as gr
from PIL import Image as PILImage
import torchvision.transforms as transforms

from model import MaskedAutoencoder
from utils import make_masked_image, reconstruct_image, denormalise
from config import IMG_SIZE

device = torch.device('cpu')

model = MaskedAutoencoder()
ckpt  = torch.load('mae_best.pth', map_location='cpu')

# Strip DataParallel 'module.' prefix if present
state_dict = ckpt['model_state_dict']
if any(k.startswith('module.') for k in state_dict.keys()):
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

model.load_state_dict(state_dict)
model.eval()
model = model.half()  # float16 to cut memory usage in half
print(f'Model loaded — epoch {ckpt["epoch"]}, val_loss={ckpt["val_loss"]:.6f}')

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def run_mae(pil_image, mask_ratio):
    img_tensor = transform(pil_image.convert('RGB'))
    img_batch  = img_tensor.unsqueeze(0).half()  # match model precision

    with torch.no_grad():
        preds, masks = model(img_batch, mask_ratio=mask_ratio)

    pred = preds.float().cpu().squeeze(0)
    mask = masks.float().cpu().squeeze(0)
    img  = img_tensor.float()

    masked_arr = make_masked_image(img, mask)
    recon_arr  = reconstruct_image(pred, img, mask)
    orig_arr   = denormalise(img)

    return (PILImage.fromarray(masked_arr),
            PILImage.fromarray(recon_arr),
            PILImage.fromarray(orig_arr))


with gr.Blocks(title='Masked Autoencoder Demo') as demo:
    gr.Markdown(
        '# Masked Autoencoder (MAE)\n'
        'Upload an image, adjust the masking ratio to reconstruct missing patches.\n\n'
    )

    with gr.Row():
        with gr.Column():
            img_input   = gr.Image(type='pil', label='Upload Image')
            mask_slider = gr.Slider(minimum=0.1, maximum=0.9, value=0.75,
                                    step=0.05, label='Masking Ratio')
            run_btn     = gr.Button('Run', variant='primary')

        with gr.Column():
            out_masked = gr.Image(type='pil', label='Masked Input')
            out_recon  = gr.Image(type='pil', label='Reconstruction')
            out_orig   = gr.Image(type='pil', label='Original')

    run_btn.click(
        fn      = run_mae,
        inputs  = [img_input, mask_slider],
        outputs = [out_masked, out_recon, out_orig]
    )

demo.launch()