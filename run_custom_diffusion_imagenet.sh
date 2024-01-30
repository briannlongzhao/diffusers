INSTANCE_DIR="simpsons"
TOKEN="<${INSTANCE_DIR}>"
INFERENCE_PROMPT="a $TOKEN"
PROPERTY=object  # object or style

IDX=0

MODEL_NAME="stabilityai/stable-diffusion-2-1"
PRECISION=fp16
NUM_IMAGES=40
DATA_DIR="/lab/tmpig7b/u/brian-data/imagenet/images/train_small"
OUTPUT_DIR="output"
MODEL_DIR="textual_inversion_models"

accelerate launch --mixed_precision=$PRECISION examples/custom_diffusion/train_custom_diffusion.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --instance_data_dir=$DATA_DIR \
    --output_dir=custom_diffusion_models \
    --instance_prompt="a photo of <new1>" \
    --gradient_accumulation_steps=1 \
    --learning_rate=1e-5 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --num_train_epochs=5 \
    --scale_lr \
    --hflip \
    --modifier_token="<new1>" \
    --use_8bit_adam \
    --gradient_checkpointing \
    --mixed_precision=$PRECISION \
    --push_to_hub \
    --no_safe_serialization \
    --train_batch_size=1 \
    --idx=$IDX

python examples/custom_diffusion/inference.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --model_path=custom_diffusion_models \
    --prompt="a photo of <new1>" \
    --token="<new1>" \
    --save_dir=custom_diffusion_output \
    --num_images=10 \
    --idx=0
