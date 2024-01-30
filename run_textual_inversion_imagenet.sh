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

accelerate launch --multi_gpu --mixed_precision=$PRECISION examples/textual_inversion/textual_inversion.py \
	--pretrained_model_name_or_path=$MODEL_NAME \
	--train_data_dir=$DATA_DIR \
	--learnable_property=$PROPERTY \
	--train_batch_size=1 \
	--gradient_accumulation_steps=4 \
	--num_train_epochs=5 \
	--learning_rate=5.0e-04 \
	--scale_lr \
	--lr_scheduler="constant" \
	--lr_warmup_steps=0 \
	--mixed_precision=$PRECISION \
	--enable_xformers_memory_efficient_attention \
	--output_dir=$MODEL_DIR \
    --push_to_hub \
    --idx=$IDX

python examples/textual_inversion/inference.py \
	--model_path=textual_inversion_models \
	--prompt="a photo of" \
	--save_dir=textual_inversion_output \
	--num_images=10 \
    --idx=$IDX
