INSTANCE_DIR="simpsons"
TOKEN="<${INSTANCE_DIR}>"
INFERENCE_PROMPT="a $TOKEN"
PROPERTY=object  # object or style

MODEL_NAME="stabilityai/stable-diffusion-2-1"
PRECISION=bf16
NUM_IMAGES=40
DATA_DIR="images"
OUTPUT_DIR="output"
MODEL_DIR="models"

accelerate launch --multi_gpu --mixed_precision=$PRECISION examples/textual_inversion/textual_inversion.py \
	--pretrained_model_name_or_path=$MODEL_NAME \
	--train_data_dir=$DATA_DIR/$INSTANCE_DIR \
	--learnable_property=$PROPERTY \
	--placeholder_token=$TOKEN \
	--initializer_token="${INSTANCE_DIR}" \
	--resolution=768 \
	--train_batch_size=1 \
	--gradient_accumulation_steps=4 \
	--max_train_steps=3000 \
	--learning_rate=5.0e-04 \
	--scale_lr \
	--lr_scheduler="constant" \
	--lr_warmup_steps=0 \
	--push_to_hub \
	--mixed_precision=$PRECISION \
	--enable_xformers_memory_efficient_attention \
	--output_dir=$MODEL_DIR/${INSTANCE_DIR}_textual_inversion

python examples/textual_inversion/inference.py \
	--model_path=$MODEL_DIR/${INSTANCE_DIR}_textual_inversion \
	--prompt="$INFERENCE_PROMPT" \
	--save_dir=$OUTPUT_DIR/${INSTANCE_DIR}_textual_inversion \
	--num_images=$NUM_IMAGES
