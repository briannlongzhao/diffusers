INSTANCE_DIR="simpsons"
TOKEN="<new1>"
TRAIN_PROMPT="a cartoon of a $TOKEN"
INFERENCE_PROMPT="a cartoon of a $TOKEN"
PRESERVE_PROMPT="a cartoon of a Simpsons character"  # For training with preservation loss

MODEL_NAME="stabilityai/stable-diffusion-2-1"
NUM_IMAGES=40
DATA_DIR="images"
OUTPUT_DIR="output"
MODEL_DIR="models"
PRECISION=no  # Currently not supporting mixed precision
PRESERVE_DIR=$OUTPUT_DIR/${INSTANCE_DIR}_preserve_real

pip install clip-retrieval --no-dependencies
python examples/custom_diffusion/retrieve.py \
	--class_prompt="${INSTANCE_DIR}" \
	--class_data_dir="$PRESERVE_DIR" \
	--num_class_images 200

accelerate launch --mixed_precision=$PRECISION examples/custom_diffusion/train_custom_diffusion.py \
	--pretrained_model_name_or_path=$MODEL_NAME \
	--instance_data_dir=$DATA_DIR/$INSTANCE_DIR \
	--output_dir=$MODEL_DIR/${INSTANCE_DIR}_custom_diffusion \
	--instance_prompt="$TRAIN_PROMPT" \
	--resolution=768 \
	--gradient_accumulation_steps=1 \
	--learning_rate=1e-5 \
	--lr_scheduler="constant" \
	--lr_warmup_steps=0 \
	--max_train_steps=250 \
	--scale_lr \
	--hflip \
	--modifier_token=$TOKEN \
	--push_to_hub \
	--use_8bit_adam \
	--gradient_checkpointing \
	--mixed_precision=$PRECISION \
	--with_prior_preservation \
	--class_data_dir=$PRESERVE_DIR \
	--real_prior \
	--prior_loss_weight=1.0 \
	--class_prompt="$PRESERVE_PROMPT" \
	--num_class_images 200 \
	--no_safe_serialization \
	--train_batch_size=2

python examples/custom_diffusion/inference.py \
	--pretrained_model_name_or_path=$MODEL_NAME \
	--model_path=$MODEL_DIR/${INSTANCE_DIR}_custom_diffusion \
	--prompt="$INFERENCE_PROMPT" \
	--token="$TOKEN" \
	--save_dir=$OUTPUT_DIR/${INSTANCE_DIR}_custom_diffusion \
	--num_images=$NUM_IMAGES

