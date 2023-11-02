INSTANCE_DIR="simpsons"
TOKEN="sks"
TRAIN_PROMPT="a cartoon of a $TOKEN"
INFERENCE_PROMPT="a cartoon of a $TOKEN"
PRESERVE_PROMPT="a cartoon of a Simpsons character"  # For training with preservation loss

MODEL_NAME="stabilityai/stable-diffusion-2-1"
NUM_IMAGES=40
DATA_DIR="images"
OUTPUT_DIR="output"
MODEL_DIR="models"
PRECISION=bf16
PRESERVE_DIR=$OUTPUT_DIR/${INSTANCE_DIR}_preserve

accelerate launch --multi_gpu --mixed_precision=$PRECISION examples/dreambooth/train_dreambooth.py \
	--pretrained_model_name_or_path=$MODEL_NAME \
	--instance_data_dir=$DATA_DIR/$INSTANCE_DIR \
	--output_dir=$MODEL_DIR/${INSTANCE_DIR}_dreambooth \
	--instance_prompt="$TRAIN_PROMPT" \
	--resolution=768 \
	--gradient_accumulation_steps=1 \
	--learning_rate=5e-6 \
	--lr_scheduler="constant" \
	--lr_warmup_steps=0 \
	--max_train_steps=800 \
	--push_to_hub \
	--use_8bit_adam \
	--gradient_checkpointing \
	--mixed_precision=$PRECISION \
	--with_prior_preservation \
	--class_data_dir=$PRESERVE_DIR \
	--prior_loss_weight=1.0 \
	--class_prompt="$PRESERVE_PROMPT" \
	--num_class_images 200 \
	--train_batch_size=1

python examples/dreambooth/inference.py \
	--model_path=$MODEL_DIR/${INSTANCE_DIR}_dreambooth \
	--prompt="$INFERENCE_PROMPT" \
	--save_dir=$OUTPUT_DIR/${INSTANCE_DIR}_dreambooth \
	--num_images=$NUM_IMAGES


