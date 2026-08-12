"""SAM ViT-H architecture constants."""

# --- Encoder (ViT-H) ---
NUM_BLOCKS = 32
NUM_HEADS = 16
HEAD_DIM = 80
HIDDEN_DIM = 1280  # NUM_HEADS * HEAD_DIM
MLP_DIM = 5120     # HIDDEN_DIM * 4

WINDOW_SIZE = 14
GLOBAL_ATTN_INDEXES = (7, 15, 23, 31)

IMG_SIZE = 1024
PATCH_SIZE = 16
NUM_PATCHES_SIDE = 64  # IMG_SIZE // PATCH_SIZE
TOKENS_GLOBAL = 4096   # NUM_PATCHES_SIDE ** 2
TOKENS_WINDOW = 196    # WINDOW_SIZE ** 2

# After padding 64 → 70 (ceil(64/14)*14), 70/14 = 5 windows per side
PAD_SIZE = 70
NUM_WINDOWS_SIDE = 5
NUM_WINDOWS = 25  # NUM_WINDOWS_SIDE ** 2

NECK_OUT_CHANNELS = 256

# --- Mask Decoder ---
DECODER_LAYERS = 2
DECODER_HEADS = 8
DECODER_HEAD_DIM = 32
DECODER_DIM = 256

# --- SAM checkpoint URL ---
SAM_VIT_H_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
SAM_VIT_H_FILENAME = "sam_vit_h_4b8939.pth"

# --- COCO domain groups ---
DOMAIN_GROUPS = {
    "general": None,
    "people": ["person"],
    "vehicles": ["car", "truck", "bus", "motorcycle", "bicycle", "train", "airplane", "boat"],
    "animals": ["dog", "cat", "horse", "bird", "cow", "sheep", "elephant", "bear", "zebra", "giraffe"],
    "indoor": ["chair", "couch", "bed", "dining table", "toilet", "tv", "laptop", "microwave", "oven", "refrigerator"],
    "food": ["banana", "apple", "pizza", "cake", "sandwich", "orange", "broccoli", "carrot", "hot dog", "donut"],
    "sports": ["sports ball", "tennis racket", "baseball bat", "baseball glove", "skateboard", "surfboard", "snowboard", "skis", "frisbee", "kite"],
    "street": ["traffic light", "stop sign", "fire hydrant", "parking meter", "bench"],
    "kitchen": ["bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl"],
    "electronics": ["tv", "laptop", "mouse", "remote", "keyboard", "cell phone"],
    "outdoor": ["bench", "potted plant", "vase", "clock", "umbrella", "backpack", "handbag", "suitcase"],
}

ALL_DOMAIN_NAMES = list(DOMAIN_GROUPS.keys())

AVAILABLE_MODES = ("profile", "freeze", "eval")
