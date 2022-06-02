gdown --id 1dntzjWFg7ufWaTaFy80nRz-Tu02xWZos -O tts_model.pth.tar
gdown --id 18CQ6G6tBEOfvCHlPqP8EBI4xWbrr9dBc -O config.json
gdown --id 1Ty5DZdOc0F7OTGj9oJThYbL5iVu_2G0K -O vocoder_model.pth.tar
gdown --id 1Rd0R_nRCrbjEdpOwq6XwZAktvugiBvmu -O config_vocoder.json
gdown --id 11oY3Tv0kQtxK_JPgxrfesa99maVXHNxU -O scale_stats.npy

pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html