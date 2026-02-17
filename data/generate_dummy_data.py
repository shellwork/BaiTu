import sys
import os

# 将项目根目录添加到 python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import generate_dummy_data
from config import Config

if __name__ == "__main__":
    print("Generating dummy data...")
    generate_dummy_data(num_samples=1000, output_path=Config.DATA_PATH)
    print("Done.")
